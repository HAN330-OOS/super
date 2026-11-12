"""Read-only satellite extraction for the existing formal test set.

No training, satellite rematching, ground observations, or validation metrics.
Source CSI is never scaled or clipped. FY-4A is the stored LR GHI; FY-3C is the
stored HR GHI. Bicubic and DRCT use the existing aligned clear-sky GHI array.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import zipfile

import numpy as np

UTC = timezone.utc
BEIJING = timezone(timedelta(hours=8))
FIELDS = ('sample_id', 'date', 'fy3_time', 'fy4_time', 'row', 'col',
          'center_lon', 'center_lat', 'npz_file', 'array_index')
SOURCE_KEYS = ('fy3', 'fy4', 'hr_csi', 'lr_csi')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_value(value):
    return float(value) if np.isfinite(value) else None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    st = Path(path).stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def read_csv(path, required=FIELDS):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(set(required).issubset(reader.fieldnames or []), f'Missing CSV fields: {path}')
        return list(reader)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_config(path=None):
    path = Path(path or Path(__file__).with_name('config.json')).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    out = Path(cfg['output_dir'])
    cfg['_output'] = str(out.resolve() if out.is_absolute() else (path.parent / out).resolve())
    cfg['_path'] = str(path)
    return cfg


def iso(dt):
    return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def utc_datetime(text):
    dt = datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    require(dt.tzinfo is not None, f'Timestamp must include timezone: {text}')
    return dt.astimezone(UTC)


def date_hhmm(date, hhmm, tz):
    require(len(hhmm) == 4 and hhmm.isdigit(), f'Invalid HHMM: {date}/{hhmm}')
    return datetime.strptime(date + hhmm, '%Y%m%d%H%M').replace(tzinfo=tz)


def npz_path(cfg, row):
    require(len(row['date']) == 8 and row['date'].isdigit(), 'Invalid sample date')
    name = row['npz_file']
    require(Path(name).name == name and '/' not in name and '\\' not in name
            and name.startswith(row['date'] + '_') and name.endswith('.npz'), 'Invalid NPZ filename')
    return Path(cfg['samples_root']) / row['date'] / name


class PatchReader:
    """Reuse the formal exporter's ZIP_STORED NPY memory-map technique."""
    def __init__(self, capacity=2):
        self.capacity, self.cache = capacity, OrderedDict()

    @staticmethod
    def stored_array(path, entry):
        with Path(path).open('rb') as f:
            f.seek(entry.header_offset)
            header = f.read(30)
            require(len(header) == 30 and header[:4] == b'PK\x03\x04', f'Bad ZIP header: {path}')
            name_len, extra_len = struct.unpack_from('<HH', header, 26)
            f.seek(name_len + extra_len, 1)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f'Unsupported NPY header: {version}')
            offset = f.tell()
        require(not dtype.hasobject, 'Object arrays are not permitted')
        require(int(np.prod(shape)) * dtype.itemsize + offset <= entry.header_offset + 30
                + name_len + extra_len + entry.file_size, 'NPY payload exceeds ZIP entry')
        return np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=shape,
                         order='F' if fortran else 'C')

    @staticmethod
    def close_arrays(arrays):
        for a in arrays.values():
            if isinstance(a, np.memmap):
                a._mmap.close()

    def arrays(self, path):
        path = str(Path(path).resolve())
        if path not in self.cache:
            arrays = {}
            try:
                with zipfile.ZipFile(path) as z:
                    for member in z.infolist():
                        require(member.filename.endswith('.npy'), 'Unexpected NPZ member')
                        key = member.filename[:-4]
                        if member.compress_type == zipfile.ZIP_STORED:
                            arrays[key] = self.stored_array(path, member)
                        else:
                            with z.open(member) as f:
                                arrays[key] = np.load(f, allow_pickle=False)
                n = len(arrays['fy3'])
                for key in SOURCE_KEYS:
                    size = 32 if key in ('fy4', 'lr_csi') else 128
                    require(arrays[key].shape == (n, size, size) and arrays[key].dtype == np.float32,
                            f'Wrong {key} shape/dtype in {path}')
                require(1 <= n <= 64, f'Invalid NPZ sample count: {path}')
            except BaseException:
                self.close_arrays(arrays)
                raise
            if len(self.cache) >= self.capacity:
                _, removed = self.cache.popitem(last=False)
                self.close_arrays(removed)
            self.cache[path] = arrays
        self.cache.move_to_end(path)
        return self.cache[path]

    def close(self):
        for arrays in self.cache.values():
            self.close_arrays(arrays)
        self.cache.clear()


def great_circle_m(lon1, lat1, lon2, lat2):
    a, b = math.radians(lat1), math.radians(lat2)
    dp, dl = b - a, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dl / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def locate_pixel(record, lon, lat, grid):
    require(math.isfinite(lon) and math.isfinite(lat), 'Coordinates must be finite')
    require(record['west_edge'] <= lon < record['east_edge']
            and record['south_edge'] < lat <= record['north_edge'], 'Point outside this patch')
    fr = (grid['latitude_origin'] - lat) / grid['step_degrees'] - record['global_row0']
    fc = (lon - grid['longitude_origin']) / grid['step_degrees'] - record['global_col0']
    # Ties on cell boundaries go east/south; account only for float roundoff.
    rr, cc = int(math.floor(fr + .5 + 1e-9)), int(math.floor(fc + .5 + 1e-9))
    require(0 <= rr < 128 and 0 <= cc < 128, 'Coordinate rounds outside patch')
    lr_float_r = (fr - grid['lr_anchor_offset']) / grid['lr_stride']
    lr_float_c = (fc - grid['lr_anchor_offset']) / grid['lr_stride']
    lr_r = min(31, max(0, int(math.floor(lr_float_r + .5 + 1e-9))))
    lr_c = min(31, max(0, int(math.floor(lr_float_c + .5 + 1e-9))))
    hr_lon = grid['longitude_origin'] + (record['global_col0'] + cc) * grid['step_degrees']
    hr_lat = grid['latitude_origin'] - (record['global_row0'] + rr) * grid['step_degrees']
    lr_anchor_r, lr_anchor_c = lr_r * 4, lr_c * 4
    lr_lon = grid['longitude_origin'] + (record['global_col0'] + lr_anchor_c) * grid['step_degrees']
    lr_lat = grid['latitude_origin'] - (record['global_row0'] + lr_anchor_r) * grid['step_degrees']
    return dict(hr_row=rr, hr_col=cc, fractional_hr_row=fr, fractional_hr_col=fc,
                hr_longitude=hr_lon, hr_latitude=hr_lat,
                lr_row=lr_r, lr_col=lr_c, lr_anchor_hr_row=lr_anchor_r, lr_anchor_hr_col=lr_anchor_c,
                lr_anchor_longitude=lr_lon, lr_anchor_latitude=lr_lat,
                hr_distance_to_query_m=great_circle_m(lon, lat, hr_lon, hr_lat),
                lr_anchor_distance_to_query_m=great_circle_m(lon, lat, lr_lon, lr_lat),
                lr_nearest_anchor_at_patch_edge=bool(lr_float_r < 0 or lr_float_r > 31
                                                    or lr_float_c < 0 or lr_float_c > 31),
                spatial_policy='nearest HR centre and nearest available sampled LR anchor; no interpolation',
                native_fy4_sensor_footprint_reconstructed=False)


class SatelliteStore:
    def __init__(self, config=None):
        self.cfg = load_config(config)
        self.output = Path(self.cfg['_output'])
        self.identity = json.loads((self.output / 'satellite_identity.json').read_text(encoding='utf-8'))
        require(self.identity['status'] == 'ready', 'Run step_1_prepare.py build before extraction')
        require(sha256(self.cfg['_path']) == self.identity['config_sha256'], 'Config changed; rebuild the index')
        for path, expected in self.identity['small_file_sha256'].items():
            require(sha256(path) == expected, f'Protected identity file changed: {path}')
        for path, expected in self.identity['large_file_signatures'].items():
            require(signature(path) == expected, f'Validated source file changed: {path}')
        require(sha256(self.output / 'test_patch_index.sqlite') == self.identity['index_db_sha256'],
                'Prepared index changed; rebuild rather than using an unverified index')
        self.db = sqlite3.connect((self.output / 'test_patch_index.sqlite').as_uri() + '?mode=ro', uri=True)
        self.prediction = np.load(Path(self.cfg['formal_run']) / 'drct_prediction_csi.npy', mmap_mode='r', allow_pickle=False)
        self.clear = np.load(self.cfg['clear_ghi'], mmap_mode='r', allow_pickle=False)
        self.bicubic_manifest = json.loads(Path(self.cfg['bicubic_manifest']).read_text(encoding='utf-8'))
        self.bicubic_path = Path(self.cfg['bicubic_prediction'])
        require(self.bicubic_manifest['opencv_version'] == '5.0.0', 'Bicubic export uses the wrong OpenCV version')
        require(self.bicubic_manifest['sha256'] == sha256(self.bicubic_path), 'Bicubic prediction checksum mismatch')
        self.bicubic_prediction = np.load(self.bicubic_path, mmap_mode='r', allow_pickle=False)
        n = self.cfg['expected_test_samples']
        require(self.prediction.shape == self.clear.shape == self.bicubic_prediction.shape == (n, 128, 128),
                'Array shape changed')
        require(self.bicubic_prediction.dtype == np.float32, 'Bicubic prediction dtype changed')
        self.reader = PatchReader()
        self.bicubic_cache = OrderedDict()

    def record(self, sample_id):
        require(type(sample_id) is int, 'sample_id must be a global integer ID, not CSV row position')
        row = self.db.execute('SELECT record_json FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        if row is None:
            raise KeyError(f'sample_id {sample_id} is not in the formal test set')
        return json.loads(row[0])

    def locate(self, longitude, latitude, date=None):
        require(math.isfinite(longitude) and math.isfinite(latitude)
                and -180 <= longitude <= 180 and -90 <= latitude <= 90, 'Invalid longitude/latitude')
        sql = ('SELECT record_json FROM samples WHERE west_edge<=? AND east_edge>? '
               'AND south_edge<? AND north_edge>=?')
        params = [longitude, longitude, latitude, latitude]
        if date is not None:
            datetime.strptime(date, '%Y%m%d')
            sql += ' AND date=?'
            params.append(date)
        sql += ' ORDER BY test_index'
        return [dict(record=json.loads(v[0]), location=locate_pixel(json.loads(v[0]), longitude, latitude,
                     self.cfg['grid'])) for v in self.db.execute(sql, params)]

    def extract(self, sample_id, pixel_row, pixel_col, *, longitude=None, latitude=None):
        require(type(pixel_row) is int and type(pixel_col) is int
                and 0 <= pixel_row < 128 and 0 <= pixel_col < 128, 'HR pixel indices must be integers in [0,127]')
        rec = self.record(sample_id)
        path = Path(rec['npz_path'])
        expected = self.identity['npz_signatures'][str(path)]
        require(signature(path) == expected, f'Source NPZ changed: {path}')
        a = self.reader.arrays(path)
        i, j, member, pos = pixel_row, pixel_col, rec['array_index'], rec['test_index']
        g = self.cfg['grid']
        qlon = g['longitude_origin'] + (rec['global_col0'] + j) * g['step_degrees']
        qlat = g['latitude_origin'] - (rec['global_row0'] + i) * g['step_degrees']
        require((longitude is None) == (latitude is None), 'Supply both query coordinates')
        if longitude is not None:
            qlon, qlat = longitude, latitude
        location = locate_pixel(rec, qlon, qlat, g)
        require((location['hr_row'], location['hr_col']) == (i, j), 'Query coordinates/pixel indices disagree')
        clear = np.float32(self.clear[pos, i, j])
        pred = np.float32(self.prediction[pos, i, j])
        bicubic = np.float32(self.bicubic_prediction[pos, i, j])
        target = np.float32(a['hr_csi'][member, i, j])
        clear_valid = bool(np.isfinite(clear) and clear > 0)
        values = dict(fy4a=np.float32(a['fy4'][member, location['lr_row'], location['lr_col']]),
                      fy3c=np.float32(a['fy3'][member, i, j]),
                      bicubic=np.float32(bicubic * clear) if clear_valid else np.float32(np.nan),
                      drct=np.float32(pred * clear) if clear_valid else np.float32(np.nan))
        observed_valid = lambda v: bool(np.isfinite(v) and v >= 0 and v not in (-999., 65532., 65535.))
        valid = dict(fy4a=observed_valid(values['fy4a']), fy3c=observed_valid(values['fy3c']),
                     bicubic=bool(np.isfinite(values['bicubic'])), drct=bool(np.isfinite(values['drct'])))
        return dict(sample_id=sample_id, test_index=pos, npz_path=str(path), array_index=member,
                    query_longitude=qlon, query_latitude=qlat, location=location,
                    ghi_w_m2={k: finite_value(v) for k, v in values.items()},
                    csi=dict(drct=finite_value(pred), bicubic=finite_value(bicubic), fy3c=finite_value(target)),
                    clear_ghi_w_m2=finite_value(clear),
                    clear_ghi_provenance='existing aligned array reconstructed by fy3/hr_csi; not independent native REST2 output',
                    clear_ghi_zero_target_recovery=bool(target == 0),
                    satellite_valid=valid, all_four_satellite_values_valid=all(valid.values()),
                    fy4_window_start_utc=rec['fy4_window_start_utc'], fy4_window_end_utc=rec['fy4_window_end_utc'],
                    fy4_time_support='scan window, NOT a 15-minute average or known instantaneous pixel time',
                    fy3_center_time_utc=rec['fy3_center_time_utc'],
                    fy3_center_time_status=rec['fy3_center_time_status'],
                    fy3_center_inside_fy4_window=rec['fy3_center_inside_fy4_window'],
                    drct_source='existing formal epoch-65 EMA prediction; no inference or training run',
                    negative_reconstructed_ghi_retained=True)

    def at_location(self, longitude, latitude, date=None):
        matches = self.locate(longitude, latitude, date)
        return [self.extract(m['record']['sample_id'], m['location']['hr_row'], m['location']['hr_col'],
                             longitude=longitude, latitude=latitude) for m in matches]

    def close(self):
        self.reader.close()
        self.prediction._mmap.close()
        self.clear._mmap.close()
        self.bicubic_prediction._mmap.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
