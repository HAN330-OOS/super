"""Query existing satellite pairs and evaluate them against CMA ground GHI."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

import numpy as np
from step_2_satellite import (FIELDS, BEIJING, UTC, PatchReader, SatelliteStore, date_hhmm,
                       iso, load_config, npz_path, read_csv, require, sha256,
                       signature, write_json)


def validate_order(cfg, run):
    seen_ids, seen_members, counts, hashes, test = set(), set(), {}, {}, None
    for split in ('train', 'validation', 'test'):
        path = Path(cfg['splits_root']) / (split + '_samples.csv')
        digest = sha256(path)
        require(digest == run['data_identity']['split_csv_sha256'][split], f'Formal split changed: {path}')
        rows = read_csv(path)
        counts[split], hashes[str(path)] = len(rows), digest
        for row in rows:
            member = (row['date'], row['npz_file'], int(row['array_index']))
            sid = int(row['sample_id'])
            require(sid not in seen_ids and member not in seen_members, 'Duplicate split membership')
            seen_ids.add(sid); seen_members.add(member)
        if split == 'test':
            test = rows
        del rows
    require(counts == {'train': 177380, 'validation': 35482, 'test': 35476}, 'Wrong formal split counts')
    old = read_csv(cfg['original_test_csv'])
    ix = np.load(cfg['clear_indices'], allow_pickle=False)
    require(ix.dtype == np.int32 and ix.shape == (35476,) and len(np.unique(ix)) == len(ix), 'Bad clean index mapping')
    require(all(0 <= int(i) < len(old) for i in ix), 'Original test index out of bounds')
    require([old[int(i)] for i in ix] == test, 'Cleaned test order does not match original clear-GHI mapping')
    require(sha256(cfg['clear_indices']) == run['data_identity']['clear_indices_sha256'], 'Index array hash changed')
    require(sha256(cfg['original_test_csv']) == run['data_identity']['original_test_csv_sha256'], 'Original CSV changed')
    order = read_csv(Path(cfg['formal_run']) / 'test_order.csv', required=(*FIELDS, 'prediction_index', 'csv_row_index'))
    require(len(order) == len(test), 'Prediction order length differs')
    for position, (row, predicted) in enumerate(zip(test, order)):
        require(predicted['prediction_index'] == predicted['csv_row_index'] == str(position)
                and all(row[k] == predicted[k] for k in FIELDS), f'Prediction order mismatch at {position}')
    needed = {int(row['sample_id']): row for row in test}
    checked = set()
    with Path(cfg['source_library_csv']).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            sid = int(row['sample_id'])
            if sid in needed:
                require(row == needed[sid] and sid not in checked, f'Library row mismatch/duplicate: {sid}')
                checked.add(sid)
    require(len(checked) == len(test), 'Formal test members missing from source library')
    return test, ix, counts, hashes


def scan_npy(path, expected_shape, expected_payload=None, expected_file=None, positive=False):
    """One sequential pass, finite checks + two hashes; memory bounded to 8 MiB."""
    path = Path(path)
    before = signature(path)
    mm = np.load(path, mmap_mode='r', allow_pickle=False)
    require(mm.dtype == np.float32 and mm.shape == expected_shape and mm.flags.c_contiguous,
            f'Wrong NPY layout: {path}')
    offset, size, dtype = mm.offset, mm.size, mm.dtype
    mm._mmap.close()
    full, payload = hashlib.sha256(), hashlib.sha256()
    seen, chunks = 0, 0
    with path.open('rb') as f:
        full.update(f.read(offset))
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            require(len(block) % dtype.itemsize == 0, f'Truncated NPY payload: {path}')
            values = np.frombuffer(block, dtype=dtype)
            require(np.isfinite(values).all(), f'Non-finite values in {path}, block {chunks}')
            if positive:
                require((values > 0).all(), f'Non-positive clear GHI: {path}, block {chunks}')
            full.update(block); payload.update(block)
            seen += len(values); chunks += 1
            if chunks % 64 == 0:
                print(f'Integrity {path.name}: {100 * seen / size:.1f}% ({seen:,}/{size:,} pixels)', flush=True)
    require(seen == size and signature(path) == before, f'Source length or signature changed: {path}')
    require(expected_payload is None or payload.hexdigest() == expected_payload, f'Payload hash mismatch: {path}')
    require(expected_file is None or full.hexdigest() == expected_file, f'File hash mismatch: {path}')
    return dict(path=str(path), shape=list(expected_shape), dtype='float32', finite_pixels=seen,
                file_sha256=full.hexdigest(), payload_sha256=payload.hexdigest(), signature=before,
                all_positive=bool(positive), sequential_single_pass=True, scan_buffer_bytes=8 * 1024 * 1024)


def inspect_npz_metadata(path):
    """Only four NPY headers and tiny per-sample metadata, with one ZIP handle."""
    headers, arrays = {}, {}
    with zipfile.ZipFile(path) as z:
        for key in ('fy3', 'fy4', 'hr_csi', 'lr_csi'):
            with z.open(key + '.npy') as f:
                v = np.lib.format.read_magic(f)
                if v == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif v == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f'Unsupported array header: {path}/{key}')
                require(not dtype.hasobject, 'Object arrays are not allowed')
                require(f.tell() + int(np.prod(shape)) * dtype.itemsize == z.getinfo(key + '.npy').file_size,
                        f'Truncated array member: {path}/{key}')
                headers[key] = (shape, dtype)
        n = headers['fy3'][0][0]
        require(1 <= n <= 64, f'Invalid NPZ sample count: {path}')
        for key, (shape, dtype) in headers.items():
            size = 32 if key in ('fy4', 'lr_csi') else 128
            require(shape == (n, size, size) and dtype == np.float32, f'Wrong NPZ shape/dtype: {path}/{key}')
        for key in ('row0', 'col0', 'center_lon', 'center_lat', 'dominant_slot_minute_utc',
                    'dominant_slot_minute_cst', 'valid_frac', 'bad_diff_frac', 'dominant_slot_frac', 'csi_valid_frac'):
            member = z.getinfo(key + '.npy')
            require(member.file_size <= 4096, f'Unexpected large metadata field: {path}/{key}')
            with z.open(member) as f:
                arrays[key] = np.load(f, allow_pickle=False)
            require(arrays[key].shape == (n,) and np.isfinite(arrays[key]).all(), f'Bad metadata: {path}/{key}')
    return n, arrays


def build(config, force=False):
    started = time.perf_counter()
    cfg = load_config(config)
    out = Path(cfg['_output'])
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / 'satellite_identity.json'
    if identity_path.is_file() and not force:
        old = json.loads(identity_path.read_text(encoding='utf-8'))
        require(old.get('status') != 'ready', 'An index is already ready; use --force only for an intentional rebuild')
    write_json(identity_path, {'status': 'building', 'ground_metrics_computed': False})
    run_dir = Path(cfg['formal_run'])
    run = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8-sig'))
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8-sig'))
    rows, old_indices, counts, hashes = validate_order(cfg, run)
    print('PASS: all 35,476 test rows align with the original CSV indices and formal prediction order', flush=True)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['date'], row['npz_file'])].append((i, row))
    db_path = out / 'test_patch_index.sqlite.tmp'
    if db_path.exists():
        # Only a known temporary file inside the configured output directory.
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE samples (test_index INTEGER PRIMARY KEY, sample_id INTEGER UNIQUE, date TEXT, '
               'west_edge REAL, east_edge REAL, south_edge REAL, north_edge REAL, record_json TEXT)')
    reader, npz_signatures, small_sigs = PatchReader(), {}, {}
    last_date, day_rows, meta = None, None, None
    time_outside_window, files = 0, 0
    invalid_times, outside_times = [], []
    try:
        for (date, filename), members in sorted(groups.items()):
            folder = Path(cfg['samples_root']) / date
            if date != last_date:
                manifest_path, meta_path = folder / 'manifest.csv', folder / 'build_meta.npz'
                daily = read_csv(manifest_path)
                day_rows = {(r['npz_file'], int(r['array_index'])): r for r in daily}
                require(len(day_rows) == len(daily), f'Duplicate daily manifest entry: {date}')
                with np.load(meta_path, allow_pickle=False) as z:
                    meta = {k: z[k].item() for k in ('date', 'lon_min', 'lon_max', 'lat_min', 'lat_max',
                                                    'fy3_patch', 'fy4_patch', 'window_minutes', 'accepted_samples')}
                require(str(meta['date']) == date and meta['accepted_samples'] == len(daily), f'Daily count mismatch: {date}')
                require((meta['lon_min'], meta['lon_max'], meta['lat_min'], meta['lat_max']) == (73., 135., 18., 54.)
                        and (meta['fy3_patch'], meta['fy4_patch'], meta['window_minutes']) == (128, 32, 15),
                        f'Grid or time convention differs: {date}')
                for p in (manifest_path, meta_path):
                    hashes[str(p)] = sha256(p)
                    small_sigs[str(p)] = signature(p)
                last_date = date
            path = npz_path(cfg, members[0][1]).resolve()
            sample_count, a = inspect_npz_metadata(path)
            npz_signatures[str(path)] = signature(path)
            for position, row in members:
                k = int(row['array_index'])
                require(0 <= k < sample_count, f'NPZ member out of bounds: {row}')
                daily = day_rows[(filename, k)]
                require(all(daily[x] == row[x] for x in FIELDS if x != 'sample_id'), f'Daily/global CSV mismatch: {row}')
                gr, gc = int(row['row']), int(row['col'])
                require(0 <= gr <= 3600 - 128 and 0 <= gc <= 6200 - 128
                        and gr % 128 == gc % 128 == 0, f'Invalid source grid position: {row}')
                require(int(a['row0'][k]) == gr and int(a['col0'][k]) == gc, f'NPZ grid origin differs: {row}')
                require(abs(float(a['center_lon'][k]) - float(row['center_lon'])) < 1e-8
                        and abs(float(a['center_lat'][k]) - float(row['center_lat'])) < 1e-8,
                        f'NPZ centre differs: {row}')
                require(abs(73 + (gc + 64) * .01 - float(row['center_lon'])) < 1e-8
                        and abs(54 - (gr + 64) * .01 - float(row['center_lat'])) < 1e-8,
                        f'Grid convention differs: {row}')
                scan = date_hhmm(date, row['fy4_time'], UTC)
                try:
                    centre = date_hhmm(date, row['fy3_time'], BEIJING)
                except ValueError:
                    centre = None
                    invalid_times.append(dict(test_index=position, **row))
                require(int(a['dominant_slot_minute_utc'][k]) == scan.hour * 60 + scan.minute,
                        f'NPZ time differs: {row}')
                require(int(a['dominant_slot_minute_cst'][k]) == (scan.hour * 60 + scan.minute + 480) % 1440,
                        f'NPZ Beijing time differs: {row}')
                inside = scan <= centre.astimezone(UTC) < scan + timedelta(minutes=15) if centre else None
                if inside is False:
                    time_outside_window += 1
                    outside_times.append(dict(test_index=position, **row))
                west, east = 73 + gc * .01, 73 + (gc + 127) * .01
                north, south = 54 - gr * .01, 54 - (gr + 127) * .01
                rec = dict(test_index=position, sample_id=int(row['sample_id']), date=date,
                           original_test_index=int(old_indices[position]), prediction_index=position, clear_ghi_index=position,
                           npz_path=str(path), array_index=k, manifest_path=str(folder / 'manifest.csv'),
                           date_local_sample_id=int(daily['sample_id']), build_meta_path=str(folder / 'build_meta.npz'),
                           global_row0=gr, global_col0=gc, hr_rows=128, hr_cols=128, lr_rows=32, lr_cols=32,
                           center_longitude=float(row['center_lon']), center_latitude=float(row['center_lat']),
                           longitude_min_center=west, longitude_max_center=east, latitude_min_center=south,
                           latitude_max_center=north, west_edge=west - .005, east_edge=east + .005,
                           south_edge=south - .005, north_edge=north + .005, grid_step_degrees=.01,
                           lr_stride=4, lr_anchor_hr_offset=0, row_direction='north_to_south',
                           fy4_time_label_utc=row['fy4_time'], fy4_window_start_utc=iso(scan),
                           fy4_window_end_utc=iso(scan + timedelta(minutes=15)), fy4_window_closed='[start,end)',
                           fy4_is_15min_mean=False, fy4_pixel_instantaneous_time_known=False,
                           fy3_time_label_beijing=row['fy3_time'], fy3_center_time_beijing=centre.isoformat() if centre else None,
                           fy3_center_time_utc=iso(centre) if centre else None,
                           fy3_center_time_status='valid_label' if centre else 'invalid_original_label',
                           fy3_time_basis='estimated centre pixel, not a patch-wide timestamp',
                           fy3_direct_instrument_time=False, fy3_center_inside_fy4_window=inside,
                           clear_ghi_file=cfg['clear_ghi'], prediction_file=str(run_dir / 'drct_prediction_csi.npy'))
                for field in ('valid_frac', 'bad_diff_frac', 'dominant_slot_frac', 'csi_valid_frac'):
                    rec['source_' + field] = float(a[field][k])
                db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)',
                           (position, rec['sample_id'], date, rec['west_edge'], rec['east_edge'],
                            rec['south_edge'], rec['north_edge'], json.dumps(rec, ensure_ascii=False, allow_nan=False)))
            files += 1
            if files % 250 == 0 or files == len(groups):
                db.commit()
                print(f'NPZ headers and metadata: {files}/{len(groups)} files', flush=True)
        db.execute('CREATE INDEX patch_date ON samples(date)')
        db.execute('CREATE INDEX patch_bounds ON samples(west_edge,east_edge,south_edge,north_edge)')
        db.commit()
        with (out / 'test_patch_index.csv.tmp').open('w', encoding='utf-8-sig', newline='') as f:
            writer = None
            for (text,) in db.execute('SELECT record_json FROM samples ORDER BY test_index'):
                rec = json.loads(text)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(rec)); writer.writeheader()
                writer.writerow(rec)
    finally:
        reader.close(); db.close()
    expected = {}
    for line in (run_dir / 'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip().lstrip('*')] = digest
    prediction = scan_npy(run_dir / 'drct_prediction_csi.npy', (35476, 128, 128),
                          metrics['prediction']['payload_sha256'], expected['drct_prediction_csi.npy'])
    clear = scan_npy(cfg['clear_ghi'], (35476, 128, 128), metrics['selected_clear_ghi_payload_sha256'], positive=True)
    checkpoint = Path(cfg['checkpoint'])
    require(sha256(checkpoint) == run['checkpoint_sha256'], 'Formal checkpoint hash mismatch')
    for name in ('run_config.json', 'metrics.json', 'test_order.csv', 'export_test_predictions.py'):
        p = run_dir / name
        require(sha256(p) == expected[name], f'Formal export identity file changed: {name}')
        hashes[str(p)] = expected[name]
    for p in (Path(cfg['original_test_csv']), Path(cfg['clear_indices']), Path(cfg['clear_provenance']),
              Path(cfg['source_library_csv'])):
        hashes[str(p)] = sha256(p)
    for path, stat in small_sigs.items():
        require(signature(path) == stat, f'Metadata source changed while indexing: {path}')
    (out / 'test_patch_index.csv.tmp').replace(out / 'test_patch_index.csv')
    db_path.replace(out / 'test_patch_index.sqlite')
    for name, header in (('stations_template.csv', cfg['ground_contract']['stations_columns']),
                         ('minute_ghi_template.csv', cfg['ground_contract']['observations_columns'])):
        with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(header)  # Deliberately NO invented observations.
    write_json(out / 'time_label_anomalies.json', dict(invalid_original_center_time=invalid_times,
        valid_center_outside_fy4_window=outside_times, samples_removed_or_repaired=False))
    report = dict(status='ready', prepared_at=iso(datetime.now(UTC)),
        config_sha256=sha256(cfg['_path']), small_file_sha256=hashes,
        large_file_signatures={str(Path(prediction['path'])): prediction['signature'],
                               str(Path(clear['path'])): clear['signature'], str(checkpoint): signature(checkpoint)},
        npz_signatures=npz_signatures,
        index_csv_sha256=sha256(out / 'test_patch_index.csv'), index_db_sha256=sha256(out / 'test_patch_index.sqlite'),
        formal_split_counts=counts, formal_test_rows_verified=len(rows), npz_files_verified=files,
        test_dates=len({r['date'] for r in rows}), full_npy_integrity=dict(prediction=prediction, clear_ghi=clear),
        checkpoint_sha256=run['checkpoint_sha256'], checkpoint_epoch=run['checkpoint_epoch'],
        prediction_reused_without_inference=True, original_time_pairs_changed=False,
        patch_centre_outside_fy4_window_count=time_outside_window,
        invalid_original_fy3_center_time_count=len(invalid_times),
        grid_and_npz_metadata_verified_all_test_rows=True, full_npz_pixel_finiteness_rescanned=False,
        clear_ghi_provenance=json.loads(Path(cfg['clear_provenance']).read_text(encoding='utf-8-sig')),
        clear_provenance_note='Existing recovered aligned array, NOT independently recomputed native REST2 output',
        ground_observations_created=0, ground_validation_metrics_computed=False,
        drct_fy4_deviation_analysis_run=False, elapsed_seconds=time.perf_counter() - started)
    write_json(identity_path, report)
    print(json.dumps({k: report[k] for k in ('status', 'formal_split_counts', 'npz_files_verified', 'test_dates',
                    'checkpoint_epoch', 'patch_centre_outside_fy4_window_count', 'elapsed_seconds')}, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--force', action='store_true')
    e = sub.add_parser('extract'); e.add_argument('--sample-id', type=int, required=True)
    e.add_argument('--row', type=int, required=True); e.add_argument('--col', type=int, required=True)
    l = sub.add_parser('locate'); l.add_argument('--lon', type=float, required=True)
    l.add_argument('--lat', type=float, required=True); l.add_argument('--date')
    m = sub.add_parser('match'); m.add_argument('--stations', type=Path, required=True)
    m.add_argument('--ground', type=Path, required=True)
    m.add_argument('--output-dir', type=Path, required=True)
    m.add_argument('--qc-ok', action='append', required=True)
    a = p.parse_args()
    if a.command == 'build':
        build(a.config, a.force)
    elif a.command == 'match':
        from step_3_ground import match_ground
        print(json.dumps(match_ground(a.config, a.stations, a.ground, a.output_dir,
                                     a.qc_ok), indent=2, ensure_ascii=False))
    else:
        with SatelliteStore(a.config) as store:
            if a.command == 'extract':
                value = store.extract(a.sample_id, a.row, a.col)
            else:
                value = store.locate(a.lon, a.lat, a.date)
            print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
