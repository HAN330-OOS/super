"""Validate all four products against one CMA mean per existing FY-4A window."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import timedelta
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from step_2_satellite import SatelliteStore, require, utc_datetime

STATION_FIELDS = ('station_id', 'longitude', 'latitude')
GROUND_FIELDS = ('station_id', 'time_utc', 'ghi_w_m2', 'qc_flag', 'time_support',
                 'interval_start_utc', 'interval_end_utc')
METHODS = {'fy4a': 'FY-4A', 'fy3c': 'FY-3C', 'bicubic': 'Bicubic', 'drct': 'DRCT'}
PAIR_FIELDS = (
    'station_id', 'station_longitude', 'station_latitude', 'sample_id', 'test_index',
    'fy4_window_start_utc', 'fy4_window_end_utc', 'valid_ground_minutes', 'cma_ghi_w_m2',
    *(f'{method}_ghi_w_m2' for method in METHODS),
    'hr_row', 'hr_col', 'lr_row', 'lr_col', 'hr_distance_to_station_m', 'lr_distance_to_station_m',
    'fy3_center_time_utc', 'fy3_center_time_status', 'fy3_center_inside_fy4_window',
    'npz_path', 'array_index',
)
METRIC_FIELDS = ('method', 'stations', 'pairs', 'R2', 'RMSE_w_m2', 'MAE_w_m2', 'MBE_w_m2')


def read_stations(path):
    stations = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(set(STATION_FIELDS).issubset(reader.fieldnames or []), 'Missing station fields')
        for line, row in enumerate(reader, 2):
            sid = row['station_id'].strip()
            lon, lat = float(row['longitude']), float(row['latitude'])
            require(sid and sid not in stations, f'Empty/duplicate station_id at line {line}')
            require(math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90,
                    f'Invalid station coordinates at line {line}')
            stations[sid] = dict(station_id=sid, longitude=lon, latitude=lat)
    require(stations, 'No station coordinates supplied')
    return stations


def spatial_windows(store, stations, window_duration):
    """Look up the already-paired patches for each station before reading CMA."""
    result = {}
    for sid, station in stations.items():
        matches = store.locate(station['longitude'], station['latitude'])
        windows = []
        for match in matches:
            record = match['record']
            start = utc_datetime(record['fy4_window_start_utc'])
            end = utc_datetime(record['fy4_window_end_utc'])
            require(end - start == window_duration, f'Unexpected scan duration: {record["sample_id"]}')
            windows.append(dict(match, start=start, end=end, minutes={}))
        windows.sort(key=lambda window: window['start'])
        require(len({w['start'] for w in windows}) == len(windows), f'Duplicate scan window for station {sid}')
        if windows:
            result[sid] = windows
    return result


def minute_in_window(label, support_start, support_end, window_start, window_end):
    if support_start is None:
        return window_start <= label < window_end
    return window_start <= support_start and support_end <= window_end


def read_cma_windows(path, windows_by_station, accepted_qc_values, window_duration):
    """Stream the input and retain only minutes belonging to spatially matched scans."""
    accepted_qc = {str(value).strip() for value in accepted_qc_values}
    require(accepted_qc, 'Supply the accepted QC flag values from the CMA documentation')
    starts = {sid: [w['start'] for w in windows] for sid, windows in windows_by_station.items()}
    counts = Counter()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(set(GROUND_FIELDS).issubset(reader.fieldnames or []), 'Missing minute-GHI fields')
        for line, row in enumerate(reader, 2):
            counts['input_rows'] += 1
            sid = row['station_id'].strip()
            if sid not in windows_by_station:
                continue
            label = utc_datetime(row['time_utc'])
            lo = bisect_left(starts[sid], label - window_duration)
            hi = bisect_right(starts[sid], label)
            if lo == hi:
                continue
            counts['rows_near_candidate_windows'] += 1
            if row['qc_flag'].strip() not in accepted_qc:
                counts['qc_rejected_rows'] += 1
                continue
            try:
                ghi = float(row['ghi_w_m2'])
            except (TypeError, ValueError):
                counts['missing_or_invalid_ghi_rows'] += 1
                continue
            if not math.isfinite(ghi):
                counts['missing_or_invalid_ghi_rows'] += 1
                continue
            support = row['time_support'].strip()
            require(support in ('instantaneous', 'one_minute_mean'), f'Invalid time_support at line {line}')
            support_start = utc_datetime(row['interval_start_utc']) if row['interval_start_utc'] else None
            support_end = utc_datetime(row['interval_end_utc']) if row['interval_end_utc'] else None
            if support == 'one_minute_mean':
                require(support_start is not None and support_end is not None
                        and support_end - support_start == timedelta(minutes=1)
                        and support_start <= label <= support_end,
                        f'Invalid one-minute support at line {line}')
            else:
                require(support_start is None and support_end is None,
                        f'Instantaneous measurement has a support interval at line {line}')
            minute = (support_start if support_start is not None else label).replace(second=0, microsecond=0)
            value = (ghi, support_start, support_end)
            for window in windows_by_station[sid][lo:hi]:
                if not minute_in_window(label, support_start, support_end, window['start'], window['end']):
                    continue
                previous = window['minutes'].get(minute)
                if previous is not None:
                    require(previous == value, f'Conflicting accepted records for {sid} at {minute}, line {line}')
                    counts['identical_duplicate_minutes'] += 1
                window['minutes'][minute] = value
    require(counts['input_rows'], 'No actual CMA observations supplied')
    return dict(counts)


def shared_reference(minutes, minimum_minutes):
    if len(minutes) < minimum_minutes:
        return None
    values = [minutes[key] for key in sorted(minutes)]
    intervals = [(start, end) for _, start, end in values if start is not None]
    require(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])),
            'CMA one-minute averaging intervals overlap in a candidate window')
    return math.fsum(ghi for ghi, _, _ in values) / len(values)


def score_pairs(pairs):
    count = len(pairs)
    station_count = len({row['station_id'] for row in pairs})
    mean = math.fsum(row['cma_ghi_w_m2'] for row in pairs) / count if count else None
    total_variance = math.fsum((row['cma_ghi_w_m2'] - mean) ** 2 for row in pairs)
    result = []
    for method, name in METHODS.items():
        errors = [row[f'{method}_ghi_w_m2'] - row['cma_ghi_w_m2'] for row in pairs]
        squared_error = math.fsum(error ** 2 for error in errors)
        result.append(dict(method=name, stations=station_count, pairs=count,
                           R2=1 - squared_error / total_variance if total_variance > 0 else None,
                           RMSE_w_m2=math.sqrt(squared_error / count) if count else None,
                           MAE_w_m2=math.fsum(map(abs, errors)) / count if count else None,
                           MBE_w_m2=math.fsum(errors) / count if count else None))
    return result


def write_results(output_dir, pairs, metrics):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cma-', dir=output_dir.parent) as temporary:
        staged = Path(temporary) / 'results'
        staged.mkdir()
        for name, fields, rows in (('pairs.csv', PAIR_FIELDS, pairs), ('metrics.csv', METRIC_FIELDS, metrics)):
            with (staged / name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        require(not output_dir.exists(), f'Output directory already exists: {output_dir}')
        staged.rename(output_dir)


def match_ground(config, station_csv, ground_csv, output_dir, accepted_qc_values):
    stations = read_stations(station_csv)
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), f'Output directory already exists: {output_dir}')
    counts = Counter()
    pairs = []
    with SatelliteStore(config) as store:
        for name in ('samples_root', 'splits_root', 'formal_run'):
            require(not output_dir.is_relative_to(Path(store.cfg[name]).resolve()), 'Output must be outside source data')
        duration = timedelta(minutes=store.cfg['fy4_scan_window_minutes'])
        minimum_minutes = store.cfg['ground_contract'].get('minimum_valid_minutes', 8)
        require(type(minimum_minutes) is int and 1 <= minimum_minutes <= duration.total_seconds() / 60,
                'Invalid minimum_valid_minutes')
        windows = spatial_windows(store, stations, duration)
        counts['spatial_candidate_windows'] = sum(map(len, windows.values()))
        ground_counts = read_cma_windows(ground_csv, windows, accepted_qc_values, duration)
        for sid, candidates in windows.items():
            station = stations[sid]
            for window in candidates:
                record, location = window['record'], window['location']
                cma = shared_reference(window['minutes'], minimum_minutes)
                if cma is None:
                    counts['insufficient_ground_minutes'] += 1
                    continue
                sat = store.extract(record['sample_id'], location['hr_row'], location['hr_col'],
                                    longitude=station['longitude'], latitude=station['latitude'])
                if not sat['all_four_satellite_values_valid']:
                    counts['invalid_satellite_values'] += 1
                    continue
                pairs.append(dict(
                    station_id=sid, station_longitude=station['longitude'], station_latitude=station['latitude'],
                    sample_id=record['sample_id'], test_index=record['test_index'],
                    fy4_window_start_utc=record['fy4_window_start_utc'], fy4_window_end_utc=record['fy4_window_end_utc'],
                    valid_ground_minutes=len(window['minutes']), cma_ghi_w_m2=cma,
                    **{f'{method}_ghi_w_m2': sat['ghi_w_m2'][method] for method in METHODS},
                    hr_row=location['hr_row'], hr_col=location['hr_col'],
                    lr_row=location['lr_row'], lr_col=location['lr_col'],
                    hr_distance_to_station_m=location['hr_distance_to_query_m'],
                    lr_distance_to_station_m=location['lr_anchor_distance_to_query_m'],
                    fy3_center_time_utc=record['fy3_center_time_utc'],
                    fy3_center_time_status=record['fy3_center_time_status'],
                    fy3_center_inside_fy4_window=record['fy3_center_inside_fy4_window'],
                    npz_path=record['npz_path'], array_index=record['array_index'],
                ))
    metrics = score_pairs(pairs)
    write_results(output_dir, pairs, metrics)
    counts['valid_shared_pairs'] = len(pairs)
    return dict(output_directory=str(output_dir), minimum_valid_minutes=minimum_minutes,
                stations_with_spatial_coverage=len(windows), evaluated_stations=metrics[0]['stations'],
                window_counts=dict(counts), cma_row_counts=ground_counts)
