"""Read the existing 2018--2019 CSV splits without changing their membership.

Source NPZ CSI values are used directly: no normalization and no clipping.
An optional cloud_cache.json identifies a local NPY cache of the same raw values.
Neither format contains the matched clear-sky GHI required for physical SR GHI.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ('train', 'validation', 'test')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(root, split):
    path = Path(root) / 'splits' / f'{split}_samples.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'sample_id', 'date', 'npz_file', 'array_index'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Missing CSV columns in {path}: {required}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty split: {path}')
    for row in rows:
        date, filename = row['date'], row['npz_file']
        if len(date) != 8 or not date.isdigit() or Path(filename).name != filename or '\\' in filename:
            raise ValueError(f'Invalid date/archive name in {path}')
        if not filename.startswith(date + '_') or not filename.endswith('.npz'):
            raise ValueError(f'Unexpected NPZ filename: {filename}')
        if int(row['array_index']) < 0 or int(row['sample_id']) < 0:
            raise ValueError(f'Negative sample index in {path}')
    return rows


class NPZCSIDataset(Dataset):
    has_clear_ghi = False
    storage_format = '2018_2019_npz'

    def __init__(self, root, split, rows=None):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.rows = read_rows(self.root, split) if rows is None else rows
        self.records = [dict(row, index=index, batch_file=row['npz_file'],
                             batch_index=int(row['array_index']), role=split)
                        for index, row in enumerate(self.rows)]
        self.index = [(Path(row['date'] + '_utc') / 'patches' / row['npz_file'],
                       int(row['array_index'])) for row in self.rows]
        self._cache = OrderedDict()
        self._owner = None

    def __len__(self):
        return len(self.rows)

    def indices_for_role(self, role):
        return list(range(len(self))) if role == self.split else []

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_cache'] = OrderedDict()
        state['_owner'] = None
        return state

    def _arrays(self, row):
        if self._owner != os.getpid():
            self._cache = OrderedDict()
            self._owner = os.getpid()
        key = (row['date'], row['npz_file'])
        if key not in self._cache:
            path = self.root / 'samples' / row['date'] / row['npz_file']
            with np.load(path, allow_pickle=False) as archive:
                low = np.asarray(archive['lr_csi'])
                high = np.asarray(archive['hr_csi'])
            if low.dtype != np.float32 or high.dtype != np.float32:
                raise TypeError(f'Expected raw float32 CSI: {path}')
            if low.shape[1:] != (32, 32) or high.shape != (low.shape[0], 128, 128):
                raise ValueError(f'Invalid CSI shapes: {path}: {low.shape}, {high.shape}')
            self._cache[key] = (low, high)
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __getitem__(self, item):
        row = self.rows[int(item)]
        low, high = self._arrays(row)
        index = int(row['array_index'])
        low = np.array(low[index], dtype=np.float32, copy=True)
        high = np.array(high[index], dtype=np.float32, copy=True)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise FloatingPointError(f'Non-finite CSI sample: {self.split}/{item}')
        return torch.from_numpy(low[None]), torch.from_numpy(high[None])

    def physical_ghi_patch(self, item):
        row = self.rows[int(item)]
        path = self.root / 'samples' / row['date'] / row['npz_file']
        with np.load(path, allow_pickle=False) as archive:
            return np.array(archive['fy3'][int(row['array_index'])], dtype=np.float32, copy=True)


class CachedCSIDataset(NPZCSIDataset):
    storage_format = '2018_2019_npy_cache'

    def __init__(self, root, split, rows=None):
        super().__init__(root, split, rows)
        self._low = self._high = None

    def __getstate__(self):
        state = super().__getstate__()
        state['_low'] = state['_high'] = None
        return state

    def __getitem__(self, item):
        if self._owner != os.getpid():
            self._low = np.load(self.root / self.split / 'lr_csi.npy', mmap_mode='r', allow_pickle=False)
            self._high = np.load(self.root / self.split / 'hr_csi.npy', mmap_mode='r', allow_pickle=False)
            if self._low.shape != (len(self), 1, 32, 32) or self._high.shape != (len(self), 1, 128, 128):
                raise ValueError(f'Invalid cached CSI shapes in {self.root / self.split}')
            if self._low.dtype != np.float32 or self._high.dtype != np.float32:
                raise TypeError('Cached arrays must be float32')
            self._owner = os.getpid()
        # Cache contains the original CSI values. No rescaling is performed.
        low = np.array(self._low[int(item)], copy=True)
        high = np.array(self._high[int(item)], copy=True)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise FloatingPointError(f'Non-finite cached sample: {self.split}/{item}')
        return torch.from_numpy(low), torch.from_numpy(high)

    def physical_ghi_patch(self, item):
        raise RuntimeError('The CSI-only cache does not store physical GHI patches')


def load_npz_splits(root, *, selected_indices=None):
    root = Path(root).expanduser().resolve()
    if (root / '.building').exists():
        raise ValueError('CSI cache has a .building marker; conversion did not finish. Do not train on it.')
    cache_path = root / 'cloud_cache.json'
    cached = cache_path.exists()
    if cached:
        if selected_indices is not None:
            raise ValueError('Download selections apply only to raw NPZ, not a CSI cache')
        meta = json.loads(cache_path.read_text(encoding='utf-8'))
        if meta.get('format') != 'npz_csi_cache_v2_raw' or meta.get('state') != 'complete' or meta.get('representation') != 'raw_csi':
            raise ValueError('Cache must be a complete raw-CSI v2 cache. Old scaled caches are not accepted.')
        for split in SPLITS:
            csv_path = root / 'splits' / f'{split}_samples.csv'
            if file_hash(csv_path) != meta['split_csv_sha256'][split]:
                raise ValueError(f'Cached split CSV changed: {csv_path}')
    all_ids, all_keys = set(), set()
    datasets = []
    for split in SPLITS:
        rows = read_rows(root, split)
        for row in rows:
            sample_id = int(row['sample_id'])
            key = (row['date'], row['npz_file'], int(row['array_index']))
            if sample_id in all_ids or key in all_keys:
                raise ValueError(f'Duplicate sample within/across splits: {key}')
            all_ids.add(sample_id)
            all_keys.add(key)
        dataset = (CachedCSIDataset if cached else NPZCSIDataset)(root, split, rows)
        datasets.append(dataset)
    summary_path = root / 'status' / 'summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('state') != 'complete':
            raise ValueError('Source dataset summary is not complete')
        if 'sample_count' in summary and int(summary['sample_count']) != sum(map(len, datasets)):
            raise ValueError('Split CSV counts disagree with the dataset summary')
        for split, dataset in zip(SPLITS, datasets):
            declared = summary.get('split', {}).get(split + '_samples')
            if declared is not None and int(declared) != len(dataset):
                raise ValueError(f'{split} CSV count disagrees with the dataset summary')
    if selected_indices is not None:
        if set(selected_indices) != set(SPLITS):
            raise ValueError('A download selection must specify train, validation and test indices')
        selected_datasets = []
        for split, dataset in zip(SPLITS, datasets):
            indices = selected_indices[split]
            if not indices or any(type(index) is not int or index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f'Invalid selected CSV row in {split}')
            if len(indices) != len(set(indices)):
                raise ValueError(f'Duplicate selected CSV rows in {split}')
            selected_datasets.append(NPZCSIDataset(root, split, [dataset.rows[index] for index in indices]))
        datasets = selected_datasets
        all_keys = {(row['date'], row['npz_file'], int(row['array_index']))
                    for dataset in datasets for row in dataset.rows}
    if not cached:
        missing = [str(root / 'samples' / date / filename)
                   for date, filename in sorted({(date, name) for date, name, _ in all_keys})
                   if not (root / 'samples' / date / filename).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} NPZ files missing. Finish/retry OSS transfer. First: {missing[:5]}')
    return tuple(datasets)


def npz_data_identity(root):
    root = Path(root).expanduser().resolve()
    summary_path = root / 'status' / 'summary.json'
    cache_path = root / 'cloud_cache.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else None
    files, archives = {}, set()
    for split in SPLITS:
        path = root / 'splits' / f'{split}_samples.csv'
        files[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
        if cache:
            for filename in ('lr_csi.npy', 'hr_csi.npy'):
                array_path = root / split / filename
                files[array_path.relative_to(root).as_posix()] = {'bytes': array_path.stat().st_size}
        else:
            archives.update((row['date'], row['npz_file']) for row in read_rows(root, split))
    for date, filename in sorted(archives):
        path = root / 'samples' / date / filename
        files[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size}
    for path in (summary_path, cache_path):
        if path.exists():
            files[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    return {'dataset_format': '2018_2019_npy_cache' if cache else '2018_2019_npz',
            'preview_only': bool(summary.get('preview_only') or (cache or {}).get('preview_only')),
            'split': summary.get('split', (cache or {}).get('split', {})),
            'fingerprint_scope': 'CSV/summary/cache metadata SHA256 plus archive/array file sizes; not full payload hashes',
            'files': files}
