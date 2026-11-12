"""FY-3C / FY-4A SSI 共用常量、默认路径与数据读取。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_COMPARE_DIR = PROJECT_ROOT / "outputs" / "fy" / "compare"
OUTPUT_TIMESTAMP_DIR = PROJECT_ROOT / "outputs" / "fy" / "timestamp"
OUTPUT_RESAMPLED_ON_FY3_DIR = PROJECT_ROOT / "outputs" / "fy" / "resampled_on_fy3"
OUTPUT_REST2_ON_FY3_DIR = PROJECT_ROOT / "outputs" / "fy" / "rest2_on_fy3"
OUTPUT_MATCHING_DIR = PROJECT_ROOT / "outputs" / "fy" / "matching"
OUTPUT_SAMPLES_DIR = PROJECT_ROOT / "outputs" / "fy" / "samples"
OUTPUT_PLOTS_TWO_SWATHS_DIR = PROJECT_ROOT / "outputs" / "fy" / "plots" / "two_swaths"

FY4_ON_FY3_NPZ_PATTERN = re.compile(r"FY4_(\d{4})_on_FY3\.npz$", re.IGNORECASE)
REST2_ON_FY3_NPZ_PATTERN = re.compile(r"REST2_(\d{4})_on_FY3\.npz$", re.IGNORECASE)

FY4_NC_TIME_PATTERN = re.compile(r"(\d{14})_(\d{14})")

# FY-4A AGRI L2 SSI 文件名 YYYYMMDDHHMMSS 为 UTC（见 FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf；
# time_coverage_start/end 格式 YYYY-MM-DDTHH:MM:SS.sZ）。
CST_OFFSET_MINUTES = 8 * 60
FY4_SPEC_PDF = PROJECT_ROOT / "FY-4A_AGRI_L2_SSI_NOM_4000M_V1.1.pdf"

# SSI 无效填充与有效范围（见 FY4_SPEC_PDF；变量属性 valid_range=[0, 1500] W/m²）。
SSI_FILL_MISSING = -999.0
SSI_FILL_SZA_MASK = 65532.0  # V1.0 NC（如 20180501）：SZA>70°；V1.1 文档为 SZA>90°
SSI_FILL_OFF_DISK = 65535.0
SSI_VALID_MIN = 0.0
SSI_VALID_MAX = 1500.0
SSI_INVALID_FILLS = (SSI_FILL_MISSING, SSI_FILL_SZA_MASK, SSI_FILL_OFF_DISK)

# FY-4A 4000 m 全圆盘 CGMS 投影（satellite 104.7°E，与 N_DISK_1047E 产品一致）
FY4_NSMC_COFF = 1373.5
FY4_NSMC_LOFF = 1373.5
FY4_NSMC_CFAC = 10_233_137.0
FY4_NSMC_LFAC = 10_233_137.0
FY4_NSMC_SSP_LON = 104.7
FY4_NSMC_A = 6_378_137.0
FY4_NSMC_B = 6_356_752.31414
FY4_NSMC_H = 35_786_023.0

DEFAULT_FY3_ROOT = Path(r"E:\FY-3C\SSI")
DEFAULT_FY4_ROOT = Path(r"E:\FY-4A")
DEFAULT_FY4_GEO_RAW = Path(r"E:\经纬度文件\FY4A_FullMask_Grid_4000.raw")
DEFAULT_REST2_CACHE_DIR = PROJECT_ROOT / "MERRA2" / "rest2_outputs"

FY3_SHAPE = (3600, 6200)
FY4_SHAPE = (2748, 2748)
FY3_LON_START = 73.0
FY3_LAT_START = 54.0
FY3_STEP = 0.01
INVALID_GEO_FILL = 1_000_000.0

DEFAULT_LON_MIN, DEFAULT_LON_MAX = 73.0, 135.0
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 18.0, 54.0


def iter_dates(start_yyyymmdd: str, end_yyyymmdd: str | None = None) -> Iterator[str]:
    """Yield YYYYMMDD strings from start through end (inclusive). Single day if end omitted."""
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    if end_yyyymmdd is None:
        yield start_yyyymmdd
        return
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    if end < start:
        raise ValueError(f"--end-date {end_yyyymmdd} 早于 --date {start_yyyymmdd}。")
    day = start
    while day <= end:
        yield day.strftime("%Y%m%d")
        day += timedelta(days=1)


def utc_day_dir(base: Path, date: str) -> Path:
    """Production outputs use ``<date>_utc`` subdirectories."""
    return base / f"{date}_utc"


def default_timestamp_npz(
    date: str,
    *,
    use_prior: bool = True,
    local_overpass_time: str = "10:15",
    prior_window_minutes: int = 120,
    stride: int = 1,
) -> Path:
    """Default step_01 npz path (matches batch runner / plot conventions)."""
    suffix = ""
    if use_prior:
        hour_text, minute_text = local_overpass_time.split(":", maxsplit=1)
        suffix = f"_prior{hour_text}{minute_text}w{prior_window_minutes}"
    suffix += f"_stride{stride}"
    return OUTPUT_TIMESTAMP_DIR / f"fy3c_timestamp_{date}{suffix}.npz"


def resolve_fy3_sz_path(date: str, path: Path | None = None) -> Path:
    if path is not None:
        return path
    for name in (f"Sz_{date}.dat", f"SZ_{date}.dat", f"sz_{date}.dat"):
        candidate = DEFAULT_FY3_ROOT / name
        if candidate.exists():
            return candidate
    return DEFAULT_FY3_ROOT / f"Sz_{date}.dat"


def resolve_fy3_gc_path(date: str, path: Path | None = None) -> Path:
    if path is not None:
        return path
    candidate = DEFAULT_FY3_ROOT / f"Gc_{date}.dat"
    if candidate.exists():
        return candidate
    return candidate


def default_fy3_path(date: str = "20180501") -> Path:
    return DEFAULT_FY3_ROOT / f"Gc_{date}.dat"


def default_fy4_path(date: str = "20180501") -> Path:
    fy4_dir = DEFAULT_FY4_ROOT / date
    candidates = sorted(fy4_dir.glob("*.NC")) + sorted(fy4_dir.glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"未在 {fy4_dir} 找到 FY-4A NC 文件。")
    return candidates[0]


def read_fy3_binary(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"FY-3C 文件不存在: {path}")
    expected = int(np.prod(FY3_SHAPE))
    data = np.fromfile(path, dtype="<f4", count=expected)
    if data.size != expected:
        raise ValueError(f"{path} 期望 {expected} 个 float32，实际 {data.size}。")
    return data.reshape(FY3_SHAPE)


def read_fy4_nc_global_attrs(path: Path) -> dict[str, Any]:
    """读取 FY-4A NC 全局属性（netCDF4 / xarray）。"""
    if not path.exists():
        raise FileNotFoundError(f"FY-4A 文件不存在: {path}")

    try:
        from netCDF4 import Dataset
    except ImportError:
        Dataset = None

    if Dataset is not None:
        with Dataset(path) as ds:
            return {name: getattr(ds, name) for name in ds.ncattrs()}

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("读取 FY-4A NC 需要 netCDF4 或 xarray。") from exc

    with xr.open_dataset(path) as ds:
        return dict(ds.attrs)


def read_fy4_version_of_software(path: Path) -> str | None:
    """从 NC 全局属性读取软件版本（``Version Of Software`` / ``version_of_software``）。"""
    attrs = read_fy4_nc_global_attrs(path)
    for key in ("version_of_software", "Version Of Software", "Version_of_Software"):
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def read_fy4_variable(path: Path, variable: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"FY-4A 文件不存在: {path}")

    try:
        from netCDF4 import Dataset
    except ImportError:
        Dataset = None

    if Dataset is not None:
        with Dataset(path) as ds:
            if variable not in ds.variables:
                available = ", ".join(ds.variables.keys())
                raise KeyError(f"变量 {variable!r} 不在 {path} 中。可用: {available}")
            return masked_to_float_array(ds.variables[variable][:])

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("读取 FY-4A NC 需要 netCDF4 或 xarray。") from exc

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            available = ", ".join(ds.data_vars)
            raise KeyError(f"变量 {variable!r} 不在 {path} 中。可用: {available}")
        return masked_to_float_array(ds[variable].values)


def masked_to_float_array(data: Any) -> np.ndarray:
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data, np.nan).astype(np.float32, copy=False)
    return np.asarray(data, dtype=np.float32)


def build_fy4_lon_lat_nsmc(shape: tuple[int, int] = FY4_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """
    用 NSMC/CGMS 静止卫星公式生成 FY-4A 4000 m 全圆盘经纬度（无 raw 查找表时的回退）。

    参数与 satpy FY-4A AGRI 4 km 全圆盘一致（ssp_lon=104.7°E）。
    """
    try:
        from pyresample.geometry import AreaDefinition
    except ImportError as exc:
        raise RuntimeError("生成 FY-4A 经纬度需要 pyresample。") from exc

    nrows, ncols = shape
    cols0, rows0 = 0.5, 0.5

    def scan_to_radian(line: float, col: float) -> tuple[float, float]:
        x = float(col - FY4_NSMC_COFF) / (FY4_NSMC_CFAC / 2**16)
        y = float(line - FY4_NSMC_LOFF) / (FY4_NSMC_LFAC / 2**16)
        return x, y

    ll_x, ll_y = scan_to_radian(rows0, cols0)
    ur_x, ur_y = scan_to_radian(rows0 + nrows, cols0 + ncols)
    area_extent = (
        np.deg2rad(ll_x) * FY4_NSMC_H,
        np.deg2rad(ll_y) * FY4_NSMC_H,
        np.deg2rad(ur_x) * FY4_NSMC_H,
        np.deg2rad(ur_y) * FY4_NSMC_H,
    )
    proj = {
        "a": FY4_NSMC_A,
        "b": FY4_NSMC_B,
        "lon_0": FY4_NSMC_SSP_LON,
        "h": FY4_NSMC_H,
        "proj": "geos",
        "units": "m",
        "sweep": "y",
    }
    area = AreaDefinition("fy4a_agri_4000m", "FY-4A AGRI 4km", "fy4", proj, ncols, nrows, area_extent)
    lons, lats = area.get_lonlats()
    lon = np.asarray(lons, dtype=np.float64)
    lat = np.asarray(lats, dtype=np.float64)
    lon[~np.isfinite(lon) | (np.abs(lon) > 180.0)] = np.nan
    lat[~np.isfinite(lat) | (np.abs(lat) > 90.0)] = np.nan
    return lon, lat


def read_fy4_lon_lat(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取 FY-4A 经纬度 raw；文件不存在时回退 NSMC/CGMS 公式。"""
    geo_path = path or DEFAULT_FY4_GEO_RAW
    if geo_path.exists():
        expected = int(np.prod(FY4_SHAPE) * 2)
        raw = np.fromfile(geo_path, dtype="<f8", count=expected)
        if raw.size != expected:
            raise ValueError(f"{geo_path} 期望 {expected} 个 float64，实际 {raw.size}。")
        grid = raw.reshape(FY4_SHAPE[0], FY4_SHAPE[1], 2)
        lat = clean_geo_channel(grid[:, :, 0])
        lon = clean_geo_channel(grid[:, :, 1])
        return lon, lat
    return build_fy4_lon_lat_nsmc()


def clean_geo_channel(data: np.ndarray) -> np.ndarray:
    clean = data.astype(np.float64, copy=True)
    clean[np.isclose(clean, INVALID_GEO_FILL) | (np.abs(clean) > 9999.0)] = np.nan
    return clean


def clean_ssi(
    data: np.ndarray,
    min_valid_ssi: float = SSI_VALID_MIN,
    max_valid_ssi: float = SSI_VALID_MAX,
) -> np.ndarray:
    """
    清洗 FY-4A SSI：剔除填充值与越界值。

    无效像元规则（FY4_SPEC_PDF + 20180501 V1.0 NC 实测）：
    - ``-999``：缺测
    - ``65532``：SZA 超阈值无效（**V1.0：>70°**；**V1.1 文档：>90°**）
    - ``65535``：圆盘外
    - ``< 0`` 或 ``> 1500`` W/m²：超出 ``valid_range``
    """
    clean = data.astype(np.float32, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    clean[np.abs(clean) > 1.0e20] = np.nan
    for fill in SSI_INVALID_FILLS:
        clean[np.isclose(clean, fill, rtol=0.0, atol=0.0)] = np.nan
    clean[clean < min_valid_ssi] = np.nan
    clean[clean > max_valid_ssi] = np.nan
    return clean


def apply_dqf_mask(
    ssi: np.ndarray,
    dqf: np.ndarray,
    *,
    mask_dqf3: bool = True,
    mask_dqf2: bool = False,
) -> np.ndarray:
    """
    按 DQF 掩膜 SSI。

    策略（FY4_SPEC_PDF）：``DQF==3`` 始终剔除；``DQF==2`` 可选剔除。
    20180501 V1.0 NC 中 DQF 常全为 3（未正确分级）——此时跳过 DQF 掩膜，
    改由 ``clean_ssi`` 的 65532/65535 规则处理无效像元。
    """
    masked = ssi.astype(np.float32, copy=True)
    dqf_arr = np.asarray(dqf)
    unique = np.unique(dqf_arr)
    if mask_dqf3 and unique.size == 1 and unique[0] == 3:
        return masked

    bad = np.zeros(dqf_arr.shape, dtype=bool)
    if mask_dqf3:
        bad |= dqf_arr == 3
    if mask_dqf2:
        bad |= dqf_arr == 2
    masked[bad] = np.nan
    return masked


def read_fy4_ssi_with_dqf(
    path: Path,
    *,
    variable: str = "SSI",
    dqf_variable: str = "DQF",
    mask_dqf3: bool = True,
    mask_dqf2: bool = False,
    min_valid_ssi: float = SSI_VALID_MIN,
    max_valid_ssi: float = SSI_VALID_MAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    读取 FY-4A SSI 并应用 ``clean_ssi`` + DQF 掩膜。

    返回 ``(ssi_clean, meta)``；``meta`` 含 ``version_of_software``、源路径等。
    """
    if not path.exists():
        raise FileNotFoundError(f"FY-4A 文件不存在: {path}")

    try:
        from netCDF4 import Dataset
    except ImportError:
        Dataset = None

    dqf: np.ndarray | None = None
    if Dataset is not None:
        with Dataset(path) as ds:
            if variable not in ds.variables:
                available = ", ".join(ds.variables.keys())
                raise KeyError(f"变量 {variable!r} 不在 {path} 中。可用: {available}")
            ssi_raw = masked_to_float_array(ds.variables[variable][:])
            if dqf_variable in ds.variables:
                dqf = np.asarray(ds.variables[dqf_variable][:])
    else:
        try:
            import xarray as xr
        except ImportError as exc:
            raise RuntimeError("读取 FY-4A NC 需要 netCDF4 或 xarray。") from exc
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                available = ", ".join(ds.data_vars)
                raise KeyError(f"变量 {variable!r} 不在 {path} 中。可用: {available}")
            ssi_raw = masked_to_float_array(ds[variable].values)
            if dqf_variable in ds:
                dqf = np.asarray(ds[dqf_variable].values)

    ssi = clean_ssi(ssi_raw, min_valid_ssi=min_valid_ssi, max_valid_ssi=max_valid_ssi)
    if dqf is not None:
        ssi = apply_dqf_mask(ssi, dqf, mask_dqf3=mask_dqf3, mask_dqf2=mask_dqf2)

    meta: dict[str, Any] = {
        "source_nc": str(path),
        "version_of_software": read_fy4_version_of_software(path),
        "mask_dqf3": mask_dqf3,
        "mask_dqf2": mask_dqf2,
    }
    return ssi, meta


def fy3_extent() -> tuple[float, float, float, float]:
    rows, cols = FY3_SHAPE
    lon_min = FY3_LON_START
    lon_max = FY3_LON_START + cols * FY3_STEP
    lat_max = FY3_LAT_START
    lat_min = FY3_LAT_START - rows * FY3_STEP
    return lon_min, lon_max, lat_min, lat_max


def build_target_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    step: float = FY3_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col_start = int(round((lon_min - FY3_LON_START) / step))
    col_end = int(round((lon_max - FY3_LON_START) / step)) + 1
    row_start = int(round((FY3_LAT_START - lat_max) / step))
    row_end = int(round((FY3_LAT_START - lat_min) / step)) + 1
    col_end = min(col_end, FY3_SHAPE[1])
    row_end = min(row_end, FY3_SHAPE[0])
    ncols = col_end - col_start
    nrows = row_end - row_start
    lons_1d = FY3_LON_START + (col_start + np.arange(ncols)) * step
    lats_1d = FY3_LAT_START - (row_start + np.arange(nrows)) * step
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
    return lons_1d, lats_1d, lons_2d, lats_2d


def parse_fy4_nc_start_minute(path: Path, date: str | None = None) -> int:
    """从 FY-4A NC 文件名解析起始观测时刻（当天 UTC 分钟数 0–1439）。"""
    match = FY4_NC_TIME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析时间: {path.name}")
    stamp = match.group(1)
    if date is not None and not stamp.startswith(date):
        raise ValueError(f"文件名日期 {stamp[:8]} 与期望 {date} 不一致: {path.name}")
    hour = int(stamp[8:10])
    minute = int(stamp[10:12])
    return hour * 60 + minute


def parse_fy4_nc_utc_label(path: Path, date: str | None = None) -> str:
    """从 FY-4A NC 文件名解析 UTC 时次标签 HHMM。"""
    return minute_to_hhmm(parse_fy4_nc_start_minute(path, date))


def cst_minutes_to_utc_minutes(minutes: int | np.ndarray) -> int | np.ndarray:
    """北京时间/CST 当天分钟数 → UTC 当天分钟数（模 1440）。"""
    return (np.asarray(minutes, dtype=np.int32) - CST_OFFSET_MINUTES) % 1440


def utc_minutes_to_cst_minutes(minutes: int | np.ndarray) -> int | np.ndarray:
    """UTC 当天分钟数 → CST 当天分钟数（模 1440）。"""
    return (np.asarray(minutes, dtype=np.int32) + CST_OFFSET_MINUTES) % 1440


def utc_label_to_cst_label(utc_label: str) -> str:
    """UTC HHMM 标签 → CST HHMM 标签（同日白天场景，无 +1 后缀）。"""
    utc_minute = parse_hhmm(utc_label)
    return minute_to_hhmm(int(utc_minutes_to_cst_minutes(utc_minute)))


def dual_time_label(utc_label: str) -> str:
    """UTC 主标签 + CST 副标签，如 ``0000 (08:00 CST)``。"""
    cst = utc_label_to_cst_label(utc_label)
    return f"{utc_label} ({cst[:2]}:{cst[2:]} CST)"


def parse_hhmm(value: str) -> int:
    """解析 HH:MM 或 HHMM 为当天分钟数（0–1439）。"""
    text = value.strip()
    if ":" in text:
        hour_text, minute_text = text.split(":", maxsplit=1)
    elif len(text) == 4 and text.isdigit():
        hour_text, minute_text = text[:2], text[2:]
    else:
        raise ValueError(f"无法解析时间: {value!r}，请使用 HH:MM 或 HHMM。")
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"时间超出范围: {value}")
    return hour * 60 + minute


def list_fy4_nc_files_in_utc_window(
    date: str,
    start: str = "00:00",
    end: str = "06:00",
    fy4_root: Path | None = None,
) -> list[tuple[Path, int]]:
    """列出某日 FY-4A NC，筛选文件名起始时刻在 [start, end] UTC 内（含端点）。"""
    start_minute = parse_hhmm(start)
    end_minute = parse_hhmm(end)
    if end_minute < start_minute:
        raise ValueError(f"结束时间 {end} 早于开始时间 {start}。")
    return [
        (path, minute)
        for path, minute in list_fy4_nc_files(date, fy4_root)
        if start_minute <= minute <= end_minute
    ]


# 旧名保留，指向 UTC 语义（曾误标为 CST）。
list_fy4_nc_files_in_cst_window = list_fy4_nc_files_in_utc_window


def resampled_on_fy3_path(output_dir: Path, time_label: str) -> Path:
    """批量重采样输出：FY4_HHMM_on_FY3.npz。"""
    return output_dir / f"FY4_{time_label}_on_FY3.npz"


def rest2_on_fy3_path(output_dir: Path, time_label: str) -> Path:
    """REST2 GHI 重采样到 FY-3 网格：REST2_HHMM_on_FY3.npz。"""
    return output_dir / f"REST2_{time_label}_on_FY3.npz"


def csi_on_fy3_path(output_dir: Path, time_label: str) -> Path:
    """FY-4 / REST2 晴空指数：CSI_HHMM_on_FY3.npz。"""
    return output_dir / f"CSI_{time_label}_on_FY3.npz"


def rest2_cache_path(cache_dir: Path, date: str, slot: str, grid_step: float) -> Path:
    """REST2 0.5° 计算缓存：rest2_ghi_<date>_<slot>_step<grid_step>.npz。"""
    return cache_dir / f"rest2_ghi_{date}_{slot}_step{grid_step:.2f}.npz"


def discover_fy4_slots_from_resampled(
    resampled_dir: Path,
    window_minutes: int = 15,
) -> list[tuple[str, int, int, str]]:
    """
    从重采样缓存文件名解析 FY-4 时次与半开窗口 [start, start+window_minutes)（UTC）。

    返回 [(utc_label, start_utc_minute, end_utc_minute, cst_label), ...]，按 start 排序。
    """
    if not resampled_dir.exists():
        raise FileNotFoundError(f"重采样目录不存在: {resampled_dir}")

    entries: list[tuple[str, int, int, str]] = []
    for path in sorted(resampled_dir.glob("FY4_*_on_FY3.npz")):
        match = FY4_ON_FY3_NPZ_PATTERN.match(path.name)
        if match is None:
            continue
        utc_label = match.group(1)
        start = parse_hhmm(utc_label)
        cst_label = minute_to_hhmm(int(utc_minutes_to_cst_minutes(start)))
        entries.append((utc_label, start, start + window_minutes, cst_label))

    if not entries:
        raise FileNotFoundError(f"未在 {resampled_dir} 找到 FY4_*_on_FY3.npz。")
    entries.sort(key=lambda item: item[1])
    return entries


def match_pixels_to_half_open_windows(
    best_minute: np.ndarray,
    valid: np.ndarray,
    slot_starts_utc: np.ndarray,
    window_minutes: int = 15,
    best_minute_tz: str = "CST",
) -> np.ndarray:
    """
    师兄匹配：best_minute 落在 FY-4 半开窗口 [start, start+window) 的像元赋 slot 索引；否则 -1。
    不使用最近邻。

    FY-4 窗口由文件名 UTC 时次定义（``slot_starts_utc``）。``best_minute`` 默认为 step_01
    输出的 CST；会在匹配前转为 UTC 与窗口比较。
    """
    slot_map = np.full(best_minute.shape, -1, dtype=np.int16)
    if not np.any(valid):
        return slot_map

    minutes = best_minute.astype(np.int32)
    if best_minute_tz.upper() == "CST":
        minutes = cst_minutes_to_utc_minutes(minutes).astype(np.int32)
    elif best_minute_tz.upper() != "UTC":
        raise ValueError(f"不支持的 best_minute_tz: {best_minute_tz!r}，请用 CST 或 UTC。")

    for slot_idx, start in enumerate(slot_starts_utc):
        end = int(start) + window_minutes
        in_window = valid & (minutes >= start) & (minutes < end)
        slot_map[in_window] = slot_idx
    return slot_map


def list_fy4_nc_files(
    date: str,
    fy4_root: Path | None = None,
) -> list[tuple[Path, int]]:
    """列出某日全部 FY-4A NC 文件，按文件名起始 UTC 分钟排序。"""
    fy4_dir = (fy4_root or DEFAULT_FY4_ROOT) / date
    if not fy4_dir.exists():
        raise FileNotFoundError(f"FY-4A 目录不存在: {fy4_dir}")
    files = sorted({path.resolve() for path in fy4_dir.glob("*.NC")} | {path.resolve() for path in fy4_dir.glob("*.nc")})
    files = [Path(path) for path in files]
    if not files:
        raise FileNotFoundError(f"未在 {fy4_dir} 找到 FY-4A NC 文件。")
    entries = [(path, parse_fy4_nc_start_minute(path, date)) for path in files]
    entries.sort(key=lambda item: (item[1], item[0].name))
    return entries


def minute_to_hhmm(minute_of_day: int | float) -> str:
    if not np.isfinite(minute_of_day):
        return "NA"
    minute = int(round(float(minute_of_day)))
    return f"{minute // 60:02d}{minute % 60:02d}"


def region_slice(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    step: float = FY3_STEP,
) -> tuple[int, int, int, int]:
    col_start = int(round((lon_min - FY3_LON_START) / step))
    col_end = int(round((lon_max - FY3_LON_START) / step)) + 1
    row_start = int(round((FY3_LAT_START - lat_max) / step))
    row_end = int(round((FY3_LAT_START - lat_min) / step)) + 1
    col_end = min(col_end, FY3_SHAPE[1])
    row_end = min(row_end, FY3_SHAPE[0])
    return row_start, row_end, col_start, col_end


def region_tag(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> str:
    """根据目标区域生成输出文件名后缀。"""
    if (
        lon_min <= FY3_LON_START + 0.01
        and lon_max >= FY3_LON_START + FY3_SHAPE[1] * FY3_STEP - 0.01
        and lat_min <= FY3_LAT_START - FY3_SHAPE[0] * FY3_STEP + 0.01
        and lat_max >= FY3_LAT_START - 0.01
    ):
        return "full_region"
    return f"{lon_min:.0f}_{lon_max:.0f}E_{lat_min:.0f}_{lat_max:.0f}N"
