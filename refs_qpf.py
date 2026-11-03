"""
REFS QPF site builder
=====================

Everything specific to one office or range (map extent, stations, branding,
flow-regime sounding) lives in site_config.json; this file is identical across
deployments. Stations can come straight from IEM's ASOS/AWOS network, filtered
to one WFO, so the list tracks commissioning and retirement automatically.

Static GitHub Pages site from operational REFS ensemble products, running
verification against MRMS, sea-breeze flow-regime tagging, and event look-backs
(see lookback.py).

Inputs (.../refs.YYYYMMDD/CC/ensprod/refs.tCCz.<type>.fFF.conus.grib2)
  lpmm  1-h localized PMM QPF                 latest + 2 time-lagged runs
  avrg  1-h PMM/mean blend QPF                latest run
  eas   6-h exceedance probabilities          latest run
  ffri  6-h P(>FFG), P(>N-yr ARI)             latest run
  mean  UGRD/VGRD 10 m, 925, 850, 700 mb      flow regime (byte-range via .idx)
  MRMS  MultiSensor_QPE_<N>H_Pass2            verification

Sources (confirmed 2026-11-03)
  REFS  NOMADS refs/prod (~2 days), then AWS noaa-rrfs-ops-pds (since 2026-08-13)
  MRMS  mrms.ncep.noaa.gov/2D (~25 h), then AWS noaa-mrms-pds (long-term)

Persistent state lives in ./data, which the workflow checks out from and pushes
to the `data` branch: archive/ (forecasts, scores, regime log), cases/
(look-backs), site_cache/ (last published site, used by the look-back workflow).
"""
import matplotlib
matplotlib.use("Agg")

import glob
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pygrib
import requests
from matplotlib.lines import Line2D
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# 1. CONFIG
# =============================================================================
NOMADS_REFS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/refs/prod"
AWS_REFS = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
REFS_PATH = "refs.{date}/{hh}/ensprod/refs.t{hh}z.{ptype}.f{fhr:02d}.conus.grib2"
MRMS_NAME = "MRMS_MultiSensor_QPE_{h:02d}H_Pass2_00.00_{t:%Y%m%d-%H}0000.grib2.gz"
MRMS_NCEP = "https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_{h:02d}H_Pass2/" + MRMS_NAME
MRMS_AWS = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_{h:02d}H_Pass2_00.00/{t:%Y%m%d}/" + MRMS_NAME
RAOB_URL = "https://mesonet.agron.iastate.edu/json/raob.py?station={stn}&ts={t:%Y-%m-%dT%H:00:00Z}"

OUT_DIR = "site"
DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
FCST_DIR = os.path.join(ARCHIVE_DIR, "fcst")
SCORES_FILE = os.path.join(ARCHIVE_DIR, "scores.json")
VERIFY_LATEST = os.path.join(ARCHIVE_DIR, "verify_latest.npz")
REGIME_LOG = os.path.join(ARCHIVE_DIR, "regime_log.json")
CASES_DIR = os.path.join(DATA_DIR, "cases")
SITE_CACHE = os.path.join(DATA_DIR, "site_cache")
LEGACY_ARCHIVE = "archive"          # v2 kept the archive on main; migrated once
INDEX_SRC = "index.html"

EXTENT = [-83.0, -79.0, 26.5, 30.5]
CROP_PAD = 0.5

ACCUM_HOURS = 48
N_RUNS = 3
RUN_SPACING = 6
MAX_FHR = 60
WIN = 6
THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
EAS_THRESHOLDS = [0.25, 0.5, 1.0, 2.0]
FFRI_ARI_YEARS = [2, 10]
VERIFY_THRESHOLDS = [0.5, 1.0, 2.0]

EXPECTED_LATENCY = timedelta(hours=3, minutes=15)
POLL_INTERVAL_S = 300
MAX_POLLS = 12
MRMS_LATENCY = timedelta(minutes=70)
MRMS_NCEP_RETENTION = timedelta(hours=23)
FCST_MAX_AGE = timedelta(days=30)   # give up scoring after this

# ---- flow regime (simple quadrant scheme; edit to match squadron practice)
REGIME_POINT = (28.4676, -80.5669)  # XMR
REGIME_RADIUS_KM = 25.0
REGIME_LEVELS = [("10 m above ground", 1013.0), ("925 mb", 925.0), ("850 mb", 850.0), ("700 mb", 700.0)]
LIGHT_KT = 5.0                      # layer-mean speed below this = light/variable
MIXED_CONSISTENCY = 0.5             # |vector mean| / mean speed below this = mixed
RAOB_STATIONS = ["KXMR", "XMR"]     # unverified IEM IDs; first that returns data wins
RAOB_HOURS = [10, 12, 22, 0]        # tried per day inside the window
RAOB_BACKFILL_PER_RUN = 4
REGIME_MIN_N = 5                    # runs needed before showing a by-regime comparison

AUTO_DAILY_CASE = True              # build a 24-h look-back on each 12Z cycle
AUTO_CASES_KEPT = 14

MAX_WORKERS = 4
NBHD_KM = 10.0
MM_TO_IN = 0.0393701
MS_TO_KT = 1.94384

# Launch-complex coordinates are approximate (±0.005°, well under one 3-km grid box)
SITES = {
    "LC-39A":             (28.6082, -80.6041, "*"),
    "LC-39B":             (28.6272, -80.6208, "*"),
    "SLC-40":             (28.5619, -80.5772, "*"),
    "SLC-41":             (28.5833, -80.5830, "*"),
    "SLC-37":             (28.5317, -80.5650, "*"),
    "LC-36":              (28.4706, -80.5378, "*"),
    "SLC-46":             (28.4584, -80.5284, "*"),
    "Skid Strip (KXMR)":  (28.4676, -80.5669, "^"),
    "SLF (KTTS)":         (28.6150, -80.6945, "^"),
    "Patrick (KCOF)":     (28.2349, -80.6101, "^"),
    "Melbourne (KMLB)":   (28.1028, -80.6453, "o"),
    "Palm Bay":           (28.0345, -80.5887, "o"),
    "Orlando (KMCO)":     (28.4294, -81.3089, "o"),
    "Daytona (KDAB)":     (29.1799, -81.0581, "o"),
}
PRIMARY_SITE = "LC-39A"
TIMELINE_SITES = ["LC-39A", "LC-39B", "SLC-40", "SLC-41", "SLC-37", "LC-36", "SLC-46",
                  "Skid Strip (KXMR)", "SLF (KTTS)", "Patrick (KCOF)"]
MAP_LABELS = ["LC-39A", "Skid Strip (KXMR)", "Melbourne (KMLB)", "Orlando (KMCO)", "Daytona (KDAB)"]
SITE_LABELS = {}                    # name -> short map label
BRANDING = {"title": "Space Coast REFS QPF",
            "subtitle": "48-h ensemble QPF for Cape Canaveral and Kennedy Space Center",
            "area": "Cape Canaveral and KSC"}
REGIME_POINT_NAME = "XMR"
IEM_STATIONS = None                 # {"network": "NC_ASOS", "wfo": "RAH", "major": [...]}
CONFIG_FILE = "site_config.json"
STATIONS_CACHE = os.path.join("data", "stations.json")
IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network.py?network={network}"
_SITES_READY = False
SITE_IDX = {}


def _site_entry(name, lat, lon, kind="site", label=None):
    marker = {"major": "*", "asos": "^", "awos": "o"}.get(kind, "^" if kind == "site" else "o")
    SITES[name] = (float(lat), float(lon), marker)
    SITE_LABELS[name] = label or name.split(" (")[0]


def _resolve(key):
    """Config refers to sites by display name or station id (e.g. 'RDU')."""
    if key in SITES:
        return key
    for n in SITES:
        if n.endswith(f"(K{key})") or n.endswith(f"({key})") or SITE_LABELS.get(n) == key:
            return n
    return None


def _iem_stations(cfg):
    """Online ASOS/AWOS for one WFO from IEM; cached so a later IEM outage reuses the last list."""
    url = IEM_NETWORK_URL.format(network=cfg["network"])
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        feats = r.json().get("features", [])
        out = []
        for f in feats:
            p = f.get("properties", {})
            if p.get("wfo") != cfg["wfo"] or not p.get("online", True):
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            awos = str((p.get("attributes") or {}).get("IS_AWOS", "0")) == "1"
            out.append({"id": p.get("sid") or f.get("id"), "sname": p.get("sname", ""), "lat": lat, "lon": lon,
                        "awos": awos, "county": p.get("county")})
        if out:
            os.makedirs(os.path.dirname(STATIONS_CACHE), exist_ok=True)
            with open(STATIONS_CACHE, "w") as fh:
                json.dump({"fetched": utcnow().isoformat(timespec="seconds"), "stations": out}, fh, indent=1)
            print(f"[stations] {len(out)} online {cfg['wfo']} stations from IEM {cfg['network']}")
            return out
        print(f"[stations] IEM returned no {cfg['wfo']} stations")
    except Exception as e:
        print(f"[stations] IEM fetch failed: {e}")
    try:
        with open(STATIONS_CACHE) as fh:
            cached = json.load(fh)
        print(f"[stations] using cached list from {cached.get('fetched')}")
        return cached["stations"]
    except Exception:
        return None


def _pretty(sname):
    words = sname.replace("_", " ").split()
    keep = {"AFB", "AAF", "ASOS", "AWOS", "NAS", "RDU", "II"}
    cap = lambda w: w if w in keep else "-".join(x.capitalize() for x in w.split("-"))
    return " ".join(cap(w) for w in words)


def init_sites():
    """Apply site_config.json (once). Safe to call from lookback and request scripts."""
    global SITES, EXTENT, PRIMARY_SITE, TIMELINE_SITES, MAP_LABELS, THRESHOLDS, BRANDING
    global REGIME_POINT, REGIME_POINT_NAME, RAOB_STATIONS, RAOB_HOURS, IEM_STATIONS, _SITES_READY
    if _SITES_READY:
        return
    _SITES_READY = True
    if not os.path.exists(CONFIG_FILE):
        for n in SITES:
            SITE_LABELS.setdefault(n, n.split(" (")[0])
        return
    with open(CONFIG_FILE) as fh:
        cfg = json.load(fh)
    BRANDING = {**BRANDING, **cfg.get("branding", {})}
    EXTENT = cfg.get("extent", EXTENT)
    THRESHOLDS = cfg.get("thresholds", THRESHOLDS)
    reg = cfg.get("regime", {})
    REGIME_POINT = tuple(reg.get("point", REGIME_POINT))
    REGIME_POINT_NAME = reg.get("name", REGIME_POINT_NAME)
    RAOB_STATIONS = reg.get("raob_stations", RAOB_STATIONS)
    RAOB_HOURS = reg.get("raob_hours", RAOB_HOURS)
    IEM_STATIONS = cfg.get("iem_stations")

    SITES = {}
    SITE_LABELS.clear()
    for s in cfg.get("sites", []):
        _site_entry(s["name"], s["lat"], s["lon"], s.get("kind", "site"), s.get("label"))
    if IEM_STATIONS:
        stations = _iem_stations(IEM_STATIONS)
        if stations is None:
            stations = [dict(x, awos=x.get("kind") == "awos") for x in cfg.get("fallback_stations", [])]
            print(f"[stations] IEM and cache unavailable; using {len(stations)} fallback stations from config")
        major = set(IEM_STATIONS.get("major", []))
        for st in sorted(stations, key=lambda x: (x["id"] not in major, x["id"])):
            sid = st["id"]
            kind = "major" if sid in major else ("awos" if st.get("awos") else "asos")
            name = f"{_pretty(st.get('sname') or sid)} (K{sid})" if len(sid) == 3 else f"{_pretty(st.get('sname') or sid)} ({sid})"
            _site_entry(name, st["lat"], st["lon"], kind, label=sid)
    PRIMARY_SITE = _resolve(cfg.get("primary", "")) or next(iter(SITES))
    tl = cfg.get("timelines", "all")
    TIMELINE_SITES = list(SITES) if tl == "all" else [n for n in (_resolve(k) for k in tl) if n]
    MAP_LABELS = [n for n in (_resolve(k) for k in cfg.get("map_labels", [])) if n]
    SITE_IDX.clear()
    print(f"[config] {BRANDING['title']}: {len(SITES)} sites, primary {PRIMARY_SITE}, extent {EXTENT}")

# =============================================================================
# 2. STYLE
# =============================================================================
BG, LAND, OCEAN, LINE, TEXT, MUTED = "#0f1a24", "#1c2a36", "#101d28", "#5b7183", "#e6edf2", "#8ea3b3"

CLEVS = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
CMAP_COLORS = [
    (0.31, 0.81, 0.81), (0, 1, 1), (0, 0.87, 0.5), (0, 0.75, 0), (0.5, 0.87, 0),
    (1, 1, 0), (1, 0.62, 0), (1, 0, 0), (1, 0.12, 0.5), (0.94, 0.25, 1),
    (0.5, 0.12, 1), (0.25, 0.25, 1), (0.12, 0.12, 0.5), (0.35, 0.35, 0.35),
    (0.5, 0.5, 0.5), (0.87, 0.87, 0.87), (0.93, 0.83, 0.73),
]
CMAP = mcolors.ListedColormap(CMAP_COLORS, "precip")
CMAP.set_over((0.85, 0.65, 0.47))
NORM = mcolors.BoundaryNorm(CLEVS, CMAP.N)

PLEVS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PCMAP = mcolors.ListedColormap(plt.get_cmap("plasma")(np.linspace(0.12, 1.0, len(PLEVS) - 1)), "prob")
PNORM = mcolors.BoundaryNorm(PLEVS, PCMAP.N)

DLEVS = [-3, -2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2, 3]
DCMAP = plt.get_cmap("BrBG", len(DLEVS) + 1)
DNORM = mcolors.BoundaryNorm(DLEVS, DCMAP.N, extend="both")

RUN_COLORS = ["#4fd1e8", "#f0b429", "#e0607e"]
RUN_LABELS = ["Latest", "−6 h", "−12 h"]
AVRG_COLOR = "#b59cf5"
EAS_COLORS = {0.25: "#9fb3c2", 0.5: "#4fd1e8", 1.0: "#f0b429", 2.0: "#e0607e"}

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": LINE,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "font.family": "DejaVu Sans",
})

# =============================================================================
# 3. NETWORK
# =============================================================================
def make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS + 2))
    s.headers["User-Agent"] = "refs-qpf/3.0 (GitHub Actions)"
    return s


SESSION = make_session()


def utcnow():
    return datetime.now(timezone.utc)


def refs_urls(init, ptype, fhr):
    p = REFS_PATH.format(date=init.strftime("%Y%m%d"), hh=init.strftime("%H"), ptype=ptype, fhr=fhr)
    urls = [f"{NOMADS_REFS}/{p}", f"{AWS_REFS}/{p}"]
    return urls if utcnow() - init < timedelta(hours=40) else urls[::-1]


def exists(url):
    try:
        return SESSION.head(url, timeout=20).status_code == 200
    except requests.RequestException:
        return False


def _tmp(suffix=".grib2"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def download_any(urls, gunzip=False):
    """First URL that returns 200 wins. Raises FileNotFoundError listing all attempts."""
    errs = []
    for url in urls:
        try:
            r = SESSION.get(url, timeout=120)
        except requests.RequestException as e:
            errs.append(f"{e.__class__.__name__} {url}")
            continue
        if r.status_code != 200:
            errs.append(f"{r.status_code} {url}")
            continue
        path = _tmp()
        with open(path, "wb") as f:
            f.write(gzip.decompress(r.content) if gunzip else r.content)
        return path
    raise FileNotFoundError("; ".join(errs))


def download_subset(urls, keep):
    """Byte-range download of the GRIB records whose .idx line satisfies keep(line)."""
    errs = []
    for url in urls:
        try:
            r = SESSION.get(url + ".idx", timeout=30)
            if r.status_code != 200:
                errs.append(f"{r.status_code} {url}.idx")
                continue
            lines = [l for l in r.text.strip().splitlines() if l.count(":") >= 5]
            offs = [int(l.split(":")[1]) for l in lines]
            spans = []
            for k, l in enumerate(lines):
                if keep(l):
                    end = offs[k + 1] - 1 if k + 1 < len(offs) else None
                    if spans and spans[-1][1] is not None and spans[-1][1] + 1 == offs[k]:
                        spans[-1][1] = end
                    else:
                        spans.append([offs[k], end])
            if not spans:
                errs.append(f"no matching records in {url}.idx")
                continue
            path = _tmp()
            with open(path, "wb") as f:
                for a, b in spans:
                    rr = SESSION.get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"}, timeout=120)
                    if rr.status_code not in (200, 206):
                        raise IOError(f"{rr.status_code} range {a}-{b} {url}")
                    f.write(rr.content)
            return path
        except Exception as e:
            errs.append(f"{e.__class__.__name__}: {e} {url}")
    raise FileNotFoundError("; ".join(errs))


def cycle_floor(t):
    return t.replace(hour=(t.hour // RUN_SPACING) * RUN_SPACING, minute=0, second=0, microsecond=0)


def find_latest_cycle():
    for attempt in range(1, MAX_POLLS + 1):
        now = utcnow()
        expected = cycle_floor(now - EXPECTED_LATENCY)
        for c in [cycle_floor(now) - timedelta(hours=RUN_SPACING * k) for k in range(5)]:
            if any(exists(u) for u in refs_urls(c, "lpmm", ACCUM_HOURS)):
                if c >= expected or attempt == MAX_POLLS:
                    return c
                break
        print(f"[wait] expected {expected:%Y%m%d %HZ} incomplete (attempt {attempt}/{MAX_POLLS})")
        time.sleep(POLL_INTERVAL_S)
    return None

# =============================================================================
# 4. GRIB
# =============================================================================
_GRID = {}
_CROPS = {}
_LOGGED = set()


def _step_range(msg):
    try:
        return int(msg.startStep), int(msg.endStep)
    except Exception:
        sr = str(getattr(msg, "stepRange", ""))
        if "-" in sr:
            a, b = sr.split("-")
            return int(a), int(b)
        return None, int(sr) if sr.isdigit() else None


def _limit(msg, which):
    try:
        return float(msg[f"{which}Limit"])
    except Exception:
        return float(msg[f"scaledValueOf{which.capitalize()}Limit"]) * \
               10.0 ** (-float(msg[f"scaleFactorOf{which.capitalize()}Limit"]))


def _lower_limit(msg):
    """Event threshold of a probability record (GRIB2 code table 4.9).

    Types 1 (above upper limit) and 4 (below upper limit) carry the threshold in
    the UPPER limit; 0, 2 and 3 in the lower. NCEP "prob >X" records are
    usually type 1, so reading only the lower limit gives garbage thresholds.
    """
    try:
        ptype = int(msg["probabilityType"])
    except Exception:
        ptype = 3
    return _limit(msg, "upper" if ptype in (1, 4) else "lower")


def _is_apcp(m):
    return getattr(m, "discipline", 0) == 0 and m.parameterCategory == 1 and m.parameterNumber == 8


def _log_inventory(tag, msgs):
    key = tag.split()[0]
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    parts = []
    for m in msgs:
        try:
            ll = f" >{_lower_limit(m):g}"
        except Exception:
            ll = ""
        parts.append(f"#{m.messagenumber} {getattr(m, 'shortName', '?')} {_step_range(m)}{ll}")
    print(f"[inventory {tag}] " + "; ".join(parts))


def make_crop(msg, extent=None, pad=CROP_PAD):
    """Crop description for msg's grid: row/col slices plus cropped lats/lons (lon in -180..180)."""
    w, e, s, n = extent or EXTENT
    key = (msg["Ni"], msg["Nj"], str(msg["gridType"]), round(float(msg["latitudeOfFirstGridPointInDegrees"]), 3),
           w, e, s, n)
    if key in _CROPS:
        return _CROPS[key]
    lats, lons = msg.latlons()
    lons = np.where(lons > 180, lons - 360, lons)
    inside = (lats >= s - pad) & (lats <= n + pad) & (lons >= w - pad) & (lons <= e + pad)
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    rs, cs = slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)
    crop = dict(full=lats.shape, rows=rs, cols=cs, lats=lats[rs, cs], lons=lons[rs, cs])
    _CROPS[key] = crop
    return crop


def _set_grid(msg):
    _GRID.update(make_crop(msg))
    print(f"[grid] REFS full {_GRID['full']} -> cropped {_GRID['lats'].shape}")


def crop_values(msg, crop):
    vals = np.ma.filled(msg.values, 0.0)
    if vals.shape != crop["full"]:
        raise ValueError(f"grid {vals.shape} != {crop['full']}")
    return vals[crop["rows"], crop["cols"]].astype(np.float32)


def _crop(msg):
    if not _GRID:
        _set_grid(msg)
    return crop_values(msg, _GRID)


def _pct(a):
    """Percent, clipped to 0-100. Fractions (max <= 1) are scaled first."""
    a = a * 100.0 if np.nanmax(a) <= 1.0001 else a
    return np.clip(a, 0.0, 100.0)


def _snap(value, choices, tol):
    """Nearest configured threshold within tol, else None (absorbs mm->in rounding)."""
    best = min(choices, key=lambda c: abs(c - value))
    return best if abs(best - value) <= tol else None


DIAG = {"eas": {}, "ffri": {}}


def read_1h_qpf(path, fhr, tag):
    with pygrib.open(path) as g:
        msgs = [m for m in g if _is_apcp(m)]
        if len(msgs) > 1:
            _log_inventory(f"{tag} f{fhr:02d}", msgs)
        pick = next((m for m in msgs if _step_range(m) == (fhr - 1, fhr)), None)
        if pick is None and len(msgs) == 1:
            pick = msgs[0]
        if pick is None:
            raise ValueError(f"no 1-h APCP at f{fhr:02d}: {[_step_range(m) for m in msgs]}")
        return _crop(pick) * MM_TO_IN


def read_eas_6h(path, fhr):
    """{threshold_in: P(%)} for the 6-h window ending at fhr."""
    out, seen = {}, []
    with pygrib.open(path) as g:
        msgs = [m for m in g if _is_apcp(m)]
        _log_inventory(f"eas f{fhr:02d}", msgs)
        for m in msgs:
            if _step_range(m) != (fhr - WIN, fhr):
                continue
            thr_in = _lower_limit(m) * MM_TO_IN
            seen.append(round(thr_in, 3))
            key = _snap(thr_in, EAS_THRESHOLDS + [0.01, 3.0], 0.03)
            if key is not None:
                out[key] = _pct(_crop(m))
    DIAG["eas"][f"f{fhr:02d}"] = {"thresholds_in": sorted(seen), "matched": sorted(k for k in out if k in EAS_THRESHOLDS)}
    if not out:
        print(f"  ! eas f{fhr:02d}: no 6-h records matched; thresholds seen (in) {sorted(seen)}")
    return out


def read_ffri_6h(path, fhr):
    out = {"ffg": None, "ari": {}}
    with pygrib.open(path) as g:
        msgs = list(g)
        _log_inventory(f"ffri f{fhr:02d}", msgs)
        for m in msgs:
            if _step_range(m) != (fhr - WIN, fhr):
                continue
            if _is_apcp(m):
                out["ari"][int(round(_lower_limit(m)))] = _pct(_crop(m))
            else:
                out["ffg"] = _pct(_crop(m))
    DIAG["ffri"][f"f{fhr:02d}"] = {"ari_years": sorted(out["ari"]), "ffg": out["ffg"] is not None}
    return out


def fetch_decoded(init, ptype, fhr, reader):
    path = download_any(refs_urls(init, ptype, fhr))
    try:
        return reader(path, fhr)
    finally:
        os.remove(path)


def fetch_many(init, ptype, fhrs, reader):
    out = {}
    first, rest = fhrs[0], fhrs[1:]
    try:
        out[first] = fetch_decoded(init, ptype, first, reader)
    except Exception as e:
        print(f"  ! {ptype} {init:%d/%HZ} f{first:02d}: {e}")
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_decoded, init, ptype, f, reader): f for f in rest}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                out[f] = fut.result()
            except Exception as e:
                print(f"  ! {ptype} {init:%d/%HZ} f{f:02d}: {e}")
    return out


def fetch_qpf_hours(init, ptype, fhrs):
    got = fetch_many(init, ptype, fhrs, lambda p, f: read_1h_qpf(p, f, ptype))
    if len(got) != len(fhrs):
        print(f"  skip {ptype} {init:%Y%m%d %HZ}: {len(got)}/{len(fhrs)} hours")
        return None
    return np.stack([got[f] for f in fhrs])


def fetch_qpf_run(init, ptype, lag):
    return fetch_qpf_hours(init, ptype, list(range(lag + 1, lag + ACCUM_HOURS + 1)))


def window_hours():
    return list(range(WIN, ACCUM_HOURS + 1, WIN))

# =============================================================================
# 5. POINTS
# =============================================================================

def dist_km(lats, lons, lat, lon):
    dlat, dlon = np.radians(lats - lat), np.radians(lons - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def site_index(name):
    if name not in SITE_IDX:
        lat, lon, _ = SITES[name]
        d = dist_km(_GRID["lats"], _GRID["lons"], lat, lon)
        SITE_IDX[name] = (np.unravel_index(np.argmin(d), d.shape), d <= NBHD_KM)
    return SITE_IDX[name]


def in_view(lats=None, lons=None):
    w, e, s, n = EXTENT
    la = _GRID["lats"] if lats is None else lats
    lo = _GRID["lons"] if lons is None else lons
    return (la >= s) & (la <= n) & (lo >= w) & (lo <= e)


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)

# =============================================================================
# 6. MRMS
# =============================================================================
def fetch_mrms_raw(hours, valid_end, extent=None, pad=1.0):
    """MultiSensor Pass2 QPE (mm) cropped to extent+pad. NCEP while fresh, then AWS archive."""
    urls = []
    if utcnow() - valid_end < MRMS_NCEP_RETENTION:
        urls.append(MRMS_NCEP.format(h=hours, t=valid_end))
    urls.append(MRMS_AWS.format(h=hours, t=valid_end))
    try:
        path = download_any(urls, gunzip=True)
    except FileNotFoundError as e:
        print(f"[mrms] {hours}H {valid_end:%Y%m%d %HZ} unavailable: {e}")
        return None
    try:
        with pygrib.open(path) as g:
            m = g.message(1)
            vals = np.ma.filled(m.values, -999.0).astype(np.float32)
            lat0 = float(m["latitudeOfFirstGridPointInDegrees"])
            lon0 = float(m["longitudeOfFirstGridPointInDegrees"])
            dj = float(m["jDirectionIncrementInDegrees"])
            di = float(m["iDirectionIncrementInDegrees"])
            jpos = int(m["jScansPositively"])
    finally:
        os.remove(path)
    lon0 = lon0 - 360 if lon0 > 180 else lon0
    w, e, s, n = extent or EXTENT
    if jpos:
        r0, r1 = int((s - pad - lat0) / dj), int(math.ceil((n + pad - lat0) / dj))
    else:
        r0, r1 = int((lat0 - (n + pad)) / dj), int(math.ceil((lat0 - (s - pad)) / dj))
    c0, c1 = int((w - pad - lon0) / di), int(math.ceil((e + pad - lon0) / di))
    r0, r1 = max(r0, 0), min(r1, vals.shape[0] - 1)
    c0, c1 = max(c0, 0), min(c1, vals.shape[1] - 1)
    sub = vals[r0:r1 + 1, c0:c1 + 1]
    lat0c = lat0 + r0 * dj if jpos else lat0 - r0 * dj
    lon0c = lon0 + c0 * di
    print(f"[mrms] {hours}H {valid_end:%Y%m%d %HZ}: cropped {sub.shape}, max {np.nanmax(np.where(sub >= 0, sub, np.nan)) * MM_TO_IN:.2f}\"")
    return dict(vals=sub, lat0=lat0c, lon0=lon0c, dj=dj, di=di, jpos=jpos)


MRMS_NATIVE_HOURS = (1, 3, 6, 12, 24, 48, 72)


def fetch_mrms_window(start, end):
    """MRMS Pass2 QPE for any whole-hour window up to 72 h: a native NNH file when
    one matches, otherwise the sum of hourly 01H files."""
    hours = int(round((end - start).total_seconds() / 3600))
    if hours in MRMS_NATIVE_HOURS:
        raw = fetch_mrms_raw(hours, end)
        if raw is not None:
            return raw
    total = None
    for k in range(1, hours + 1):
        r = fetch_mrms_raw(1, start + timedelta(hours=k))
        if r is None:
            print(f"[mrms] hourly sum incomplete at {start + timedelta(hours=k):%d/%HZ}")
            return None
        if total is None:
            total = r
        else:
            both = (total["vals"] >= 0) & (r["vals"] >= 0)
            total["vals"] = np.where(both, total["vals"] + r["vals"], -1.0).astype(np.float32)
    return total


def mrms_native(raw):
    """Observed field (in) on its own 0.01° grid, NaN where missing."""
    nj, ni = raw["vals"].shape
    lat1 = raw["lat0"] + np.arange(nj) * raw["dj"] * (1 if raw["jpos"] else -1)
    lon1 = raw["lon0"] + np.arange(ni) * raw["di"]
    lons, lats = np.meshgrid(lon1, lat1)
    obs = np.where(raw["vals"] >= 0, raw["vals"], np.nan) * MM_TO_IN
    return obs, lats, lons


def grid_spacing_km(lats, lons):
    j, i = lats.shape[0] // 2, lats.shape[1] // 2
    return float(dist_km(np.array(lats[j, i]), np.array(lons[j, i]), lats[j, min(i + 1, lats.shape[1] - 1)],
                         lons[j, min(i + 1, lats.shape[1] - 1)]))


def sample_mrms(raw, lats, lons, half=None):
    """Box-mean MRMS (in) at each target point, NaN where missing.

    half = box half-width in MRMS pixels. Default: matched to the target grid
    spacing (1 px = ~3 km box for 3-km models, ~12 px = ~25 km box for 0.25°),
    so coarse models are scored against area means, not 1-km peaks."""
    if half is None:
        half = max(1, int(round(grid_spacing_km(lats, lons) / 2 / 1.1))) if lats.size > 1 else 1
    if half > 1:
        return _box_mean_sat(raw, lats, lons, half)
    return _sample_3x3(raw, lats, lons)


def _box_mean_sat(raw, lats, lons, half):
    v = raw["vals"]
    ok = v >= 0
    s = np.pad(np.where(ok, v, 0).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    c = np.pad(ok.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rows = np.rint(((lats - raw["lat0"]) if raw["jpos"] else (raw["lat0"] - lats)) / raw["dj"]).astype(int)
    cols = np.rint((lons - raw["lon0"]) / raw["di"]).astype(int)
    r0 = np.clip(rows - half, 0, v.shape[0]); r1 = np.clip(rows + half + 1, 0, v.shape[0])
    c0 = np.clip(cols - half, 0, v.shape[1]); c1 = np.clip(cols + half + 1, 0, v.shape[1])
    tot = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    cnt = c[r1, c1] - c[r0, c1] - c[r1, c0] + c[r0, c0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0.5 * (r1 - r0) * (c1 - c0), tot / np.maximum(cnt, 1), np.nan)
    return out * MM_TO_IN


def _sample_3x3(raw, lats, lons):
    """3x3 (~3 km) box mean of MRMS (in) at each target point; NaN where missing or outside."""
    v = raw["vals"]
    rows = np.rint(((lats - raw["lat0"]) if raw["jpos"] else (raw["lat0"] - lats)) / raw["dj"]).astype(int)
    cols = np.rint((lons - raw["lon0"]) / raw["di"]).astype(int)
    stack = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < v.shape[0]) & (c >= 0) & (c < v.shape[1])
            x = v[np.clip(r, 0, v.shape[0] - 1), np.clip(c, 0, v.shape[1] - 1)]
            stack.append(np.where(ok & (x >= 0), x, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0) * MM_TO_IN


def categorical(f, o, thr):
    fy, oy = f >= thr, o >= thr
    hits, fa, miss = int((fy & oy).sum()), int((fy & ~oy).sum()), int((~fy & oy).sum())
    n = f.size
    hr = (hits + fa) * (hits + miss) / n if n else 0
    den = hits + fa + miss - hr
    ets = (hits - hr) / den if den > 0 else None
    fbias = (hits + fa) / (hits + miss) if (hits + miss) > 0 else None
    return (r2(ets) if ets is not None else None), (r2(fbias) if fbias is not None else None)

# =============================================================================
# 7. FLOW REGIME
# =============================================================================
def _cone(msg):
    l1, l2 = math.radians(float(msg["Latin1InDegrees"])), math.radians(float(msg["Latin2InDegrees"]))
    if abs(l1 - l2) < 1e-6:
        return math.sin(l1)
    return (math.log(math.cos(l1)) - math.log(math.cos(l2))) / \
           (math.log(math.tan(math.pi / 4 + l2 / 2)) - math.log(math.tan(math.pi / 4 + l1 / 2)))


def _earth_relative(msg, u, v, lon):
    """Rotate grid-relative Lambert winds to earth-relative at longitude lon."""
    try:
        grid_rel = int(msg["uvRelativeToGrid"]) == 1
    except Exception:
        grid_rel = False
    if not grid_rel or str(msg["gridType"]) != "lambert":
        return u, v
    ang = _cone(msg) * math.radians(lon - float(msg["LoVInDegrees"]))
    return math.cos(ang) * u + math.sin(ang) * v, -math.sin(ang) * u + math.cos(ang) * v


def _box_mean(msg, lat, lon, radius_km):
    crop = make_crop(msg, extent=[lon - 1, lon + 1, lat - 1, lat + 1], pad=0.2)
    vals = crop_values(msg, crop)
    m = dist_km(crop["lats"], crop["lons"], lat, lon) <= radius_km
    return float(vals[m].mean())


def _level_tag(line):
    for name, _ in REGIME_LEVELS:
        if f":{name}:" in line:
            return name
    return None


def read_layer_wind(path, lat, lon):
    """Pressure-weighted (trapezoid) layer-mean u, v (m/s, earth-relative) over REGIME_LEVELS."""
    comp = {}
    with pygrib.open(path) as g:
        for m in g:
            if m.parameterCategory != 2 or m.parameterNumber not in (2, 3):
                continue
            tl, lev = str(m.typeOfLevel), int(m.level)
            name = "10 m above ground" if (tl == "heightAboveGround" and lev == 10) else \
                   f"{lev} mb" if tl == "isobaricInhPa" else None
            if name is None:
                continue
            comp.setdefault(name, {})["uv"[m.parameterNumber - 2]] = (_box_mean(m, lat, lon, REGIME_RADIUS_KM), m)
    prof = []
    for name, p in REGIME_LEVELS:
        if name in comp and "u" in comp[name] and "v" in comp[name]:
            (u, msg), (v, _) = comp[name]["u"], comp[name]["v"]
            u, v = _earth_relative(msg, u, v, lon)
            prof.append((p, u, v))
    if len(prof) < 2:
        raise ValueError(f"only {len(prof)} levels decoded")
    return _layer_mean(prof)


def _layer_mean(prof):
    prof = sorted(prof, key=lambda x: -x[0])
    num_u = num_v = den = 0.0
    for (p1, u1, v1), (p2, u2, v2) in zip(prof[:-1], prof[1:]):
        dp = p1 - p2
        num_u += 0.5 * (u1 + u2) * dp
        num_v += 0.5 * (v1 + v2) * dp
        den += dp
    return num_u / den, num_v / den


def refs_layer_wind(init, fhr):
    keep = lambda l: (":UGRD:" in l or ":VGRD:" in l) and _level_tag(l) is not None
    path = download_subset(refs_urls(init, "mean", fhr), keep)
    try:
        return read_layer_wind(path, *REGIME_POINT)
    finally:
        os.remove(path)


def classify(samples):
    """samples: [(u, v)] in m/s -> regime dict."""
    if not samples:
        return None
    u = np.array([s[0] for s in samples]); v = np.array([s[1] for s in samples])
    U, V = u.mean(), v.mean()
    vec = math.hypot(U, V)
    mean_spd = float(np.hypot(u, v).mean())
    cons = vec / mean_spd if mean_spd > 0 else 0.0
    dir_from = (270.0 - math.degrees(math.atan2(V, U))) % 360.0
    if mean_spd * MS_TO_KT < LIGHT_KT:
        name = "Light"
    elif cons < MIXED_CONSISTENCY:
        name = "Mixed"
    else:
        name = ["NE", "SE", "SW", "NW"][int(dir_from // 90) % 4]
    return {"name": name, "dir": int(round(dir_from)), "spd_kt": round(vec * MS_TO_KT, 1),
            "consistency": round(cons, 2), "n": len(samples)}


def forecast_regime(init):
    samples = []
    for f in window_hours():
        try:
            samples.append(refs_layer_wind(init, f))
        except Exception as e:
            print(f"  ! regime mean f{f:02d}: {e}")
    reg = classify(samples)
    print(f"[regime] forecast {init:%d/%HZ}: {reg}")
    return reg


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj, indent=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def log_analysis_wind(init):
    """Append the 1-h REFS mean layer wind (analysis proxy) to the regime log."""
    log = load_json(REGIME_LOG, [])
    valid = (init + timedelta(hours=1)).isoformat()
    if any(e["valid"] == valid for e in log):
        return log
    try:
        u, v = refs_layer_wind(init, 1)
        log.append({"valid": valid, "u": round(u, 2), "v": round(v, 2)})
    except Exception as e:
        print(f"  ! analysis wind f01: {e}")
    cutoff = (utcnow() - timedelta(days=45)).isoformat()
    log = sorted([e for e in log if e["valid"] >= cutoff], key=lambda e: e["valid"])
    save_json(REGIME_LOG, log)
    return log


def analysis_regime(log, start, end):
    s = [(e["u"], e["v"]) for e in log if start.isoformat() <= e["valid"] <= end.isoformat()]
    return classify(s) if len(s) >= 3 else None


_RAOB = {"fails": 0}


def raob_layer_wind(t):
    """Layer-mean wind from an IEM sounding. Gives up for the run after repeated HTTP failures."""
    if _RAOB["fails"] >= 8:
        return None, None
    for stn in RAOB_STATIONS:
        try:
            r = SESSION.get(RAOB_URL.format(stn=stn, t=t), timeout=15)
            if r.status_code != 200:
                _RAOB["fails"] += 1
                continue
            _RAOB["fails"] = 0
            profs = r.json().get("profiles") or []
            if not profs or not profs[0].get("profile"):
                continue
            prof = []
            for lv in profs[0]["profile"]:
                p, d, s = lv.get("pres"), lv.get("drct"), lv.get("sknt")
                if None in (p, d, s) or p < 700 or p > 1050:
                    continue
                spd = s / MS_TO_KT
                prof.append((p, -spd * math.sin(math.radians(d)), -spd * math.cos(math.radians(d))))
            if len(prof) >= 3 and min(p for p, _, _ in prof) <= 710:
                return _layer_mean(prof), stn
        except Exception:
            _RAOB["fails"] += 1
            continue
    return None, None


def raob_regime(start, end):
    samples, used = [], []
    d = start.replace(hour=0)
    while d <= end:
        for h in RAOB_HOURS:
            t = d.replace(hour=h)
            if not (start <= t <= end):
                continue
            uv, stn = raob_layer_wind(t)
            if uv:
                samples.append(uv); used.append(f"{stn} {t:%d/%HZ}")
        d += timedelta(days=1)
    reg = classify(samples)
    if reg:
        reg["soundings"] = used
    print(f"[regime] raob {start:%d/%HZ}-{end:%d/%HZ}: {reg if reg else 'no soundings found'}")
    return reg

# =============================================================================
# 8. PLOTTING HELPERS
# =============================================================================
def add_basemap(ax, labels=True):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=OCEAN, edgecolor=LINE, linewidth=0.4, zorder=3)
    try:
        from metpy.plots import USCOUNTIES
        ax.add_feature(USCOUNTIES.with_scale("500k"), edgecolor="#33495a", linewidth=0.35, zorder=3)
    except Exception as e:
        print(f"[warn] counties unavailable: {e}")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#9fb3c2", linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor=LINE, linewidth=0.6, zorder=4)
    for name, (lat, lon, mk) in SITES.items():
        star = mk == "*"
        ax.plot(lon, lat, marker=mk, ms=7 if star else 4, mfc="#ffffff" if star else TEXT,
                mec=BG, mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
        if labels and name in MAP_LABELS:
            ax.text(lon + 0.05, lat + 0.04, SITE_LABELS.get(name, name.split(" (")[0]), fontsize=7, color=TEXT,
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.6))
    ax.spines["geo"].set_edgecolor(LINE)


def shade(ax, field, kind="qpf", lats=None, lons=None):
    lats = _GRID["lats"] if lats is None else lats
    lons = _GRID["lons"] if lons is None else lons
    field = np.nan_to_num(field)
    if kind == "qpf":
        data, lev, cm, nm, ext = np.ma.masked_less(field, CLEVS[0]), CLEVS, CMAP, NORM, "max"
    elif kind == "prob":
        data, lev, cm, nm, ext = np.ma.masked_less(field, PLEVS[0]), PLEVS, PCMAP, PNORM, "neither"
    else:
        data, lev, cm, nm, ext = np.ma.masked_inside(field, -0.1, 0.1), DLEVS, DCMAP, DNORM, "both"
    return ax.contourf(lons, lats, data, lev, cmap=cm, norm=nm, extend=ext,
                       alpha=0.92, transform=ccrs.PlateCarree(), zorder=2)


def colorbar(fig, cs, rect, label, ticks):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8)
    cb.set_label(label, fontsize=10, color=TEXT)
    cb.outline.set_edgecolor(LINE)


def annotate_max(ax, field, unit='"', fmt="{:.2f}", floor=0.1, lats=None, lons=None):
    lats = _GRID["lats"] if lats is None else lats
    lons = _GRID["lons"] if lons is None else lons
    masked = np.where(in_view(lats, lons) & np.isfinite(field), field, -np.inf)
    j, i = np.unravel_index(np.argmax(masked), masked.shape)
    if masked[j, i] >= floor:
        ax.plot(lons[j, i], lats[j, i], "x", color="#ffffff", ms=6, mew=1.4, transform=ccrs.PlateCarree(), zorder=7)
        ax.text(lons[j, i], lats[j, i] - 0.12, fmt.format(field[j, i]) + unit, fontsize=8, ha="center", va="top",
                color="#ffffff", transform=ccrs.PlateCarree(), zorder=7)


def savefig(fig, name, dpi, out_dir=None):
    fig.savefig(os.path.join(out_dir or OUT_DIR, name), dpi=dpi)
    plt.close(fig)


def fmt_init(t):
    return t.strftime("%a %d %b %HZ")

# =============================================================================
# 9. FORECAST PRODUCTS
# =============================================================================
def plot_trend(runs, valid_txt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ax in enumerate(axes):
        add_basemap(ax, labels=(k == 0))
        if k < len(runs):
            tot = runs[k]["data"].sum(axis=0)
            cs = shade(ax, tot)
            annotate_max(ax, tot)
            ax.set_title(f"{RUN_LABELS[k]}  ·  {fmt_init(runs[k]['init'])} init", fontsize=11,
                         color=RUN_COLORS[k], fontweight="bold", pad=6)
        else:
            ax.set_title(f"{RUN_LABELS[k]}  ·  unavailable", fontsize=11, color=MUTED, pad=6)
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Precipitation (in)", CLEVS)
    fig.suptitle(f"REFS LPMM 48-h QPF, run-to-run trend\n{valid_txt}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "trend_48h.png", 170)


def plot_thresholds(runs, valid_txt):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.9, wspace=0.05, hspace=0.1)
    handles = [Line2D([0], [0], marker="s", ls="", ms=10, mfc=RUN_COLORS[k], mec="none",
                      label=f"{RUN_LABELS[k]} ({runs[k]['init']:%d/%HZ})") for k in range(len(runs))]
    for ax, thr in zip(axs.flat, THRESHOLDS):
        add_basemap(ax, labels=False)
        for k in reversed(range(len(runs))):
            tot = runs[k]["data"].sum(axis=0)
            if (tot >= thr).any():
                ax.contourf(_GRID["lons"], _GRID["lats"], tot, levels=[thr, 1e6], colors=[RUN_COLORS[k]],
                            alpha=0.55, transform=ccrs.PlateCarree(), zorder=2)
                ax.contour(_GRID["lons"], _GRID["lats"], tot, levels=[thr], colors=[RUN_COLORS[k]],
                           linewidths=1.1, transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(f"48-h LPMM ≥ {thr:g}\"", fontsize=12, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"REFS LPMM threshold paintball\n{valid_txt}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "thresholds_48h.png", 150)


def _frame(field, kind, title, subtitle, fname, cbar_label, ticks, unit, fmt, floor):
    fig = plt.figure(figsize=(8.4, 9))
    ax = fig.add_axes([0.03, 0.15, 0.94, 0.75], projection=ccrs.PlateCarree())
    add_basemap(ax)
    cs = shade(ax, field, kind)
    annotate_max(ax, field, unit=unit, fmt=fmt, floor=floor)
    fig.text(0.03, 0.955, title, fontsize=14, fontweight="bold")
    fig.text(0.03, 0.925, subtitle, fontsize=10, color=MUTED)
    colorbar(fig, cs, [0.08, 0.07, 0.84, 0.025], cbar_label, ticks)
    savefig(fig, fname, 115)


def plot_loop(run):
    frames, init = [], run["init"]
    for n, end in enumerate(window_hours(), start=1):
        start = end - WIN
        v0, v1 = init + timedelta(hours=start), init + timedelta(hours=end)
        fname = f"loop_06h_{n:02d}.png"
        _frame(run["data"][start:end].sum(axis=0), "qpf",
               f"REFS LPMM {WIN}-h QPF  ·  f{start:02d}–f{end:02d}",
               f"Valid {v0:%a %HZ} – {v1:%a %HZ %d %b}  ·  {init:%HZ %d %b} init",
               fname, "Precipitation (in)", CLEVS, '"', "{:.2f}", 0.1)
        frames.append({"file": fname, "start": v0.isoformat(), "end": v1.isoformat(),
                       "fhr": f"f{start:02d}-f{end:02d}"})
    return frames


def plot_eas_loop(init, eas):
    out = {}
    for thr in EAS_THRESHOLDS:
        frames = []
        for n, end in enumerate(window_hours(), start=1):
            field = eas.get(end, {}).get(thr)
            if field is None:
                continue
            start = end - WIN
            v0, v1 = init + timedelta(hours=start), init + timedelta(hours=end)
            fname = f"eas_{str(thr).replace('.', 'p')}_{n:02d}.png"
            _frame(field, "prob", f"REFS EAS  P(6-h QPF ≥ {thr:g}\")  ·  f{start:02d}–f{end:02d}",
                   f"Valid {v0:%a %HZ} – {v1:%a %HZ %d %b}  ·  {init:%HZ %d %b} init",
                   fname, "Probability (%)", PLEVS, "%", "{:.0f}", 5)
            frames.append({"file": fname, "start": v0.isoformat(), "end": v1.isoformat(),
                           "fhr": f"f{start:02d}-f{end:02d}"})
        if frames:
            out[f"{thr:g}"] = frames
    return out


def plot_lpmm_vs_avrg(lpmm_tot, avrg_tot, init, valid_txt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    add_basemap(axes[0]); add_basemap(axes[1], labels=False); add_basemap(axes[2], labels=False)
    cs = shade(axes[0], lpmm_tot); annotate_max(axes[0], lpmm_tot)
    shade(axes[1], avrg_tot); annotate_max(axes[1], avrg_tot)
    ds = shade(axes[2], lpmm_tot - avrg_tot, "diff")
    axes[0].set_title("LPMM", fontsize=12, fontweight="bold", color=RUN_COLORS[0])
    axes[1].set_title("AVRG (PMM / mean blend)", fontsize=12, fontweight="bold", color=AVRG_COLOR)
    axes[2].set_title("LPMM − AVRG", fontsize=12, fontweight="bold")
    colorbar(fig, cs, [0.04, 0.1, 0.6, 0.03], "Precipitation (in)", CLEVS)
    colorbar(fig, ds, [0.69, 0.1, 0.27, 0.03], "Difference (in), green = LPMM wetter", DLEVS)
    fig.suptitle(f"REFS 48-h QPF, {fmt_init(init)} init: LPMM vs AVRG\n{valid_txt}",
                 fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "lpmm_vs_avrg.png", 170)
    return True


def plot_ffri(ffri_max, init, valid_txt):
    panels = [("ffg", "6-h QPF > flash flood guidance")] + \
             [(y, f"6-h QPF > {y}-yr recurrence") for y in FFRI_ARI_YEARS]
    panels = [(k, t) for k, t in panels if ffri_max.get(k) is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 6.6),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ((key, title), ax) in enumerate(zip(panels, axes[0])):
        add_basemap(ax, labels=(k == 0))
        cs = shade(ax, ffri_max[key], "prob")
        annotate_max(ax, ffri_max[key], unit="%", fmt="{:.0f}", floor=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Highest neighborhood probability in any 6-h window (%)", PLEVS)
    fig.suptitle(f"REFS flash flood and recurrence interval, {fmt_init(init)} init\n{valid_txt}",
                 fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "ffri_48h.png", 150)
    return True


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.split(" (")[0]).strip("_").lower()


def plot_pad_timeline(runs, avrg, eas, valid_start, site=PRIMARY_SITE):
    (j, i), _ = site_index(site)
    times = [valid_start + timedelta(hours=h) for h in range(1, ACCUM_HOURS + 1)]
    has_eas = bool(eas)
    fig, axs = plt.subplots(3 if has_eas else 2, 1, figsize=(12, 9.5 if has_eas else 7), sharex=True,
                            gridspec_kw={"height_ratios": [1.1, 1, 0.9] if has_eas else [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.08, hspace=0.12)
    a1, a2 = axs[0], axs[1]
    for k in reversed(range(len(runs))):
        s = runs[k]["data"][:, j, i]
        lab = f"{RUN_LABELS[k]} LPMM ({runs[k]['init']:%d/%HZ})"
        if k == 0:
            a1.bar([t - timedelta(minutes=30) for t in times], s, width=timedelta(minutes=50),
                   color=RUN_COLORS[0], label=lab, zorder=3)
        else:
            a1.step(times, s, where="pre", color=RUN_COLORS[k], lw=1.4, label=lab, zorder=4)
        a2.plot(times, np.cumsum(s), color=RUN_COLORS[k], lw=2.2 if k == 0 else 1.5, label=lab)
    if avrg is not None:
        a1.step(times, avrg[:, j, i], where="pre", color=AVRG_COLOR, lw=1.2, ls="--", label="Latest AVRG", zorder=5)
        a2.plot(times, np.cumsum(avrg[:, j, i]), color=AVRG_COLOR, lw=1.5, ls="--", label="Latest AVRG")
    if has_eas:
        a3 = axs[2]
        edges = [valid_start + timedelta(hours=h - WIN) for h in window_hours()] + [valid_start + timedelta(hours=ACCUM_HOURS)]
        for thr in EAS_THRESHOLDS:
            vals = [eas.get(h, {}).get(thr) for h in window_hours()]
            if all(v is None for v in vals):
                continue
            y = [np.nan if v is None else v[j, i] for v in vals]
            a3.step(edges, y + [y[-1]], where="post", color=EAS_COLORS[thr], lw=1.8, label=f"≥ {thr:g}\"")
        a3.set_ylabel("EAS 6-h P (%)")
        a3.set_ylim(0, 100)
        a3.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)
    for a in axs:
        a.grid(color="#263848", lw=0.6)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color(LINE)
        d = valid_start.replace(hour=0) - timedelta(days=1)
        while d < times[-1]:
            a.axvspan(d - timedelta(hours=1), d + timedelta(hours=11), color="#070d12", alpha=0.85, zorder=0)
            d += timedelta(days=1)
        a.set_xlim(valid_start, times[-1])
    a1.set_ylabel("1-h QPF (in)")
    a2.set_ylabel("Accumulated (in)")
    a1.set_ylim(0, max(0.1, a1.get_ylim()[1]))
    a2.set_ylim(0, max(0.1, a2.get_ylim()[1]))
    h, l = a1.get_legend_handles_labels()
    a1.legend(h[::-1], l[::-1], loc="upper left", frameon=False, fontsize=9, ncol=4)
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
    lat, lon, _ = SITES[site]
    fig.suptitle(f"REFS at {site} ({lat:.3f}, {lon:.3f}), nearest grid point\nShaded = local night",
                 fontsize=13, fontweight="bold", y=0.975)
    fname = f"timeline_{slug(site)}.png"
    savefig(fig, fname, 130)
    return {"site": site, "file": fname}

# =============================================================================
# 10. VERIFICATION
# =============================================================================
def to_i16(x):
    return np.clip(np.round(np.nan_to_num(x) * 100), 0, 32767).astype(np.int16)


def from_i16(x):
    return x.astype(np.float32) / 100.0


def archive_forecast(init, lpmm_tot, avrg_tot, regime):
    os.makedirs(FCST_DIR, exist_ok=True)
    fn = os.path.join(FCST_DIR, f"refs_{init:%Y%m%d%H}.npz")
    if not os.path.exists(fn):
        np.savez_compressed(fn, lpmm=to_i16(lpmm_tot),
                            avrg=to_i16(avrg_tot) if avrg_tot is not None else np.zeros(0, np.int16),
                            regime=json.dumps(regime))
        print(f"[archive] saved {fn}")


def score(init, lpmm, avrg, obs):
    mask = in_view() & np.isfinite(obs)
    o = obs[mask]
    (j, i), _ = site_index(PRIMARY_SITE)
    rec = {"init": init.isoformat(), "valid_end": (init + timedelta(hours=ACCUM_HOURS)).isoformat(),
           "n": int(mask.sum()), "obs_mean": r2(o.mean()), "obs_max": r2(o.max()),
           "pad": {"obs": r2(obs[j, i]), "lpmm": r2(lpmm[j, i]), "avrg": r2(avrg[j, i]) if avrg is not None else None}}
    for name, fc in (("lpmm", lpmm), ("avrg", avrg)):
        if fc is None:
            continue
        f = fc[mask]
        e = {f"{t:g}": categorical(f, o, t) for t in VERIFY_THRESHOLDS}
        rec[name] = {"mean": r2(f.mean()),
                     "bias_ratio": r2(f.sum() / o.sum()) if o.sum() > 0.01 * o.size else None,
                     "ets": {k: v[0] for k, v in e.items()},
                     "fbias": {k: v[1] for k, v in e.items()}}
    return rec


def preferred_regime(s):
    for key in ("regime_raob", "regime_anl", "regime_fcst"):
        if s.get(key):
            return s[key]["name"], key.split("_")[1]
    return None, None


def verify_pending(regime_log):
    scores = load_json(SCORES_FILE, [])
    done = {s["init"] for s in scores}
    now = utcnow()
    newest = None
    for fn in sorted(glob.glob(os.path.join(FCST_DIR, "refs_*.npz"))):
        init = datetime.strptime(os.path.basename(fn)[5:15], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        vend = init + timedelta(hours=ACCUM_HOURS)
        if init.isoformat() in done:
            os.remove(fn); continue
        if vend > now - MRMS_LATENCY:
            continue
        z = np.load(fn)
        if now - vend > FCST_MAX_AGE or z["lpmm"].shape != _GRID["lats"].shape:
            print(f"[verify] dropping {fn} (too old or grid changed)")
            os.remove(fn); continue
        raw = fetch_mrms_raw(ACCUM_HOURS, vend)
        if raw is None:
            continue
        obs = sample_mrms(raw, _GRID["lats"], _GRID["lons"])
        lp = from_i16(z["lpmm"])
        av = from_i16(z["avrg"]) if z["avrg"].size else None
        rec = score(init, lp, av, obs)
        rec["regime_fcst"] = json.loads(str(z["regime"])) if "regime" in z.files else None
        rec["regime_anl"] = analysis_regime(regime_log, init, vend)
        rec["regime_raob"] = raob_regime(init, vend)
        scores.append(rec)
        newest = (init, lp, av, obs)
        os.remove(fn)
        print(f"[verify] scored {init:%Y%m%d %HZ}  regime fcst={(rec['regime_fcst'] or {}).get('name')} "
              f"anl={(rec['regime_anl'] or {}).get('name')} raob={(rec['regime_raob'] or {}).get('name')}")
    # backfill observed regimes for runs scored before tagging existed
    n = 0
    for s in scores:
        if n >= RAOB_BACKFILL_PER_RUN:
            break
        if "regime_raob" not in s or ("regime_anl" not in s):
            st = datetime.fromisoformat(s["init"]); en = datetime.fromisoformat(s["valid_end"])
            s.setdefault("regime_anl", analysis_regime(regime_log, st, en))
            s["regime_raob"] = raob_regime(st, en)
            s.setdefault("regime_fcst", None)
            n += 1
    scores.sort(key=lambda s: s["init"])
    save_json(SCORES_FILE, scores)
    if newest:
        init, lp, av, obs = newest
        np.savez_compressed(VERIFY_LATEST, init=init.isoformat(), lpmm=to_i16(lp),
                            avrg=to_i16(av) if av is not None else np.zeros(0, np.int16),
                            obs=np.where(np.isfinite(obs), to_i16(obs), -1).astype(np.int16))
    return scores


def plot_verify_map():
    if not os.path.exists(VERIFY_LATEST):
        return False
    z = np.load(VERIFY_LATEST)
    if z["lpmm"].shape != _GRID["lats"].shape:
        return False
    init = datetime.fromisoformat(str(z["init"]))
    obs = np.where(z["obs"] >= 0, from_i16(z["obs"]), np.nan)
    fields = [("MRMS 48-h QPE (Pass 2)", obs, TEXT), ("LPMM", from_i16(z["lpmm"]), RUN_COLORS[0])]
    if z["avrg"].size:
        fields.append(("AVRG", from_i16(z["avrg"]), AVRG_COLOR))
    fig, axes = plt.subplots(1, len(fields), figsize=(5.4 * len(fields), 6.6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.84, wspace=0.04)
    cs = None
    for k, ((title, fld, col), ax) in enumerate(zip(fields, axes)):
        add_basemap(ax, labels=(k == 0))
        cs = shade(ax, fld)
        annotate_max(ax, fld)
        ax.set_title(title, fontsize=12, fontweight="bold", color=col)
    colorbar(fig, cs, [0.18, 0.1, 0.64, 0.03], "Precipitation (in)", CLEVS)
    vend = init + timedelta(hours=ACCUM_HOURS)
    fig.suptitle(f"Most recent verified run: {fmt_init(init)} init\n"
                 f"Valid {init:%HZ %a %d %b} to {vend:%HZ %a %d %b}", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "verify_map.png", 150)
    return True


def plot_verify_scores(scores):
    if not scores:
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.22)
    for key, col, lab in (("lpmm", RUN_COLORS[0], "LPMM"), ("avrg", AVRG_COLOR, "AVRG")):
        pts = [(s["pad"]["obs"], s["pad"][key]) for s in scores
               if s["pad"].get(key) is not None and s["pad"]["obs"] is not None]
        if pts:
            o, f = zip(*pts)
            a1.scatter(o, f, s=22, color=col, label=lab, alpha=0.85, edgecolor="none")
        series = [(datetime.fromisoformat(s["init"]), s.get(key, {}).get("ets", {}).get("1"))
                  for s in scores[-120:] if key in s]
        series = [(t, v) for t, v in series if v is not None]
        if series:
            t, v = zip(*series)
            a2.plot(t, v, "-o", ms=3, lw=1.2, color=col, label=f"{lab} (mean {np.mean(v):.2f})")
    lim = max(0.5, a1.get_xlim()[1], a1.get_ylim()[1])
    a1.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    a1.set_xlim(0, lim); a1.set_ylim(0, lim)
    a1.set_xlabel("MRMS 48-h (in)"); a1.set_ylabel("REFS 48-h (in)")
    a1.set_title(f"{PRIMARY_SITE}, {len(scores)} runs", fontsize=11, fontweight="bold")
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_ylabel("ETS, 48-h ≥ 1\"")
    a2.set_title("Domain equitable threat score by run (blank = no 1\" observed or forecast)",
                 fontsize=11, fontweight="bold")
    loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
    a2.xaxis.set_major_locator(loc)
    a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    t_all = [datetime.fromisoformat(s["init"]) for s in scores[-120:]]
    a2.set_xlim(min(t_all) - timedelta(days=1), max(t_all) + timedelta(days=1))
    for a in (a1, a2):
        a.grid(color="#263848", lw=0.6)
        a.legend(frameon=False, fontsize=9, loc="upper left")
        for sp in a.spines.values():
            sp.set_color(LINE)
    fig.suptitle("REFS vs MRMS 48-h QPE", fontsize=14, fontweight="bold", y=0.97)
    savefig(fig, "verify.png", 140)
    return True


def by_regime(scores):
    groups = {}
    for s in scores:
        name, src = preferred_regime(s)
        if not name:
            continue
        g = groups.setdefault(name, {"n": 0, "src": {}, "ets_l": [], "ets_a": [], "bias_l": [], "bias_a": []})
        g["n"] += 1
        g["src"][src] = g["src"].get(src, 0) + 1
        for k, key, sub in (("ets_l", "lpmm", "ets"), ("ets_a", "avrg", "ets")):
            v = s.get(key, {}).get(sub, {}).get("1")
            if v is not None:
                g[k].append(v)
        for k, key in (("bias_l", "lpmm"), ("bias_a", "avrg")):
            v = s.get(key, {}).get("bias_ratio")
            if v is not None:
                g[k].append(v)
    out = []
    order = ["SW", "SE", "NE", "NW", "Light", "Mixed"]
    for name in sorted(groups, key=lambda x: order.index(x) if x in order else 99):
        g = groups[name]
        mean = lambda a: r2(np.mean(a)) if a else None
        out.append({"regime": name, "n": g["n"], "sources": g["src"],
                    "ets1_lpmm": mean(g["ets_l"]), "ets1_avrg": mean(g["ets_a"]), "n_ets": len(g["ets_l"]),
                    "bias_lpmm": mean(g["bias_l"]), "bias_avrg": mean(g["bias_a"])})
    return out

# =============================================================================
# 11. META
# =============================================================================
def build_meta(runs, avrg, eas, ffri_max, valid_start, valid_end, frames, eas_frames, products, scores, regime):
    sites = []
    for name, (lat, lon, mk) in SITES.items():
        (j, i), nb = site_index(name)
        per_run = []
        for k, run in enumerate(runs):
            s = run["data"][:, j, i]
            h = int(np.argmax(s))
            per_run.append({"run": RUN_LABELS[k], "total": r2(s.sum()),
                            "nbhd_max": r2(run["data"].sum(axis=0)[nb].max()) if nb.any() else None,
                            "max_1h": r2(s[h]),
                            "max_1h_valid": (valid_start + timedelta(hours=h + 1)).isoformat()})
        eas_max = {}
        for thr in EAS_THRESHOLDS:
            v = [eas[h][thr][j, i] for h in window_hours() if thr in eas.get(h, {})]
            if v:
                eas_max[f"{thr:g}"] = int(round(max(v)))
        sites.append({
            "name": name, "label": SITE_LABELS.get(name, name), "lat": lat, "lon": lon, "pad": mk == "*", "runs": per_run,
            "avrg_total": r2(avrg[:, j, i].sum()) if avrg is not None else None,
            "eas_max": eas_max,
            "ffg_max": int(round(ffri_max["ffg"][j, i])) if ffri_max.get("ffg") is not None else None,
        })

    tot = runs[0]["data"].sum(axis=0)
    masked = np.where(in_view(), tot, -1)
    j, i = np.unravel_index(np.argmax(masked), masked.shape)

    verify_summary = None
    if scores:
        def mean_of(key, path):
            v = [s.get(key, {}).get(path[0], {}).get(path[1]) for s in scores[-30:]]
            v = [x for x in v if x is not None]
            return (r2(np.mean(v)), len(v)) if v else (None, 0)
        verify_summary = {
            "n": len(scores),
            "ets1_lpmm": mean_of("lpmm", ("ets", "1")), "ets1_avrg": mean_of("avrg", ("ets", "1")),
            "fbias1_lpmm": mean_of("lpmm", ("fbias", "1")), "fbias1_avrg": mean_of("avrg", ("fbias", "1")),
            "by_regime": by_regime(scores),
            "recent": [dict(s, regime=preferred_regime(s)[0]) for s in scores[-10:][::-1]],
        }

    return {
        "generated_utc": utcnow().isoformat(timespec="seconds"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "accum_hours": ACCUM_HOURS,
        "valid_start": valid_start.isoformat(), "valid_end": valid_end.isoformat(),
        "runs": [{"label": RUN_LABELS[k], "init": r["init"].isoformat(), "color": RUN_COLORS[k]}
                 for k, r in enumerate(runs)],
        "primary_site": PRIMARY_SITE, "nbhd_km": NBHD_KM,
        "thresholds": THRESHOLDS, "eas_thresholds": EAS_THRESHOLDS,
        "regime": regime, "regime_min_n": REGIME_MIN_N,
        "regime_rules": {"light_kt": LIGHT_KT, "mixed_consistency": MIXED_CONSISTENCY,
                         "layer": "10 m–700 mb (pressure-weighted)",
                         "point": f"{REGIME_POINT_NAME}, {REGIME_RADIUS_KM:g} km mean", "point_name": REGIME_POINT_NAME},
        "branding": BRANDING,
        "domain_max": {"value": r2(tot[j, i]), "lat": round(float(_GRID["lats"][j, i]), 3),
                       "lon": round(float(_GRID["lons"][j, i]), 3)},
        "sites": sites, "frames": frames, "eas_frames": eas_frames, "products": products,
        "verify": verify_summary,
        "colorscale": {"levels": CLEVS, "colors": [mcolors.to_hex(c) for c in CMAP_COLORS]},
    }

# =============================================================================
# 12. MAIN
# =============================================================================
def optional(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        print(f"[warn] {label} failed:\n{traceback.format_exc()}")
        return None


def migrate_legacy_archive():
    if os.path.isdir(LEGACY_ARCHIVE) and not os.path.exists(SCORES_FILE):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        shutil.copytree(LEGACY_ARCHIVE, ARCHIVE_DIR, dirs_exist_ok=True)
        print(f"[migrate] copied ./{LEGACY_ARCHIVE} into {ARCHIVE_DIR}; you can delete it from main")


def publish_cases(site_dir):
    """Copy look-back cases into the site and write cases/index.json."""
    dst = os.path.join(site_dir, "cases")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    index = []
    for cj in sorted(glob.glob(os.path.join(CASES_DIR, "*", "case.json"))):
        cdir = os.path.dirname(cj)
        meta = load_json(cj, None)
        if not meta:
            continue
        shutil.copytree(cdir, os.path.join(dst, os.path.basename(cdir)))
        index.append({k: meta.get(k) for k in ("id", "title", "start", "end", "hours", "auto", "obs", "headline", "built")})
    index.sort(key=lambda c: c["end"], reverse=True)
    save_json(os.path.join(dst, "index.json"), index, indent=1)
    return len(index)


def main():
    assert (N_RUNS - 1) * RUN_SPACING + ACCUM_HOURS <= MAX_FHR, "time-lag window exceeds f60"
    os.makedirs(OUT_DIR, exist_ok=True)
    migrate_legacy_archive()
    init_sites()

    latest = find_latest_cycle()
    if latest is None:
        print("No complete REFS cycle found; leaving the published site unchanged.")
        sys.exit(1)
    print(f"Latest complete cycle: {latest:%Y%m%d %HZ}")

    runs = []
    for k in range(N_RUNS):
        lag = k * RUN_SPACING
        init = latest - timedelta(hours=lag)
        print(f"LPMM {RUN_LABELS[k]}: {init:%Y%m%d %HZ} f{lag + 1:02d}-f{lag + ACCUM_HOURS:02d}")
        data = fetch_qpf_run(init, "lpmm", lag)
        if data is None:
            if k == 0:
                print("Latest run incomplete; aborting.")
                sys.exit(1)
            continue
        runs.append({"init": init, "data": data})

    print("AVRG latest")
    avrg = optional("avrg", fetch_qpf_run, latest, "avrg", 0)
    print("EAS latest")
    eas = optional("eas", fetch_many, latest, "eas", window_hours(), read_eas_6h) or {}
    print("FFRI latest")
    ffri = optional("ffri", fetch_many, latest, "ffri", window_hours(), read_ffri_6h) or {}
    print("Flow regime")
    regime = optional("forecast regime", forecast_regime, latest)
    regime_log = optional("analysis wind", log_analysis_wind, latest) or load_json(REGIME_LOG, [])

    ffri_max = {}
    if ffri:
        ffg = [v["ffg"] for v in ffri.values() if v["ffg"] is not None]
        ffri_max["ffg"] = np.max(np.stack(ffg), axis=0) if ffg else None
        for y in FFRI_ARI_YEARS:
            a = [v["ari"][y] for v in ffri.values() if y in v["ari"]]
            ffri_max[y] = np.max(np.stack(a), axis=0) if a else None

    valid_start, valid_end = latest, latest + timedelta(hours=ACCUM_HOURS)
    valid_txt = f"Valid {valid_start:%HZ %a %d %b} to {valid_end:%HZ %a %d %b %Y}"
    lpmm_tot = runs[0]["data"].sum(axis=0)
    avrg_tot = avrg.sum(axis=0) if avrg is not None else None

    archive_forecast(latest, lpmm_tot, avrg_tot, regime)
    scores = optional("verification", verify_pending, regime_log) or load_json(SCORES_FILE, [])

    plot_trend(runs, valid_txt)
    plot_thresholds(runs, valid_txt)
    frames = plot_loop(runs[0])
    eas_frames = optional("eas loop", plot_eas_loop, latest, eas) or {}
    products = {
        "eas": bool(eas_frames),
        "avrg": avrg_tot is not None and bool(optional("lpmm vs avrg", plot_lpmm_vs_avrg,
                                                        lpmm_tot, avrg_tot, latest, valid_txt)),
        "ffri": any(v is not None for v in ffri_max.values())
                and bool(optional("ffri", plot_ffri, ffri_max, latest, valid_txt)),
        "verify_map": bool(optional("verify map", plot_verify_map)),
        "verify": bool(optional("verify scores", plot_verify_scores, scores)),
    }
    timelines = [t for t in (optional(f"timeline {n}", plot_pad_timeline, runs, avrg, eas, valid_start, n)
                             for n in TIMELINE_SITES if n in SITES) if t]

    if AUTO_DAILY_CASE and latest.hour == 12:
        import lookback
        optional("auto look-back", lookback.build_case, end=latest, hours=24,
                 title=f"24 h ending 12Z {latest:%a %d %b}", auto=True)
        optional("prune auto cases", lookback.prune_auto, AUTO_CASES_KEPT)
    products["cases"] = (optional("publish cases", publish_cases, OUT_DIR) or 0) > 0

    meta = build_meta(runs, avrg, eas, ffri_max, valid_start, valid_end, frames, eas_frames,
                      products, scores, regime)
    meta["timelines"] = timelines
    meta["diagnostics"] = DIAG
    save_json(os.path.join(OUT_DIR, "meta.json"), meta, indent=1)
    if os.path.exists(INDEX_SRC):
        shutil.copy(INDEX_SRC, os.path.join(OUT_DIR, "index.html"))
    open(os.path.join(OUT_DIR, ".nojekyll"), "w").close()

    if os.path.isdir(SITE_CACHE):
        shutil.rmtree(SITE_CACHE)
    shutil.copytree(OUT_DIR, SITE_CACHE, ignore=shutil.ignore_patterns("cases"))

    p = next(s for s in meta["sites"] if s["name"] == PRIMARY_SITE)
    print(f"Done. {PRIMARY_SITE} 48-h LPMM {p['runs'][0]['total']}\", AVRG {p['avrg_total']}\"; "
          f"regime {(regime or {}).get('name')}; products {products}; verified runs {len(scores)}")


if __name__ == "__main__":
    main()
