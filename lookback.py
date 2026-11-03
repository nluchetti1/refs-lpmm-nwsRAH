"""
Event look-back: how the models forecast a past rainfall window, verified
against MRMS MultiSensor Pass2 QPE.

  python lookback.py --start 2026110112 --end 2026110312 --title "Brevard flooding"

Any whole-hour window from 3 to 72 h. Each model is included only where its
output steps line up with the window (ECMWF IFS/ENS 3-hourly, AIFS 6-hourly).

Model                 Source (all anonymous HTTPS)            Precip handling
REFS LPMM / AVRG      NOMADS refs/prod, AWS noaa-rrfs-ops-pds  sum of 1-h records
HRRR                  AWS noaa-hrrr-bdp-pds                    run-total difference
RRFS                  AWS noaa-rrfs-ops-pds                    run-total difference
GFS                   AWS noaa-gfs-bdp-pds (0.25°)            run-total difference
ECMWF IFS / AIFS      data.ecmwf.int, then AWS ecmwf-forecasts tp is run-accumulated
GEFS  (31 members)    NOMADS / AWS noaa-gefs-pds (0.25°)      run totals or buckets, read per file
ECMWF ENS (51 mem)    data.ecmwf.int, then AWS ecmwf-forecasts tp is run-accumulated
Ensembles are reported as the member mean, 75th percentile and maximum at each
grid point. The max is an upper envelope, not a forecast any member made.

Output: data/cases/<id>/{case.json, refs.png, models.png, ens.png, dprog.png}
"""
import argparse
import glob
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygrib

import refs_qpf as R

# ---------------------------------------------------------------- sources
HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d:%Y%m%d}/conus/hrrr.t{d:%H}z.wrfsfcf{f:02d}.grib2"
RRFS_URL = "https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{d:%Y%m%d}/{d:%H}/rrfs.t{d:%H}z.2dfld.3km.f{f:03d}.conus.grib2"
GFS_URLS = ["https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p25.f{f:03d}",
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p25.f{f:03d}"]
GEFS_URLS = ["https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{d:%Y%m%d}/{d:%H}/atmos/pgrb2sp25/{m}.t{d:%H}z.pgrb2s.0p25.f{f:03d}",
             "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d:%Y%m%d}/{d:%H}/atmos/pgrb2sp25/{m}.t{d:%H}z.pgrb2s.0p25.f{f:03d}"]
# ECMWF open data mirrors. data.ecmwf.int keeps ~4 days; AWS throttles with 503 SlowDown under
# load; Azure (Planetary Computer) and Google Cloud carry the same files and layout.
ECMWF_RECENT = "https://data.ecmwf.int/forecasts"
ECMWF_MIRRORS = ["https://ai4edataeuwest.blob.core.windows.net/ecmwf",
                 "https://storage.googleapis.com/ecmwf-open-data",
                 "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]
ECMWF_COALESCE_GAP = 2 * 1024 * 1024     # fetch neighbouring records in one request if gap < 2 MB
ECMWF_PACE_S = 0.15                       # pause between range requests to the same host
ECMWF_PATH = "{d:%Y%m%d}/{d:%H}z/{model}/0p25/{stream}/{d:%Y%m%d%H}0000-{f}h-{stream}-{type}.grib2"
REFS_AWS_START = datetime(2026, 8, 13, tzinfo=timezone.utc)
GEFS_MEMBERS = ["gec00"] + [f"gep{k:02d}" for k in range(1, 31)]

MIN_HOURS, MAX_HOURS = 3, 72
EVENT_RADIUS_KM = 20.0
CASE_THRESHOLDS = [1.0, 2.0, 4.0, 8.0]
EVENT_COVERAGE = 0.05            # event threshold = highest one MRMS exceeds over >= 5% of the domain
SUMMARY_VERSION = 2
WORKERS = 8

# key: display, color, cycle spacing (h), max fhr, output step (h) as fn of fhr, leads (h before start)
MODELS = {
    "REFS": dict(label="REFS LPMM", color="#4fd1e8", every=6, fmax=60, step=lambda f: 1, leads=range(0, 37, 6)),
    "AVRG": dict(label="REFS AVRG", color="#b59cf5", every=6, fmax=60, step=lambda f: 1, leads=[0]),
    "HRRR": dict(label="HRRR", color="#f0b429", every=6, fmax=48, step=lambda f: 1, leads=range(0, 37, 6)),
    "RRFS": dict(label="RRFS", color="#5fd38d", every=6, fmax=84, step=lambda f: 1, leads=range(0, 49, 6)),
    "GFS":  dict(label="GFS", color="#ef8a62", every=6, fmax=384, step=lambda f: 1 if f <= 120 else 3, leads=range(0, 121, 12)),
    "IFS":  dict(label="ECMWF IFS", color="#6fa8ff", every=12, fmax=240, step=lambda f: 3 if f <= 144 else 6, leads=range(0, 121, 12)),
    "AIFS": dict(label="ECMWF AIFS", color="#f78fb3", every=6, fmax=360, step=lambda f: 6, leads=range(0, 121, 12)),
    "GEFS": dict(label="GEFS", color="#e0607e", every=6, fmax=240, step=lambda f: 3, leads=[0, 24, 48, 72]),
    "ENS":  dict(label="ECMWF ENS", color="#9fe0c8", every=12, fmax=360, step=lambda f: 3 if f <= 144 else 6, leads=[0, 24, 48, 72]),
}
ENSEMBLES = ("GEFS", "ENS")
STATS = ("mean", "p75", "max")


def hours_between(a, b):
    return int(round((b - a).total_seconds() / 3600))


def plan_inits(key, start, end):
    """[(lead_h, init)] for the model, newest first, only where the run covers the window
    and its output steps fall exactly on the window edges."""
    m = MODELS[key]
    base = start.replace(hour=(start.hour // m["every"]) * m["every"], minute=0, second=0, microsecond=0)
    out, seen = [], set()
    for want in m["leads"]:
        init = base - timedelta(hours=int(np.ceil(want / m["every"])) * m["every"])
        if init in seen:
            continue
        seen.add(init)
        f0, f1 = hours_between(init, start), hours_between(init, end)
        if f1 > m["fmax"]:
            continue
        aligned = all(f == 0 or f % m["step"](f) == 0 for f in (f0, f1))
        if aligned:
            out.append((f0, init))
    return out

def skip_reason(key, start, end):
    """Plain-language reason plan_inits found no usable run."""
    m = MODELS[key]
    hrs = hours_between(start, end)
    step = m["step"](hrs)
    if step > 1 and (start.hour % step or end.hour % step):
        return f"output every {step} h, so start and end must be on {step}-h boundaries such as 00, {step:02d}, 12Z"
    if hrs > m["fmax"]:
        return f"runs end at f{m['fmax']}, shorter than a {hrs}-h window"
    return f"no run reaches f{m['fmax']} with the window inside it; try 00/06/12/18Z edges"


# ---------------------------------------------------------------- readers
def refs_window(init, ptype, f0, f1):
    if init < REFS_AWS_START and R.utcnow() - init > timedelta(hours=40):
        return None
    data = R.fetch_qpf_hours(init, ptype, list(range(f0 + 1, f1 + 1)))
    return None if data is None else (data.sum(axis=0), R._GRID["lats"], R._GRID["lons"])


def _decode_apcp_total(path, fhr):
    """0-fhr run-total APCP (mm) and its crop from a subset file."""
    with pygrib.open(path) as g:
        msgs = [m for m in g if R._is_apcp(m)]
    pick = next((m for m in msgs if R._step_range(m) == (0, fhr)), None)
    if pick is None:
        raise ValueError(f"no 0-{fhr} h APCP: {[R._step_range(m) for m in msgs]}")
    crop = R.make_crop(pick)
    return R.crop_values(pick, crop), crop


def run_total(urls, fhr, tag):
    """0-fhr APCP total via byte-range on the APCP records only."""
    path = R.download_subset(urls, lambda l: ":APCP:surface:" in l)
    try:
        with pygrib.open(path) as g:
            R._log_inventory(f"apcp-{tag} f{fhr}", [m for m in g if R._is_apcp(m)])
        return _decode_apcp_total(path, fhr)
    finally:
        os.remove(path)


def det_window(key, init, f0, f1):
    """Deterministic NOAA models: total(f1) - total(f0)."""
    def urls(f):
        if key == "HRRR":
            return [HRRR_URL.format(d=init, f=f)]
        if key == "RRFS":
            return [RRFS_URL.format(d=init, f=f)]
        u = [t.format(d=init, f=f) for t in GFS_URLS]
        return u if R.utcnow() - init < timedelta(days=8) else u[::-1]
    b, crop = run_total(urls(f1), f1, key)
    a = run_total(urls(f0), f0, key)[0] if f0 > 0 else 0.0
    return np.clip((b - a) * R.MM_TO_IN, 0, None), crop["lats"], crop["lons"]


def ecmwf_urls(model, stream, typ, init, f):
    roots = ([ECMWF_RECENT] if R.utcnow() - init < timedelta(days=3, hours=12) else []) + ECMWF_MIRRORS
    return [f"{root}/{ECMWF_PATH.format(d=init, model=model, stream=stream, type=typ, f=f)}" for root in roots]


def _ecmwf_session():
    """Separate session with patient retries: S3 SlowDown and Azure throttling clear within a minute or two."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = R.requests.Session()
    kw = dict(total=6, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=("GET",), respect_retry_after_header=True)
    try:
        retry = Retry(backoff_max=90, **kw)          # urllib3 >= 2
    except TypeError:
        retry = Retry(**kw)                          # urllib3 1.26
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = R.SESSION.headers.get("User-Agent", "refs-qpf")
    return s


ECMWF_SESSION = None
_ECMWF_DIAG = set()


def _coalesce(recs):
    """[(start, end, [records])] spans covering the records, merging ones within ECMWF_COALESCE_GAP."""
    spans = []
    for x in sorted(recs, key=lambda x: int(x["_offset"])):
        a, b = int(x["_offset"]), int(x["_offset"]) + int(x["_length"]) - 1
        if spans and a - spans[-1][1] <= ECMWF_COALESCE_GAP:
            spans[-1][1] = max(spans[-1][1], b); spans[-1][2].append(x)
        else:
            spans.append([a, b, [x]])
    return spans


def ecmwf_tp(model, stream, typ, init, f):
    """{member: (tp_mm, crop)} for every tp record in one ECMWF open-data file (byte-range via .index)."""
    global ECMWF_SESSION
    if f == 0:
        return None
    if ECMWF_SESSION is None:
        ECMWF_SESSION = _ecmwf_session()
    sess = ECMWF_SESSION if R.SESSION.get.__name__ == "get" else R.SESSION   # tests patch R.SESSION
    errs = []
    for url in ecmwf_urls(model, stream, typ, init, f):
        host = url.split("/")[2]
        try:
            r = sess.get(url[:-len(".grib2")] + ".index", timeout=30)
            if r.status_code != 200:
                errs.append(f"{r.status_code} {host}"); continue
            recs = [json.loads(l) for l in r.text.splitlines() if l.strip()]
            recs = [x for x in recs if x.get("param") == "tp"]
            if not recs:
                errs.append(f"no tp in index on {host}"); continue
            spans = _coalesce(recs)
            path = R._tmp()
            with open(path, "wb") as fh:
                for a, b, members in spans:
                    rr = sess.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    if rr.status_code not in (200, 206):
                        raise IOError(f"{rr.status_code} range on {host}")
                    blob = rr.content
                    for x in members:
                        o = int(x["_offset"]) - a
                        fh.write(blob[o:o + int(x["_length"])])
                    time.sleep(ECMWF_PACE_S)
            print(f"    ECMWF {stream} {init:%d/%HZ} f{f}: {len(recs)} tp records in {len(spans)} requests from {host}")
            out = {}
            try:
                with pygrib.open(path) as g:
                    for m in g:
                        units = str(getattr(m, "units", ""))
                        scale = 1000.0 if (units == "m" or int(m["paramId"]) == 228) else 1.0
                        crop = R.make_crop(m)
                        try:
                            mem = int(m["perturbationNumber"])
                        except Exception:
                            mem = 0
                        out[mem] = (R.crop_values(m, crop) * scale, crop)
                        if stream not in _ECMWF_DIAG:
                            _ECMWF_DIAG.add(stream)
                            v = out[mem][0]
                            print(f"    [ecmwf diag] {model}/{stream}: paramId {m['paramId']}, units {units!r}, "
                                  f"step {R._step_range(m)}, packing {m['packingType']}, scale x{scale:g}, "
                                  f"cropped min/max {np.nanmin(v):.2f}/{np.nanmax(v):.2f} mm")
            finally:
                os.remove(path)
            return out
        except Exception as e:
            msg = str(e)
            errs.append(f"{host}: {'throttled (503)' if '503' in msg else e.__class__.__name__}")
    raise FileNotFoundError("all ECMWF mirrors failed: " + "; ".join(errs))


def ecmwf_det(model, init, f0, f1):
    b = ecmwf_tp(model, "oper", "fc", init, f1)
    a = ecmwf_tp(model, "oper", "fc", init, f0) if f0 > 0 else None
    vb, crop = b[sorted(b)[0]]
    va = a[sorted(a)[0]][0] if a else 0.0
    return np.clip((vb - va) * R.MM_TO_IN, 0, None), crop["lats"], crop["lons"]


def ecmwf_ens(init, f0, f1):
    b = ecmwf_tp("ifs", "enfo", "ef", init, f1)
    a = ecmwf_tp("ifs", "enfo", "ef", init, f0) if f0 > 0 else {}
    mems = sorted(k for k in b if (f0 == 0 or k in a))
    crop = b[mems[0]][1]
    stack = np.stack([b[k][0] - (a[k][0] if f0 > 0 else 0.0) for k in mems]) * R.MM_TO_IN
    return np.clip(stack, 0, None), crop["lats"], crop["lons"]


_ACC = re.compile(r":APCP:surface:(\d+)-(\d+) (hour|day) acc")


def _apcp_ranges_from_idx(url):
    r = R.SESSION.get(url + ".idx", timeout=30)
    r.raise_for_status()
    out = []
    for line in r.text.splitlines():
        m = _ACC.search(line)
        if m:
            k = 24 if m.group(3) == "day" else 1
            out.append((int(m.group(1)) * k, int(m.group(2)) * k))
    return out


def gefs_member_window(init, mem, f0, f1, files):
    """Window total (mm) for one member. files: forecast hours to read. Each file's APCP
    record is used by its own step range, so run totals and resetting buckets both work."""
    recs = {}
    crop = None
    for f in files:
        urls = [t.format(d=init, m=mem, f=f) for t in GEFS_URLS]
        urls = urls if R.utcnow() - init < timedelta(days=2) else urls[::-1]
        path = R.download_subset(urls, lambda l: ":APCP:surface:" in l)
        try:
            with pygrib.open(path) as g:
                for m in g:
                    if R._is_apcp(m):
                        crop = R.make_crop(m)
                        recs[R._step_range(m)] = R.crop_values(m, crop)
        finally:
            os.remove(path)
    # cumulative relative to the earliest record start, then difference
    base = min(a for a, _ in recs)
    C = {0: 0.0, base: 0.0}
    for (a, b) in sorted(recs, key=lambda x: (x[1], x[0])):
        if a == 0:
            C[b] = recs[(a, b)] - (C.get(base, 0.0) if base else 0.0)
        elif a in C:
            C[b] = C[a] + recs[(a, b)]
    if f1 not in C or f0 not in C:
        raise ValueError(f"cannot build {f0}-{f1} from records {sorted(recs)}")
    return C[f1] - C[f0], crop


def gefs_ens(init, f0, f1):
    """Plan once from the control's index, then fetch every member in parallel."""
    ctl = [t.format(d=init, m="gec00", f=f1) for t in GEFS_URLS]
    ranges = None
    for u in (ctl if R.utcnow() - init < timedelta(days=2) else ctl[::-1]):
        try:
            ranges = _apcp_ranges_from_idx(u); break
        except Exception:
            continue
    if not ranges:
        raise FileNotFoundError("GEFS control index unavailable")
    a, b = ranges[0]
    if a == 0:                                   # run total: two files
        files = [f for f in (f0, f1) if f > 0]
    else:
        bucket = b - a
        if f0 % bucket == 0 and f1 % bucket == 0:   # window on bucket edges: bucket-end files only
            files = list(range(f0 + bucket, f1 + 1, bucket))
        else:                                       # otherwise chain 3-h files from the bucket start
            base = (f0 // bucket) * bucket
            files = list(range(base + 3, f1 + 1, 3))
    print(f"  GEFS {init:%d/%HZ}: record {a}-{b} h, reading f{files[0]:03d}..f{files[-1]:03d} ({len(files)} files x 31)")
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda m: _safe(gefs_member_window, init, m, f0, f1, files), GEFS_MEMBERS))
    good = [r for r in results if r is not None]
    if len(good) < 10:
        raise ValueError(f"only {len(good)} GEFS members decoded")
    crop = good[0][1]
    stack = np.stack([g[0] for g in good]) * R.MM_TO_IN
    return np.clip(stack, 0, None), crop["lats"], crop["lons"]


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"    ! {a[1] if len(a) > 1 else ''}: {e}")
        return None


def ens_stats(stack):
    return {"mean": stack.mean(axis=0), "p75": np.percentile(stack, 75, axis=0), "max": stack.max(axis=0),
            "n": stack.shape[0]}

# ---------------------------------------------------------------- scoring
def evaluate(field, lats, lons, raw, event_pt):
    obs = R.sample_mrms(raw, lats, lons)
    mask = R.in_view(lats, lons) & np.isfinite(obs)
    f, o = field[mask], obs[mask]
    dx = R.grid_spacing_km(lats, lons)
    near = R.dist_km(lats, lons, *event_pt) <= max(EVENT_RADIUS_KM, 0.75 * dx)
    on = np.where(near & np.isfinite(obs), obs, np.nan)
    row = {"domain_max": R.r2(np.where(R.in_view(lats, lons), field, -1).max()),
           "near_event_max": R.r2(field[near].max()) if near.any() else None,
           "obs_near_max": R.r2(np.nanmax(on)) if np.isfinite(on).any() else None,
           "bias_ratio": R.r2(f.sum() / o.sum()) if o.size and o.sum() > 0.01 * o.size else None,
           "grid_km": round(dx, 1)}
    for t in CASE_THRESHOLDS:
        row[f"ets_{t:g}"], row[f"fbias_{t:g}"] = R.categorical(f, o, t) if f.size else (None, None)
    sites = {}
    for name, (la, lo, _) in R.SITES.items():
        d = R.dist_km(lats, lons, la, lo)
        j, i = np.unravel_index(np.argmin(d), d.shape)
        sites[name] = R.r2(field[j, i])
    row["sites"] = sites
    return row

# ---------------------------------------------------------------- plotting
def _panel(ax, field, lats, lons, title, color, event_pt, labels=False):
    R.add_basemap(ax, labels=labels)
    cs = R.shade(ax, field, lats=lats, lons=lons)
    R.annotate_max(ax, field, lats=lats, lons=lons)
    ax.plot(event_pt[1], event_pt[0], "o", mfc="none", mec="#ffffff", ms=11, mew=1.4,
            transform=ccrs.PlateCarree(), zorder=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    return cs


def plot_grid(panels, out, suptitle, event_pt):
    n = len(panels)
    cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.0 * rows + 0.9),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.11 if rows > 1 else 0.18, top=0.9, wspace=0.04, hspace=0.14)
    cs = None
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False); continue
        fld, la, lo, title, col = panels[k]
        cs = _panel(ax, fld, la, lo, title, col, event_pt, labels=(k == 0))
    R.colorbar(fig, cs, [0.18, 0.04 if rows > 1 else 0.08, 0.64, 0.018 if rows > 1 else 0.03],
               "Precipitation (in)  ·  ○ = observed max", R.CLEVS)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.975)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_dprog(rows, obs_max, obs_area_max, out, title, event_thr=1.0):
    ek = f"ets_{event_thr:g}"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.13, wspace=0.2)
    for key, m in MODELS.items():
        variants = [("", "-")] if key not in ENSEMBLES else [(" mean", "-"), (" p75", "--"), (" max", ":")]
        for suffix, ls in variants:
            name = key + suffix.replace(" ", "_") if suffix else key
            rr = sorted([r for r in rows if r["model"] == name], key=lambda r: r["lead"])
            if not rr or key == "AVRG":
                continue
            x = [r["lead"] for r in rr]
            lab = m["label"] + suffix.replace("p75", "P75")
            a1.plot(x, [r["near_event_max"] for r in rr], ls, marker="o", ms=3.5, color=m["color"], label=lab)
            if suffix in ("", " mean"):
                a2.plot(x, [r.get(ek) for r in rr], ls, marker="o", ms=3.5, color=m["color"], label=m["label"] + suffix)
    a1.axhline(obs_max, color=R.TEXT, ls="--", lw=1, label=f"MRMS max {obs_max:.2f}\"")
    a1.axhline(obs_area_max, color=R.MUTED, ls=":", lw=1, label=f"MRMS 25-km-mean max {obs_area_max:.2f}\"")
    a1.set_ylabel(f"Highest forecast within {EVENT_RADIUS_KM:g} km of observed max (in)")
    a1.set_ylim(bottom=0)
    a2.axhline(0, color=R.MUTED, lw=0.6)
    a2.set_ylabel(f"Placement skill: ETS for the ≥ {event_thr:g}\" area")
    a2.set_ylim(min(-0.1, a2.get_ylim()[0]), max(0.5, a2.get_ylim()[1]))
    for yv, txt in ((1.0, "1 = forecast and observed areas match exactly"), (0.0, "0 = no better than chance")):
        if a2.get_ylim()[0] <= yv <= a2.get_ylim()[1]:
            a2.text(0.99, yv, txt, transform=a2.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color=R.MUTED)
    for a in (a1, a2):
        a.set_xlabel("Lead before window start (h)")
        a.invert_xaxis()
        a.grid(color="#263848", lw=0.6)
        a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
        for sp in a.spines.values():
            sp.set_color(R.LINE)
    a1.set_title("Did they get the peak?", fontsize=11, fontweight="bold")
    a2.set_title(f"Did they put the ≥ {event_thr:g}\" rain in the right place?", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    fig.savefig(out, dpi=125)
    plt.close(fig)

# ---------------------------------------------------------------- build
def nearest_site(lat, lon):
    best = min(R.SITES.items(), key=lambda kv: float(R.dist_km(np.array(kv[1][0]), np.array(kv[1][1]), lat, lon)))
    return best[0].split(" (")[0], round(float(R.dist_km(np.array(best[1][0]), np.array(best[1][1]), lat, lon)))


def case_id_for(start, end, auto=False):
    return f"auto-{end:%Y%m%d%H}" if auto else f"{start:%Y%m%d%H}-{end:%Y%m%d%H}"


def validate_window(start, end):
    hours = hours_between(start, end)
    if start.minute or end.minute:
        raise ValueError("start and end must be whole hours")
    if not MIN_HOURS <= hours <= MAX_HOURS:
        raise ValueError(f"window must be {MIN_HOURS}-{MAX_HOURS} h (got {hours} h)")
    if end > R.utcnow() - R.MRMS_LATENCY:
        raise ValueError("window must end at least ~70 min ago so MRMS Pass 2 is available")
    return hours


def build_case(start=None, end=None, hours=None, title=None, auto=False, requested_by=None):
    R.init_sites()
    if start is None:
        start = end - timedelta(hours=hours)
    start, end = start.replace(minute=0, second=0, microsecond=0), end.replace(minute=0, second=0, microsecond=0)
    hours = validate_window(start, end)
    cid = case_id_for(start, end, auto)
    out_dir = os.path.join(R.CASES_DIR, cid)
    if os.path.exists(os.path.join(out_dir, "case.json")):
        print(f"[case] {cid} already built")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    title = title or f"{hours} h ending {end:%HZ %a %d %b %Y}"
    print(f"[case] {cid}: {start:%Y%m%d %HZ} to {end:%Y%m%d %HZ} ({hours} h)")
    notes = []

    raw = R.fetch_mrms_window(start, end)
    if raw is None:
        raise RuntimeError("MRMS QPE unavailable for this window")
    obs, olat, olon = R.mrms_native(raw)
    view = R.in_view(olat, olon) & np.isfinite(obs)
    jm, im = np.unravel_index(np.argmax(np.where(view, obs, -1)), obs.shape)
    event_pt = (float(olat[jm, im]), float(olon[jm, im]))
    obs_max = float(obs[jm, im])
    area = R.sample_mrms(raw, olat[::25, ::25], olon[::25, ::25], half=12)
    obs_area_max = float(np.nanmax(np.where(R.in_view(olat[::25, ::25], olon[::25, ::25]), area, np.nan)))

    vv = obs[view]
    coverage = {f"{t:g}": round(float((vv >= t).mean()), 4) for t in CASE_THRESHOLDS}
    event_thr = max([t for t in CASE_THRESHOLDS if coverage[f"{t:g}"] >= EVENT_COVERAGE], default=CASE_THRESHOLDS[0])
    print(f"[case] MRMS coverage {coverage}; event threshold {event_thr:g}\"")

    rows, refs_panels, shortest = [], [], {}

    def add(name, fld, la, lo, lead, init):
        rows.append(dict(evaluate(fld, la, lo, raw, event_pt), model=name, init=init.isoformat(), lead=lead))
        if name not in shortest or lead < shortest[name][3]:
            shortest[name] = (fld, la, lo, lead, init)

    for key in MODELS:
        for lead, init in plan_inits(key, start, end):
            f0, f1 = hours_between(init, start), hours_between(init, end)
            try:
                if key in ("REFS", "AVRG"):
                    got = refs_window(init, "lpmm" if key == "REFS" else "avrg", f0, f1)
                    if got is None:
                        raise FileNotFoundError("not archived")
                elif key in ("HRRR", "RRFS", "GFS"):
                    got = det_window(key, init, f0, f1)
                elif key in ("IFS", "AIFS"):
                    got = ecmwf_det("ifs" if key == "IFS" else "aifs-single", init, f0, f1)
                elif key == "GEFS":
                    got = gefs_ens(init, f0, f1)
                else:
                    got = ecmwf_ens(init, f0, f1)
            except Exception as e:
                notes.append(f"{MODELS[key]['label']} {init:%d/%HZ}")
                print(f"  ! {key} {init:%d/%HZ}: {e}")
                continue
            fld, la, lo = got
            if key in ENSEMBLES:
                st = ens_stats(fld)
                for s in STATS:
                    add(f"{key}_{s}", st[s], la, lo, lead, init)
                rows[-1]["members"] = st["n"]
            else:
                add(key, fld, la, lo, lead, init)
                if key == "REFS":
                    refs_panels.append((fld, la, lo, f"REFS LPMM  ·  {lead} h lead ({init:%d/%HZ})", MODELS[key]["color"]))
        if not any(r["model"].startswith(key) for r in rows) and not plan_inits(key, start, end):
            notes.append(f"{MODELS[key]['label']} ({skip_reason(key, start, end)})")

    win_txt = f"{start:%HZ %a %d %b} to {end:%HZ %a %d %b %Y}"
    obs_panel = (obs, olat, olon, f"MRMS {hours}-h QPE (Pass 2)", R.TEXT)
    if refs_panels:
        plot_grid([obs_panel] + refs_panels, os.path.join(out_dir, "refs.png"),
                  f"{title}\nREFS LPMM by lead time vs MRMS, {win_txt}", event_pt)
    det_names = [k for k in ("REFS", "AVRG", "HRRR", "RRFS", "GFS", "IFS", "AIFS") if k in shortest]
    if det_names:
        plot_grid([obs_panel] + [(shortest[k][0], shortest[k][1], shortest[k][2],
                                  f"{MODELS[k]['label']}  ·  {shortest[k][3]} h ({shortest[k][4]:%d/%HZ})", MODELS[k]["color"])
                                 for k in det_names],
                  os.path.join(out_dir, "models.png"),
                  f"{title}\nShortest-lead deterministic and REFS forecasts vs MRMS, {win_txt}", event_pt)
    ens_names = [f"{e}_{s}" for e in ENSEMBLES for s in STATS if f"{e}_{s}" in shortest]
    if ens_names:
        plot_grid([obs_panel] + [(shortest[k][0], shortest[k][1], shortest[k][2],
                                  f"{MODELS[k.split('_')[0]]['label']} {k.split('_')[1].replace('p75', 'P75')}  ·  "
                                  f"{shortest[k][3]} h ({shortest[k][4]:%d/%HZ})", MODELS[k.split('_')[0]]["color"])
                                 for k in ens_names],
                  os.path.join(out_dir, "ens.png"),
                  f"{title}\nGlobal ensembles (member mean, 75th percentile, max) vs MRMS, {win_txt}", event_pt)
    if rows:
        plot_dprog(rows, obs_max, obs_area_max, os.path.join(out_dir, "dprog.png"),
                   f"{title}: forecast trend as the event approached", event_thr)

    obs_sites = {n: R.r2(R.sample_mrms(raw, np.array([[la]]), np.array([[lo]]), half=1)[0, 0])
                 for n, (la, lo, _) in R.SITES.items()}
    site_name, site_km = nearest_site(*event_pt)
    case = {
        "id": cid, "title": title, "auto": auto, "hours": hours, "requested_by": requested_by,
        "start": start.isoformat(), "end": end.isoformat(), "built": R.utcnow().isoformat(timespec="seconds"),
        "obs": {"max": R.r2(obs_max), "area_max": R.r2(obs_area_max), "lat": round(event_pt[0], 3),
                "lon": round(event_pt[1], 3), "near": site_name, "near_km": site_km, "sites": obs_sites,
                "coverage": coverage},
        "event_threshold": event_thr,
        "models": {k: {"label": v["label"], "color": v["color"]} for k, v in MODELS.items()},
        "thresholds": CASE_THRESHOLDS, "event_radius_km": EVENT_RADIUS_KM,
        "rows": rows, "notes": notes,
        "images": [f for f in ("dprog.png", "models.png", "ens.png", "refs.png") if os.path.exists(os.path.join(out_dir, f))],
    }
    case["summary"] = summarize(case)
    R.save_json(os.path.join(out_dir, "case.json"), case, indent=1)
    print(f"[case] {cid}: {len(rows)} forecast rows, MRMS max {obs_max:.2f}\" near {site_name}")
    return out_dir


def summarize(case):
    """Rank models for one case on three 0-1 sub-scores, averaged:
      placement = ETS at the event threshold / 0.5, capped at 1 (0.5 is an excellent ETS for rain)
      peak      = exp(-|ln(forecast near-max / observed near-max at the model's resolution)|)
      volume    = exp(-|ln(domain bias ratio)|)
    Each uses the model's shortest-lead run. Ensemble maxima are excluded from the ranking
    (an envelope, not a forecast) but count toward "earliest signal"."""
    thr = case.get("event_threshold") or (2.0 if case.get("hours", 24) >= 24 else 1.0)
    ek = f"ets_{thr:g}"
    if not any(ek in r for r in case["rows"]):
        thr = max(t for t in (1.0, 2.0, 4.0) if any(f"ets_{t:g}" in r for r in case["rows"])); ek = f"ets_{thr:g}"
    labels = {k: v["label"] for k, v in case.get("models", {}).items()}
    def label(name):
        k, _, st = name.partition("_")
        return labels.get(k, k) + ({"mean": " mean", "p75": " P75", "max": " max"}.get(st, ""))
    def obs_peak(r):
        if r.get("obs_near_max") is not None:
            return r["obs_near_max"]
        return case["obs"]["area_max"] if (r.get("grid_km") or 3) > 10 else case["obs"]["max"]
    by = {}
    for r in case["rows"]:
        by.setdefault(r["model"], []).append(r)
    scores, signals = [], []
    for name, rr in by.items():
        rr = sorted(rr, key=lambda r: r["lead"])
        for r in reversed(rr):                   # longest lead first: how far ahead did it see it?
            op = obs_peak(r)
            if r.get("near_event_max") is not None and op:
                if r["near_event_max"] / op >= 0.5:
                    signals.append((r["lead"], name)); break
        if name.endswith("_max"):
            continue
        r0 = rr[0]
        op = obs_peak(r0)
        ratio = (r0["near_event_max"] / op) if (r0.get("near_event_max") is not None and op) else None
        ets = r0.get(ek)
        bias = r0.get("bias_ratio")
        sub = {"placement": None if ets is None else max(0.0, min(1.0, ets / 0.5)),
               "peak": None if not ratio else float(np.exp(-abs(np.log(max(ratio, 1e-3))))),
               "volume": None if not bias else float(np.exp(-abs(np.log(max(bias, 1e-3)))))}
        vals = [v for v in sub.values() if v is not None]
        if not vals:
            continue
        scores.append({"model": name, "label": label(name), "lead": r0["lead"], "init": r0["init"],
                       "score": round(float(np.mean(vals)), 2), "ets": ets, "peak_ratio": None if ratio is None else round(ratio, 2),
                       "near_max": r0.get("near_event_max"), "obs_peak": op, "bias": bias,
                       "sub": {k: (None if v is None else round(v, 2)) for k, v in sub.items()}})
    scores.sort(key=lambda x: -x["score"])
    signals.sort(key=lambda x: -x[0])
    lines = []
    if scores:
        b = scores[0]
        lines.append(f"Best overall: {b['label']} (score {b['score']:.2f}).")
        p = max([x for x in scores if x["ets"] is not None], key=lambda x: x["ets"], default=None)
        if p:
            lines.append(f"Best placement of the ≥ {thr:g}″ area: {p['label']} (ETS {p['ets']:.2f}).")
        k = min([x for x in scores if x["peak_ratio"]], key=lambda x: abs(np.log(x["peak_ratio"])), default=None)
        if k:
            lines.append(f"Closest on the peak: {k['label']}, {k['near_max']:.2f}″ vs {k['obs_peak']:.2f}″ observed at its resolution "
                         f"({k['peak_ratio'] * 100:.0f}%).")
        dry = [x for x in scores if x["bias"] is not None and x["bias"] < 0.5]
        if dry:
            lines.append("Far too dry overall (under half the observed rain volume): "
                         + ", ".join(f"{x['label']} {x['bias'] * 100:.0f}%" for x in sorted(dry, key=lambda x: x["bias"])) + ".")
    if signals:
        lead, name = signals[0]
        lines.append(f"Earliest signal: {label(name)} reached half the observed peak {lead} h before the window started."
                     if lead else f"Earliest signal: only at 0-h lead ({label(name)}); no model saw it coming beforehand.")
    else:
        lines.append("No model reached half the observed peak at any lead.")
    return {"version": SUMMARY_VERSION, "event_threshold": thr, "lines": lines, "scores": scores,
            "method": "Score = mean of placement (ETS at the event threshold ÷ 0.5, capped at 1), peak "
                      "(e^-|ln ratio| of forecast to observed near-max at the model's resolution) and volume "
                      "(e^-|ln bias|), each from the model's shortest-lead run."}


def refresh_summaries():
    """Add or update summaries on stored cases (no downloads), so older cases get the scorecard too."""
    for cj in glob.glob(os.path.join(R.CASES_DIR, "*", "case.json")):
        c = R.load_json(cj, None)
        if c and (c.get("summary") or {}).get("version") != SUMMARY_VERSION:
            c["summary"] = summarize(c)
            R.save_json(cj, c, indent=1)


def delete_case(case_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id or ""):
        raise ValueError("bad case id")
    d = os.path.join(R.CASES_DIR, case_id)
    if not os.path.exists(os.path.join(d, "case.json")):
        raise FileNotFoundError(f"no case {case_id}")
    shutil.rmtree(d)


def prune_auto(keep):
    autos = sorted(glob.glob(os.path.join(R.CASES_DIR, "auto-*")))
    for d in (autos[:-keep] if keep else autos):
        shutil.rmtree(d, ignore_errors=True)


def republish():
    if not os.path.isdir(R.SITE_CACHE):
        print("[case] no cached site yet; run the main workflow once first")
        return False
    if os.path.isdir(R.OUT_DIR):
        shutil.rmtree(R.OUT_DIR)
    shutil.copytree(R.SITE_CACHE, R.OUT_DIR)
    if os.path.exists(R.INDEX_SRC):
        shutil.copy(R.INDEX_SRC, os.path.join(R.OUT_DIR, "index.html"))
    meta_p = os.path.join(R.OUT_DIR, "meta.json")
    meta = R.load_json(meta_p, {})
    refresh_summaries()
    meta.setdefault("products", {})["cases"] = R.publish_cases(R.OUT_DIR) > 0
    R.save_json(meta_p, meta, indent=1)
    return True


def parse_time(s):
    s = re.sub(r"[^0-9]", "", s or "")
    if len(s) != 10:
        raise ValueError(f"time must be YYYYMMDDHH UTC, got {s!r}")
    return datetime.strptime(s, "%Y%m%d%H").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="window start, YYYYMMDDHH UTC")
    ap.add_argument("--end", required=True, help="window end, YYYYMMDDHH UTC")
    ap.add_argument("--hours", type=int, help="alternative to --start")
    ap.add_argument("--title", default="")
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args()
    end = parse_time(a.end)
    start = parse_time(a.start) if a.start else end - timedelta(hours=a.hours or 24)
    build_case(start=start, end=end, title=a.title or None)
    if not a.no_publish:
        republish()
