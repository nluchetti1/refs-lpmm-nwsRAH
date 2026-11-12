# Central NC REFS QPF

48-h REFS QPF, verification and look-backs for the NWS Raleigh CWA. Stations are every online ASOS/AWOS IEM assigns to WFO RAH, refreshed each run.

## One codebase, two sites
`refs_qpf.py`, `lookback.py`, `request_case.py`, `index.html` and the workflows are identical
for every deployment. Everything site-specific is in `site_config.json`:

| key | what it sets |
|---|---|
| `branding` | page title and subtitle |
| `extent` | map box `[W, E, S, N]` |
| `sites` | fixed points: `name`, `lat`, `lon`, `kind` (`major` bold / `asos` / `awos`), optional `label` |
| `iem_stations` | pull online ASOS/AWOS from IEM: `network` (e.g. `NC_ASOS`), `wfo`, `major` ids |
| `fallback_stations` | used only if IEM and the cached list are both unavailable |
| `primary`, `timelines`, `map_labels` | by name or station id; `timelines: "all"` makes one per site |
| `thresholds` | 48-h paintball levels (in) |
| `regime` | sounding site for the flow regime: `name`, `point`, `raob_stations`, `raob_hours` |

A fix to the shared files can be copied between repos unchanged.

## Setup
1. Push everything, including `.github/` (workflows and ISSUE_TEMPLATE).
2. Settings → Actions → General → Workflow permissions → Read and write.
3. Settings → General → Features → Issues on (public look-back requests).
4. Actions → "REFS QPF (build site)" → Run workflow (creates the `data` branch).
5. Settings → Pages → Deploy from branch → `gh-pages` / root.

Look-backs: the site's Look back tab (anyone with a GitHub account), Actions → "Look back"
(you), or automatically for the 24 h ending 12Z after each 12Z cycle.

## MRMS FLASH
`flash.py` (workflow "MRMS FLASH refresh", about every 30 min) redraws the FLASH panel from the
latest NCEP files: CREST max unit streamflow (flooding likely > 2, significant > 6 m³ s⁻¹ km⁻²),
QPE ÷ FFG (max of 1/3/6 h), QPE recurrence interval and CREST soil saturation. Look-backs add the
window maxima of the same products. NCEP keeps ~25 h; older windows use the AWS archive if present.

## Flash flood guidance
The FFG ratio tab divides REFS LPMM 1/3/6-h totals by WPC's CONUS gridded FFG
(ftp.wpc.ncep.noaa.gov/workoff/ffg, 5 km, 00/06/12/18Z).

## Refreshing FLASH faster
GitHub's own cron rarely runs on time below ~15 min. To trigger "MRMS FLASH refresh" from an
external scheduler, create a fine-grained token with Actions: Read and write on this repo and call:

    POST https://api.github.com/repos/OWNER/REPO/actions/workflows/flash_update.yml/dispatches
    Authorization: Bearer <token>
    Accept: application/vnd.github+json
    {"ref": "main"}

GitHub Pages deployed from a branch has a soft limit of 10 builds per hour, and every deploy
(FLASH, the 6-hourly build, look-backs) counts, so keep FLASH at 10 min or slower. Each FLASH
run loops the last FLASH_LOOP_SCANS (default 12) 10-min scans of QPE % of FFG and CREST unit
streamflow, ending at the newest.
