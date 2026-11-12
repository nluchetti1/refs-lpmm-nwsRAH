"""
Refresh only the MRMS FLASH panel on the published site (runs every 30 min).

Starts from data/site_cache (the last full build), redraws flash.png and the
meta.json "flash" block, then writes ./site for deployment. The 6-hourly main
build also draws a FLASH snapshot, so the panel is never older than ~30 min
while this workflow runs, and never older than 6 h if it stops.
"""
import os
import shutil
import sys

import refs_qpf as R
import lookback as L


def main():
    R.init_sites()
    if not os.path.isdir(R.SITE_CACHE):
        print("[flash] no cached site yet; run the main workflow once first")
        sys.exit(1)
    if os.path.isdir(R.OUT_DIR):
        shutil.rmtree(R.OUT_DIR)
    shutil.copytree(R.SITE_CACHE, R.OUT_DIR)
    if os.path.exists(R.INDEX_SRC):
        shutil.copy(R.INDEX_SRC, os.path.join(R.OUT_DIR, "index.html"))
    fl = R.build_flash_now(R.OUT_DIR)
    if not fl:
        sys.exit(1)
    meta_p = os.path.join(R.OUT_DIR, "meta.json")
    meta = R.load_json(meta_p, {})
    meta["flash"] = fl
    meta.setdefault("products", {})["flash"] = True
    meta["products"]["cases"] = R.publish_cases(R.OUT_DIR) > 0
    R.save_json(meta_p, meta, indent=1)
    print(f"[flash] site refreshed, FLASH valid {fl['valid']}")


if __name__ == "__main__":
    main()
