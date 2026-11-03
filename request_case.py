"""
Build a look-back from a GitHub issue created with the look-back issue form.

Reads ISSUE_BODY, ISSUE_AUTHOR, ISSUE_NUMBER and GITHUB_REPOSITORY from the
environment (the body is untrusted text: only the three fields are parsed and
each is validated), writes the reply to reply.md, and exits 0 when a case is
ready, 2 when the request is rejected.
"""
import os
import re
import sys
from datetime import timedelta

import refs_qpf as R
import lookback as L

MAX_PER_AUTHOR_PER_DAY = 3


def field(body, label):
    m = re.search(rf"###\s*{re.escape(label)}\s*\n+(.+?)(?:\n###|\Z)", body, re.S)
    v = (m.group(1).strip() if m else "")
    return "" if v == "_No response_" else v


def site_url(case_id):
    repo = os.environ.get("GITHUB_REPOSITORY", "owner/repo")
    owner, name = repo.split("/", 1)
    base = os.environ.get("PAGES_URL") or f"https://{owner}.github.io/{name}"
    return f"{base.rstrip('/')}/#case-{case_id}"


def reply(text, code):
    with open("reply.md", "w") as f:
        f.write(text)
    print(text)
    sys.exit(code)


def delete_request(title, author):
    owner = os.environ.get("REPO_OWNER", "")
    cid = title.split(":", 1)[1].strip().split()[0] if ":" in title else ""
    if author != owner:
        reply(f"Only the repository owner (@{owner}) can delete cases. Use **Hide** on the site to hide it just for you.", 2)
    try:
        L.delete_case(cid)
    except Exception as e:
        reply(f"Couldn't delete `{cid}`: {e}", 2)
    L.republish()
    reply(f"Deleted case `{cid}`. The site updates within a minute or two.", 0)


def main():
    body = os.environ.get("ISSUE_BODY", "")
    author = os.environ.get("ISSUE_AUTHOR", "unknown")
    title = os.environ.get("ISSUE_TITLE", "")
    if title.startswith("Delete case:"):
        delete_request(title, author)
    try:
        start = L.parse_time(field(body, "Window start (UTC)"))
        end = L.parse_time(field(body, "Window end (UTC)"))
        L.validate_window(start, end)
    except Exception as e:
        raw = f"start `{field(body, 'Window start (UTC)')}`, end `{field(body, 'Window end (UTC)')}`"
        e = f"{e} (received {raw})"
        reply(f"Couldn't build this look-back: {e}\n\nEdit the times to `YYYYMMDDHH` UTC and open a new request.", 2)
    title = re.sub(r"[^\w\s\-\.,/()&'’]", "", field(body, "Title"))[:80].strip() or None

    cid = L.case_id_for(start, end)
    if os.path.exists(os.path.join(R.CASES_DIR, cid, "case.json")):
        reply(f"This window is already built: {site_url(cid)}", 0)

    recent = 0
    for cj in L.glob.glob(os.path.join(R.CASES_DIR, "*", "case.json")):
        c = R.load_json(cj, {})
        if c.get("requested_by") == author and c.get("built", "") >= (R.utcnow() - timedelta(days=1)).isoformat():
            recent += 1
    if recent >= MAX_PER_AUTHOR_PER_DAY:
        reply(f"Limit is {MAX_PER_AUTHOR_PER_DAY} look-backs per person per day. Please try again tomorrow.", 2)

    try:
        L.build_case(start=start, end=end, title=title, requested_by=author)
    except Exception as e:
        reply(f"The build failed: {e}", 2)
    L.republish()
    c = R.load_json(os.path.join(R.CASES_DIR, cid, "case.json"), {})
    missing = f"\n\nNot available for this window: {', '.join(c['notes'])}." if c.get("notes") else ""
    window = f"{start:%HZ %a %d %b %Y} to {end:%HZ %a %d %b %Y} ({L.hours_between(start, end)} h)"
    reply(f"Ready: {site_url(cid)}\n\nWindow: **{window}**, read from your request as "
          f"`{start:%Y%m%d%H}` to `{end:%Y%m%d%H}` UTC.\n\nMRMS max {c['obs']['max']}″ near {c['obs']['near']}.{missing}\n\n"
          f"The page updates within a minute or two of this comment.", 0)


if __name__ == "__main__":
    main()
