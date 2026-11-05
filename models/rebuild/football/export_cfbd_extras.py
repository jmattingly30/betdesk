"""One-time export of CFBD preseason inputs for the CFB score rebuild (run on your own computer; CFBD is blocked
from the cloud). About 40 API calls in total.

    export CFBD_API_KEY=...            (Windows: set CFBD_API_KEY=...)
    python export_cfbd_extras.py

Writes three CSVs next to this script; upload them to the "CFB v2" Drive folder and tell Claude:
  cfbd_talent.csv       team talent composite (247), 2015-2026
  cfbd_recruiting.csv   team recruiting class rank and points, 2010-2026
  cfbd_coaches.csv      head coach by team and season, with his record, 2010-2026
"""
import csv
import os
import time

import requests

BASE = "https://api.collegefootballdata.com"
KEY = os.environ.get("CFBD_API_KEY")
if not KEY:
    raise SystemExit("Set CFBD_API_KEY first")
S = requests.Session()
S.headers.update({"Authorization": f"Bearer {KEY}", "Accept": "application/json"})


def get(endpoint, **params):
    for attempt in range(3):
        r = S.get(BASE + endpoint, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"CFBD {endpoint} {params} -> HTTP {r.status_code}: {r.text[:200]}")


def write(name, rows, cols):
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(name, len(rows), "rows")


talent = []
for y in range(2015, 2027):
    for r in get("/talent", year=y):
        talent.append({"season": y, "team": r.get("team") or r.get("school"), "talent": r.get("talent")})
write("cfbd_talent.csv", talent, ["season", "team", "talent"])

rec = []
for y in range(2010, 2027):
    for r in get("/recruiting/teams", year=y):
        rec.append({"season": y, "team": r.get("team"), "rank": r.get("rank"), "points": r.get("points")})
write("cfbd_recruiting.csv", rec, ["season", "team", "rank", "points"])

coaches = []
for y in range(2010, 2027):
    for c in get("/coaches", year=y):
        name = f"{c.get('firstName', c.get('first_name', ''))} {c.get('lastName', c.get('last_name', ''))}".strip()
        for s in c.get("seasons", []):
            coaches.append({"season": s.get("year"), "team": s.get("school"), "coach": name, "games": s.get("games"),
                            "wins": s.get("wins"), "losses": s.get("losses"), "preseason_rank": s.get("preseasonRank"),
                            "hire_date": c.get("hireDate") or c.get("hire_date")})
write("cfbd_coaches.csv", coaches, ["season", "team", "coach", "games", "wins", "losses", "preseason_rank", "hire_date"])
print("Done. Upload the three CSVs to the CFB v2 Drive folder.")
