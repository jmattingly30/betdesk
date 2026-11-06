"""Keep the whole day's games after they start (J 2026-11-06: log a missed bet or review a game after its start time).

Every writer rebuilds its file from a feed that drops a game once it starts (or loses its pregame line). Before writing,
each one calls keep() with the file it is about to overwrite: a game whose kickoff has passed (within KEEP_H hours) keeps its
LAST PRE-GAME record from the old file, frozen, and is added back if the new build left it out. Games not started yet are
always the new build. Frozen records get "pre": the old file's made stamp, so the desk can say "as of pre-game".

    from pregame import keep
    out["games"] = keep(f"{APP}/nba_ab.json", out["games"], kick=lambda gid, r: r.get("k"))      # dict keyed by id
    rows = keep(out_f, rows, kick=lambda i, r: r["start"], key="fid")                             # list of dicts
"""
import json, os, gzip, datetime as dt

KEEP_H = 30   # long enough for a late game and the whole Mountain-time day, short enough that yesterday drops off


def _t(k):
    try: return dt.datetime.fromisoformat(str(k).replace("Z", "+00:00"))
    except Exception: return None


def _load(path, field):
    try:
        d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    except Exception: return None, None
    made = d.get("made") if isinstance(d, dict) else None
    for f in (field.split('.') if field else []): d = d.get(f) if isinstance(d, dict) else None   # 'NBA.games' = nested
    return d, made


def keep(path, new, kick, key="id", field="games", now=None, hours=KEEP_H, made=None, merge=None):
    """new: dict {id: rec} or list of recs (with rec[key]). kick(id, rec) -> ISO kickoff (or None = never frozen).
    field: where the games sit in the old file (None when the file is the list itself).
    merge(old_rec, new_rec) -> rec, optional: how to combine a started game present in both (default: old record, frozen)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    old, old_made = _load(path, field)
    if not old: return new
    if not (made or old_made):
        try: old_made = dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        except Exception: pass
    made = made or old_made
    is_list = isinstance(new, list)
    loose = [r for r in new if r.get(key) is None] if is_list else []   # rows without an id are passed through untouched
    N = {str(r[key]): r for r in new if r.get(key) is not None} if is_list else {str(k): v for k, v in new.items()}
    O = {str(r[key]): r for r in old if r.get(key) is not None} if isinstance(old, list) else {str(k): v for k, v in old.items()}
    kept = 0
    for gid, r in O.items():
        t = _t(kick(gid, r) or (N.get(gid) and kick(gid, N[gid])))
        if not t or not (now - dt.timedelta(hours=hours) <= t <= now): continue   # not started yet, or too old
        if gid in N and merge: N[gid] = merge(r, N[gid])
        else: N[gid] = {**r, "pre": r.get("pre") or made} if isinstance(r, dict) else r
        kept += 1
    if kept: print(f"pregame: kept {kept} started game(s) from {os.path.basename(path)}")
    if not is_list: return N
    return sorted(list(N.values()) + loose, key=lambda r: str(kick(str(r.get(key)), r) or ""))
