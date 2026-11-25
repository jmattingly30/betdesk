"""Data for J's Monitor page (desk simplification step 3) -> app/data/monitor.json. Morning chain C8b, after both graders.

Sections, in the order of the three steps:
  accuracy    our projection (Model A, unblended) game by game: predicted, closing-line implied and actual team scores, so the
              page can compare model vs reality, line vs reality, both, and model vs line (J 2026-11-24 Compare toggle).
              NFL/NBA/CBB/CFB from models/*_ab/history/runs.jsonl (last pregame row; the line = the sharp-weighted market
              number in the last row that had one), NHL from models/nhl_sim/history/log.jsonl, MLB from the scorecard.
  props       player stat accuracy vs actual and vs the player's own average, calibration (scorecard; prop lines are not
              graded yet: props_vs_close.json is stale since 2026-10-28)
  challengers every other model the scorecard and model_a.json grade, beside ours
  price       the price check: trust table, demoted buckets, label counts (priced_bets.json), graded buckets
              (price_grades.json), and the old tiers' record for comparison (scorecard lineflags, *_ab grade.json)
  feeds       every data file's age; Odds API credits left
python3 models/pricing/monitor.py"""
import json, os, sys, gzip, glob, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"
sys.path.insert(0, f"{ROOT}/models/grading")
NOW = dt.datetime.now(dt.timezone.utc)

def jl(p, d=None):
    try: return json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
    except (OSError, ValueError): return d

def finals_with_start(sp):
    """{game id: (home, away, start UTC)} for finished games (same sources as models/grading/grade.py finals())"""
    import io, urllib.request, pandas as pd
    from grade import GAME_SRC
    with urllib.request.urlopen(GAME_SRC[sp], timeout=90) as r: b = r.read()
    if sp == 'NFL':
        g = pd.read_csv(io.BytesIO(b)); g = g[g.home_score.notna()]
        k = pd.to_datetime(g.gameday + ' ' + g.gametime.fillna('13:00')).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        return {str(i): (float(h), float(a), t.to_pydatetime()) for i, h, a, t in zip(g.game_id, g.home_score, g.away_score, k)}
    g = pd.read_parquet(io.BytesIO(b))
    done = g.status.eq('STATUS_FINAL') if 'status' in g else g.status_type_completed.astype(bool)
    g = g[done]; k = pd.to_datetime(g['date' if 'date' in g else 'game_date'], utc=True)
    return {str(i): (float(h), float(a), t.to_pydatetime()) for i, h, a, t in zip(g.game_id, g.home_score, g.away_score, k)}

START = {}
def pregame(r):
    """logged before the start (models/cbb_ab/grade.py rule; start from the schedule, else the row, else 05:00 UTC on the game date)"""
    try:
        t = dt.datetime.strptime(r['t'], '%Y-%m-%d %H:%M UTC').replace(tzinfo=dt.timezone.utc)
        if START.get(str(r['id'])): return t < START[str(r['id'])]
        if r.get('k'): return t < dt.datetime.fromisoformat(r['k'].replace('Z', '+00:00'))
        return t < dt.datetime.fromisoformat(r['d']).replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=5)
    except (KeyError, ValueError, TypeError):
        return False

def last_rows(path):
    """{id: (last row, last row that had a spread and total market)}"""
    L, M = {}, {}
    if not os.path.exists(path): return L, M
    for ln in open(path):
        try: r = json.loads(ln)
        except ValueError: continue
        if not pregame(r): continue          # rows logged after the start carry in-game / final information
        L[r['id']] = r; B = r.get('B') or {}
        if (B.get('spread') or {}).get('market') is not None and (B.get('total') or {}).get('market') is not None: M[r['id']] = r
    return L, M

def accuracy():
    out = []
    for sp, f in (('NFL', 'nfl_ab'), ('NBA', 'nba_ab'), ('CBB', 'cbb_ab'), ('CFB', 'cfb_ab')):
        try: F = finals_with_start(sp)
        except Exception as e: print(sp, 'finals failed', e); continue
        START.update({i: v[2] for i, v in F.items()})
        L, M = last_rows(f"{ROOT}/models/{f}/history/runs.jsonl")
        for gid, r in L.items():
            y = F.get(str(gid))
            if not y or not r.get('A'): continue
            m = M.get(gid); lh = la = None
            if m:
                Mm, Tt = m['B']['spread']['market'], m['B']['total']['market']; lh, la = round((Tt + Mm) / 2, 2), round((Tt - Mm) / 2, 2)
            out.append([sp, r.get('d') or '', round(r['A'][0], 2), round(r['A'][1], 2), lh, la, y[0], y[1]])
    # NHL: sim log (lh/la = goals implied by the moneyline and total in that row); finals via the pricing grader
    try:
        import importlib.util as iu
        spec = iu.spec_from_file_location('pricing_grade', f"{HERE}/grade.py"); pg = iu.module_from_spec(spec); spec.loader.exec_module(pg)
        NF = pg.nhl_finals()
    except Exception as e:
        print('NHL finals failed', e); NF = {}
    L = {}
    p = f"{ROOT}/models/nhl_sim/history/log.jsonl"
    if os.path.exists(p):
        for ln in open(p):
            try: r = json.loads(ln)
            except ValueError: continue
            if r.get('k') and r.get('made') and r['made'] < r['k']: L[r['id']] = r
    for gid, r in L.items():
        y = NF.get(str(gid))
        if y: out.append(['NHL', r.get('d') or '', r['hs'], r['as'], r.get('lh'), r.get('la'), y[0], y[1]])
    out.sort(key=lambda x: (x[1], x[0]))
    return out

def feeds():
    rows = []
    for f in sorted(glob.glob(f"{APP}/*.json")) + sorted(glob.glob(f"{APP}/odds/props_*.json")):
        b = os.path.basename(f)
        if b in ('line_history.json', 'monitor.json') or b.startswith(('odds_test_', 'tennis', 'soccer_', 'sims_tennis', 'mlb_slate_sample')): continue
        d = jl(f, {}) or {}; m = d.get('made') or d.get('updatedAt') if isinstance(d, dict) else None
        rows.append(dict(file=('odds/' if '/odds/' in f else '') + b, made=m, kb=round(os.path.getsize(f) / 1024)))
    s = jl(f"{APP}/game_slates.json", {}) or {}
    credits = {sp: (s.get(sp) or {}).get('credits_left') for sp in ('NFL', 'CFB', 'CBB', 'NBA', 'MLB') if isinstance(s.get(sp), dict)}
    return rows, credits

def main():
    SC = jl(f"{APP}/scorecard.json", {}) or {}; PB = jl(f"{APP}/priced_bets.json", {}) or {}; PG = jl(f"{APP}/price_grades.json", {}) or {}
    T = jl(f"{HERE}/trust.json", {}) or {}; MA = jl(f"{APP}/model_a.json", {}) or {}; NS = jl(f"{APP}/nhl_sim.json", {}) or {}
    old_games = {}
    for sp, f in (('NFL', 'nfl_ab'), ('NBA', 'nba_ab'), ('CBB', 'cbb_ab'), ('CFB', 'cfb_ab')):
        B = (jl(f"{ROOT}/models/{f}/history/grade.json", {}) or {}).get('B') or {}
        old_games[sp] = {k: v for k, v in B.items() if k.split('_')[1] in ('A', 'B') and len(k.split('_')) == 2}
    fd, credits = feeds()
    try: acc = accuracy()
    except Exception as e: print('accuracy failed', repr(e)); acc = []
    out = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'),
               accuracy=dict(cols=['sp', 'd', 'pred_home', 'pred_away', 'line_home', 'line_away', 'act_home', 'act_away'], rows=acc,
                             mlb=[g for g in SC.get('games') or [] if g.get('sport') == 'MLB'], nhl_sim=NS.get('grade')),
               props=dict(stats=SC.get('props') or [], calib=SC.get('calib') or [], note='Projections vs actual results. Prop LINES are not graded yet (props_vs_close.json last built 2026-10-28).'),
               challengers=dict(scorecard=SC.get('games') or [], model_a=(MA.get('grade') or {}).get('sports') or {}),
               price=dict(made=PB.get('made'), counts=PB.get('counts'), demoted=PB.get('demoted'), guarded_counts=PB.get('guarded_counts'),
                          trust=dict(games=T.get('games'), props=T.get('props'), labels=T.get('labels'), timing=T.get('timing'), refit=T.get('refit')),
                          graded={k: PG.get(k) for k in ('made', 'graded', 'pending', 'staked', 'label', 'guard', 'big_dog', 'timing', 'conf')},
                          rows=(PG.get('rows') or [])[:300], old_flags=SC.get('lineflags', {}).get('by'), old_daily=SC.get('lineflags', {}).get('daily'), old_games=old_games),
               feeds=dict(files=fd, credits=credits))
    json.dump(out, open(f"{APP}/monitor.json", 'w'), separators=(',', ':'))
    print('monitor.json:', len(acc), 'graded games,', len(fd), 'files')

if __name__ == '__main__':
    main()
