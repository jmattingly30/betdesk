"""Tennis Model A + Model B for the Bet Desk (paper only). Reads app/data/tennis_model.json (live win model, rebuilt daily by the
tennis thread) and app/data/odds/tennis.json (OddsPapi moneylines, openers + current), writes app/data/tennis_ab.json and appends
history/runs.jsonl. Run after tennis_model.json is rebuilt.
Model A: point -> game -> set -> match engine (lib.py) on our win %; serve level from tour serve-point rates + players'
tiebreak-set rate (params.json). Outputs total games (median, 80% range), each player's games, set-score chances, fair
total-games line and fair games handicap.
Model B: blend logit(market no-vig) + logit(Model A win %) with weights from model_b_test.json (2022-25 out of sample);
edge at the best Colorado book's current price. Tier B (lean) = women's ITF / WTA, edge 6-10%, both players 30+ rated matches,
price 1.2-5.0. No tier A anywhere: no tennis bet type holds out of sample AND in the 2026 walk-forward. Totals and handicaps:
numbers only (no lines in our feed; Kaggle tests found no edge)."""
import os, json, re, datetime as dt, numpy as np
import lib
D = '/mnt/project-files/sports-analytics/app/data'
P = json.load(open(f'{lib.HERE}/params.json'))
CO = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'espnbet', 'fanatics', 'betrivers', 'bet365']
# model weight in the blend (logit market, logit model), 2022-25 OOS fits vs bet365 opener (model_b_test.json ml_weights)
WB = {('m', 'Tour'): (1.0, 0.12), ('m', 'Challenger'): (1.0, 0.06), ('m', 'ITF'): (1.0, 0.06), ('w', 'Tour'): (0.9, 0.25), ('w', 'ITF'): (1.0, 0.21)}
SLAM = re.compile(r'australian open|roland|french open|wimbledon|us open', re.I)

def level(cat):
    c = (cat or '').lower()
    sx = 'w' if ('wta' in c or 'women' in c) else 'm'
    lv = 'ITF' if 'itf' in c else 'Challenger' if 'challenger' in c else 'Tour'
    return sx, lv

def mu_of(sx, lv, surf, bo, i1, i2):
    k = f'{sx}|{lv}|{surf}|{bo}'
    b = P['base'].get(k, P['base'].get(f'{sx}|{lv}|hard|{bo}', 0.4))
    g0, t0 = P['style_means'].get(f'{sx}|{lv}', P['style_means'].get(f'{sx}|Tour'))
    dt_ = 0.
    for i in (i1, i2):
        if not i or i.get('tiebreak_rate') is None: continue
        sets = 2.4 * (i.get('n') or 0); w = sets / (sets + 25)
        dt_ += (i['tiebreak_rate'] - t0) * w
    return b + P['c_tb'] * dt_   # games-per-set term fit to 0 (net zero), only tiebreak rate kept

def fair2(a, b):
    ia, ib = 1 / a, 1 / b; return ia / (ia + ib)
lg = lambda p: np.log(p / (1 - p))

def main():
    TM = json.load(open(f'{D}/tennis_model.json'))
    made = dt.datetime.fromisoformat(TM['made'].replace('Z', '+00:00'))
    if dt.datetime.now(dt.timezone.utc) - made > dt.timedelta(hours=36):
        print('tennis_model.json is from', TM['made'], '(over 36 h old); tennis_ab.json left as is'); return
    OD = {m['fid']: m for m in json.load(open(f'{D}/odds/tennis.json'))} if os.path.exists(f'{D}/odds/tennis.json') else {}
    out = []
    for m in TM['matches']:
        if m.get('p_model') is None: continue
        sx, lv = level(m.get('cat'))
        surf = (m.get('surface') or 'hard').lower()
        surf = 'hard' if surf.startswith('ind') or surf not in ('clay', 'grass', 'hard', 'carpet') else surf
        bo = 5 if (sx == 'm' and SLAM.search(m.get('tournament') or '') and lv == 'Tour') else 3
        i1 = (m.get('info') or {}).get('p1'); i2 = (m.get('info') or {}).get('p2')
        mu = mu_of(sx, lv, surf, bo, i1, i2)
        p = float(m['p_model'])
        r = lib.project(p, mu, bo, sx)
        sc = {f'{a}-{b}': round(float(v), 3) for (a, b), v in zip(r['scores'], r['ss'])}
        g1 = r['ga_mean']; g2 = r['tot_mean'] - g1
        # fair lines: total where over ~= under, handicap for p1 where cover ~= 50%
        k = np.arange(len(r['tg'])); cdf = np.cumsum(r['tg']); tot_fair = float(k[np.searchsorted(cdf, .5)]) + 0.5
        mg = r['mg']; mm = np.arange(len(mg)) - lib.GMAX; c = np.cumsum(mg)
        hs = np.arange(-15.5, 16, 1.0)   # p1 half-lines; pick the one whose cover chance is closest to 50%
        hcp_fair = float(hs[np.argmin([abs(mg[mm + h > 0].sum() - .5) for h in hs])])
        A = dict(win_p1=round(p, 3), tot_med=r['tot_med'], tot_range=[r['tot_lo'], r['tot_hi']], tot_mean=round(r['tot_mean'], 1),
                 games_p1=round(g1, 1), games_p2=round(g2, 1), margin_p1=round(r['mar_mean'], 1), sets=sc,
                 p_straight=round(float(r['ss'][0] + r['ss'][len(r['ss']) // 2]), 3), tot_line_fair=tot_fair, hcp_p1_fair=hcp_fair,
                 serve_pts=[round(float(lib.sg(mu + r['d'] / 2)), 3), round(float(lib.sg(mu - r['d'] / 2)), 3)], best_of=bo)
        B = dict(tier='C', note='no prices yet')
        om = OD.get(m.get('fid'))
        if om:
            bk = om.get('books', {})
            ref = None
            for name in ('pinnacle', 'bet365'):
                q = (bk.get(name) or {}).get('now') or (bk.get(name) or {}).get('open')
                if q and q[1] > 1 and q[2] > 1: ref = (name, fair2(q[1], q[2])); break
            opn = None
            for name in ('pinnacle', 'bet365', 'draftkings', 'fanduel'):
                q = (bk.get(name) or {}).get('open')
                if q and q[1] > 1 and q[2] > 1: opn = (name, fair2(q[1], q[2])); break
            if ref:
                wo, wm = WB.get((sx, lv), (1.0, 0.1))
                pb = float(lib.sg(wo * lg(ref[1]) + wm * lg(np.clip(p, .01, .99))))
                best = None
                for name in CO:
                    q = (bk.get(name) or {}).get('now') or (bk.get(name) or {}).get('open')
                    if not q: continue
                    for side, pr, pp in ((1, q[1], pb), (2, q[2], 1 - pb)):
                        if 1.2 <= pr <= 5 and (best is None or pp * pr - 1 > best[0]): best = (pp * pr - 1, name, side, pr)
                n1 = (i1 or {}).get('n') or 0; n2 = (i2 or {}).get('n') or 0
                B = dict(market=ref[0], p_market=round(ref[1], 3), p_open=round(opn[1], 3) if opn else None, open_book=opn[0] if opn else None,
                         p_blend=round(pb, 3), model_weight=wm, tier='C')
                if best:
                    e, name, side, pr = best
                    B.update(edge=round(e, 3), book=name, side=side, pick=m['p1'] if side == 1 else m['p2'], price=pr)
                    lean = sx == 'w' and .06 <= e < .10 and min(n1, n2) >= 30   # 10%+ edges lost (-10%, n 141): the market knows something
                    B['tier'] = 'B' if lean else 'C'
                    B['note'] = ('Lean (paper): women, 6-10% edge, both players 30+ rated matches. 2022-25 at the bet365 opener, women\'s ITF 6-10%: '
                                 'about +7.7% ROI on 324 (6%+ all: +2.3%, se 4.9), CLV about +2.8%. Not proven in the 2026 walk-forward.') if lean else \
                                ('Pass: men\'s model weight is small (6-12%), edges rarely clear the vig.' if sx == 'm' else
                                 ('Pass: edge 10%+ (in 2022-25 these lost 10%: the market usually knows something, like an injury).' if e >= .10 else 'Pass: edge under 6% or a player with under 30 rated matches.'))
        out.append(dict(p1=m['p1'], p2=m['p2'], start=m.get('start'), cat=m.get('cat'), tournament=m.get('tournament'), fid=m.get('fid'),
                        tex_mid=m.get('tex_mid'), surface=surf, sex=sx, level=lv, rated=m.get('rated'), A=A, B=B))
    res = dict(made=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), src_made=TM.get('made'),
               note='Tennis Model A (point sim on our win %) + Model B (paper). Totals/handicaps are numbers only: no lines in our feed and '
                    'Kaggle 2021-25 tests found no edge. No tier A in tennis.', matches=out)
    json.dump(res, open(f'{D}/tennis_ab.json', 'w'), separators=(',', ':'))
    os.makedirs(f'{lib.HERE}/history', exist_ok=True)
    with open(f'{lib.HERE}/history/runs.jsonl', 'a') as f:
        for x in out: f.write(json.dumps(dict(made=res['made'], **{k: x[k] for k in ('p1', 'p2', 'start', 'tex_mid', 'fid', 'sex', 'level')}, A={k: x['A'][k] for k in ('win_p1', 'tot_med', 'tot_mean', 'games_p1', 'games_p2', 'tot_line_fair', 'hcp_p1_fair', 'p_straight')}, B=x['B'])) + '\n')
    nb = sum(x['B'].get('tier') == 'B' for x in out)
    print('tennis_ab.json', len(out), 'matches;', sum('edge' in x['B'] for x in out), 'priced;', nb, 'leans')

if __name__ == '__main__': main()
