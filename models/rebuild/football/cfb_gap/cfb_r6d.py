"""Totals: team season-to-date over/under vs v3 total (shrunk), using every v3-priced game (lined or not)."""
exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
ALL = pd.concat([fit.assign(src='f'), te.assign(src='t')], ignore_index=True)
ALL["dt"] = pd.to_datetime(ALL.date, utc=True); ALL["tres"] = ALL["at"] - ALL.model_total
long = pd.concat([ALL[["season", "dt", "game_id", "home_id", "tres"]].rename(columns={"home_id": "t"}), ALL[["season", "dt", "game_id", "away_id", "tres"]].rename(columns={"away_id": "t"})]).sort_values("dt")
g = long.groupby(["season", "t"]); long["n"] = g.cumcount(); long["cs"] = g.tres.cumsum() - long.tres
R6 = long.set_index(["game_id", "t"])
for K in (4, 8):
    for d in (fit, te):
        for side, col in (("h", "home_id"), ("a", "away_id")):
            r = R6.reindex(list(zip(d.game_id.values, d[col].values)))
            d["tr_" + side] = (r.cs / (r.n + K)).fillna(0).values
        d["x_tr"] = d.tr_h + d.tr_a
    T1 = run(['model_total', 'x_ip'], y='at'); p = run(['model_total', 'x_ip', 'x_tr'], y='at')
    rep(f"tempo total + over/under form K={K}", p, y='at', base=te.model_total.values)
    d_ = np.abs(p - te["at"].values) - np.abs(T1 - te["at"].values); print(f"      vs tempo total {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
print("note: only lined games carry v3 totals here, so form is from lined games only")
