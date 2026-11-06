// Audit for Watch the sim: every stored replay, played back by the viewer's own adapters (app/sim.js), must end on the final score
// (tennis: the set scores) that models/sims/build.py recorded, with every event inside the game clock.
// Usage: node models/sims/test_sim.js [app/data dir]    (exit 1 on any mismatch)
const path = require('path'), fs = require('fs');
const {ADAPT} = require(path.join(__dirname, '../../app/sim.js'));
const dir = process.argv[2] || path.join(__dirname, '../../app/data');
let bad = 0, n = 0, ev = 0;
for (const sp of ['nba', 'cbb', 'nfl', 'cfb', 'tennis']){
  const f = path.join(dir, `sims_${sp}.json`); if (!fs.existsSync(f)){ console.log(sp, 'no file'); continue; }
  const F = JSON.parse(fs.readFileSync(f)); let k = 0;
  for (const g of Object.values(F.games)) for (const s of g.s){
    const r = ADAPT[F.unit](F, s, [g.h, g.a]); n++; k++; ev += r.ev.length;
    const want = JSON.stringify(sp === 'tennis' ? s.sets : s.f), got = JSON.stringify(sp === 'tennis' ? r.sets : r.fin);
    if (want !== got){ bad++; if (bad < 6) console.log('MISMATCH', sp, g.a, '@', g.h, 'stored', want, 'replayed', got); }
    if (r.ev.some(e => e.t > r.T + 1e-6 || e.t < 0)){ bad++; console.log('event outside the game clock', sp, g.a, '@', g.h); }
  }
  console.log(sp, k, 'replays,', Object.keys(F.games).length, 'games,', Math.round(fs.statSync(f).size / 1024), 'KB');
}
console.log(n, 'replays,', bad, 'problems, about', Math.round(ev / Math.max(n, 1)), 'events each'); process.exit(bad ? 1 : 0);
