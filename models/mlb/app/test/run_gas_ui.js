// Node + Playwright version of run_gas_ui.ps1: the real page against the Apps Script server on fake Google services,
// clicked through by test/gas_ui_driver.js at phone width (375px). Usage: node test/run_gas_ui.js [screenshot.png]
const fs = require('fs'), path = require('path'), os = require('os');
const root = path.join(__dirname, '..');
require('child_process').execFileSync(process.execPath, [path.join(root, 'build_apps_script.js')]);
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'gas_ui_'));
let head = "<script>var FIXTURES={};";
for (const f of fs.readdirSync(path.join(__dirname, 'fixtures'))) head += `FIXTURES[${JSON.stringify(f)}]=${JSON.stringify(fs.readFileSync(path.join(__dirname, 'fixtures', f), 'utf8').replace(/^﻿+/, '')).replace(/<\/script/gi, '<\\/script')};`;
head += '</script>';
for (const src of ['test/gas_stubs.js', 'apps_script/Bundle.gs', 'apps_script/Server.gs', 'test/gas_driver.js']) head += `<script>\n${fs.readFileSync(path.join(root, src), 'utf8')}\n</script>`;
head += `<script>
var SERVER_API = api;
var google = { script: {} };
Object.defineProperty(google.script, 'run', { get: function () {
  var ok = function () {}, fail = function () {}, r = {
    withSuccessHandler: function (f) { ok = f; return r; }, withFailureHandler: function (f) { fail = f; return r; },
    api: function (name, payload) { var args = JSON.parse(JSON.stringify([name, payload]));
      setTimeout(function () { try { ok(SERVER_API(args[0], args[1])); } catch (e) { fail(e); } }, 5); } };
  return r; } });
</script>`;
const index = fs.readFileSync(path.join(root, 'apps_script', 'Index.html'), 'utf8'), i = index.indexOf('<script>');
const page = index.slice(0, i) + head + index.slice(i).replace('</body>', `<script>\n${fs.readFileSync(path.join(__dirname, 'gas_ui_driver.js'), 'utf8')}\n</script></body>`);
fs.writeFileSync(path.join(work, 'ui.html'), page);
(async () => {
  const { chromium } = require(path.join(require('child_process').execSync('npm root -g').toString().trim(), 'playwright'));
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 375, height: 900 } });
  const errors = []; p.on('pageerror', e => errors.push(e.message));
  await p.goto('file://' + path.join(work, 'ui.html'));
  await p.waitForFunction(() => window.UI_RESULTS, null, { timeout: 120000 });
  const r = await p.evaluate(() => window.UI_RESULTS.concat(window.GAS_RESULTS || []));
  const shot = process.argv[2] || path.join(work, 'gas_ui_phone.png'); await p.screenshot({ path: shot, fullPage: true });
  await b.close();
  for (const x of r) console.log((x.ok ? 'PASS ' : 'FAIL ') + x.name + (x.ok ? '' : '  -> ' + x.detail));
  if (errors.length) console.log('page errors: ' + errors.join(' | '));
  console.log(`\n${r.filter(x => x.ok).length}/${r.length} passed; screenshot: ${shot}`);
  process.exit(r.every(x => x.ok) && !errors.length ? 0 : 1);
})();
