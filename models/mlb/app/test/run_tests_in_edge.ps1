# Runs test/tests.js in headless Microsoft Edge (for a computer without Node). Usage: powershell -File test\run_tests_in_edge.ps1
$root = Split-Path $PSScriptRoot -Parent
$work = Join-Path $env:TEMP 'tool_tests'; New-Item -ItemType Directory -Force $work | Out-Null; Push-Location $work
function B64($path) { [Convert]::ToBase64String([IO.File]::ReadAllBytes($path)) }
$mods = 'util','config','montecarlo','projection','data','game','odds','playlog'
$sb = New-Object System.Text.StringBuilder
[void]$sb.Append("var SRC = {};`n")
foreach ($m in $mods) { [void]$sb.Append("SRC['$m'] = '$(B64 "$root\src\$m.js")';`n") }
[void]$sb.Append("SRC['tests'] = '$(B64 "$root\test\tests.js")';`n")
[void]$sb.Append("var FIXB64 = {};`n")
foreach ($f in Get-ChildItem "$root\test\fixtures") { [void]$sb.Append("FIXB64['$($f.Name)'] = '$(B64 $f.FullName)';`n") }
$boot = @'
function dec(b64) { var bin = atob(b64), bytes = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i); return new TextDecoder('utf-8').decode(bytes); }
var FIXTURES = {}; Object.keys(FIXB64).forEach(function (k) { FIXTURES[k] = dec(FIXB64[k]); });
var registry = {};
function req(name) { var k = name.replace(/^.*\//, ''); if (!registry[k]) throw new Error('no module ' + k); return registry[k]; }
try {
  ['util','config','montecarlo','projection','data','game','odds','playlog'].forEach(function (m) {
    var mod = { exports: {} }; new Function('module', 'exports', 'require', dec(SRC[m]))(mod, mod.exports, req); registry[m] = mod.exports;
  });
  new Function('require', 'FIXTURES', dec(SRC['tests']))(req, FIXTURES);
  document.getElementById('out').textContent = JSON.stringify(globalThis.TEST_RESULTS);
} catch (e) { document.getElementById('out').textContent = 'HARNESS ERROR: ' + e.message + '\n' + e.stack; }
'@
$html = "<!doctype html><html><head><meta charset='utf-8'></head><body><pre id='out'>NOT RUN</pre><script>$($sb.ToString())$boot</script></body></html>"
Set-Content "$PWD\run_project_tests.html" $html -Encoding UTF8
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
Start-Process -FilePath $edge -ArgumentList @("--headless=new","--disable-gpu","--no-sandbox","--virtual-time-budget=180000","--dump-dom","file:///$($PWD.Path.Replace('\','/'))/run_project_tests.html") -RedirectStandardOutput "$PWD\dom4.txt" -RedirectStandardError "$PWD\dom4_err.txt" -Wait -NoNewWindow
$m = [regex]::Match((Get-Content "$PWD\dom4.txt" -Raw), '<pre id="out">(.*?)</pre>', 'Singleline')
$txt = [System.Net.WebUtility]::HtmlDecode($m.Groups[1].Value)
if ($txt.StartsWith('[')) {
  $r = $txt | ConvertFrom-Json
  foreach ($x in $r) { "{0} {1}{2}" -f ($(if ($x.ok) {'PASS'} else {'FAIL'})), $x.name, ($(if (-not $x.ok -or $x.detail -match 'end-to-end|Reds|windows') { "  -> $($x.detail)" } else { '' })) }
  "`n$(($r | Where-Object ok).Count)/$($r.Count) passed"
} else { $txt }
Pop-Location
