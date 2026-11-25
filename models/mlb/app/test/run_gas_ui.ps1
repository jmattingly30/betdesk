# Runs the real page (apps_script/Index.html) against the Apps Script server on fake Google services, clicks
# through it (test/gas_ui_driver.js), and saves a 375px-wide (phone) screenshot of the result.
param([string]$shot = (Join-Path $env:TEMP 'gas_test\gas_ui_phone.png'))
$root = Split-Path $PSScriptRoot -Parent
powershell -NoProfile -File (Join-Path $root 'build_apps_script.ps1') | Out-Null
$work = Join-Path $env:TEMP 'gas_test'; New-Item -ItemType Directory -Force $work | Out-Null
function B64($p) { [Convert]::ToBase64String([IO.File]::ReadAllBytes($p)) }
$head = New-Object System.Text.StringBuilder
[void]$head.Append("<script>function dec(b){var bin=atob(b),u=new Uint8Array(bin.length);for(var i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new TextDecoder('utf-8').decode(u);} var FIXTURES={};")
foreach ($f in Get-ChildItem (Join-Path $root 'test\fixtures')) { [void]$head.Append("FIXTURES['$($f.Name)']=dec('$(B64 $f.FullName)');") }
[void]$head.Append("</script>")
foreach ($src in 'test\gas_stubs.js', 'apps_script\Bundle.gs', 'apps_script\Server.gs', 'test\gas_driver.js') { [void]$head.Append("<script>`n$([IO.File]::ReadAllText((Join-Path $root $src)))`n</script>") }
# google.script.run stand-in: a fresh runner per use (like the real one), async, JSON round trip both ways
[void]$head.Append(@'
<script>
var SERVER_API = api;
var google = { script: {} };
Object.defineProperty(google.script, 'run', { get: function () {
  var ok = function () {}, fail = function () {}, r = {
    withSuccessHandler: function (f) { ok = f; return r; }, withFailureHandler: function (f) { fail = f; return r; },
    api: function (name, payload) { var args = JSON.parse(JSON.stringify([name, payload]));
      setTimeout(function () { try { ok(SERVER_API(args[0], args[1])); } catch (e) { fail(e); } }, 5); } };
  return r; } });
</script>
'@)
$index = [IO.File]::ReadAllText((Join-Path $root 'apps_script\Index.html'))
$i = $index.IndexOf('<script>')
$pageHtml = $index.Substring(0, $i) + $head.ToString() + $index.Substring($i).Replace('</body>', "<script>`n$([IO.File]::ReadAllText((Join-Path $root 'test\gas_ui_driver.js')))`n</script></body>")
[IO.File]::WriteAllText((Join-Path $work 'ui.html'), $pageHtml, [Text.UTF8Encoding]::new($false))
$frame = "<body style='margin:0'><iframe id='f' src='ui.html' style='width:375px;height:2600px;border:0'></iframe><script>(function w(){var d=document.getElementById('f').contentWindow;if(d&&d.UI_RESULTS){document.title='DONE';document.body.setAttribute('data-r',JSON.stringify(d.UI_RESULTS.concat(d.GAS_RESULTS)));}else setTimeout(w,100);})();</script></body>"
[IO.File]::WriteAllText((Join-Path $work 'frame.html'), $frame, [Text.UTF8Encoding]::new($false))
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'; $u = "file:///$((Join-Path $work 'frame.html').Replace('\','/'))"
Start-Process -FilePath $edge -ArgumentList @('--headless=new', '--disable-gpu', '--allow-file-access-from-files', '--window-size=500,2600', '--virtual-time-budget=60000', '--dump-dom', $u) -RedirectStandardOutput (Join-Path $work 'ui_dom.txt') -RedirectStandardError (Join-Path $work 'ui_err.txt') -Wait -NoNewWindow
Start-Process -FilePath $edge -ArgumentList @('--headless=new', '--disable-gpu', '--allow-file-access-from-files', '--window-size=500,2600', '--hide-scrollbars', '--virtual-time-budget=60000', "--screenshot=$shot", $u) -RedirectStandardError (Join-Path $work 'ui_err2.txt') -Wait -NoNewWindow
$m = [regex]::Match([IO.File]::ReadAllText((Join-Path $work 'ui_dom.txt')), "data-r=`"([^`"]*)`"")
if (-not $m.Success) { 'UI harness produced no results'; return }
$r = [Net.WebUtility]::HtmlDecode($m.Groups[1].Value) | ConvertFrom-Json
foreach ($x in $r) { "{0} {1}{2}" -f $(if ($x.ok) { 'PASS' } else { 'FAIL' }), $x.name, $(if (-not $x.ok) { "  -> $($x.detail)" } else { '' }) }
"`n$(@($r | Where-Object ok).Count)/$(@($r).Count) passed; screenshot: $shot"
