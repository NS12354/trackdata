# Kill the API server processes ONLY (uvicorn + its spawned workers).
# Explicitly spares the Celery worker and anything outside this project.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -notlike '*celery*' -and
    ($_.CommandLine -like '*uvicorn*main:app*' -or
     ($_.ExecutablePath -like '*AI_Training\backend\.venv*' -and $_.CommandLine -like '*multiprocessing*'))
}
foreach ($p in $procs) {
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Output "killed $($p.ProcessId) :: $($p.CommandLine.Substring(0, [Math]::Min(70, $p.CommandLine.Length)))" } catch {}
}
Start-Sleep -Seconds 2
$left = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($left) { Write-Output "STILL LISTENING: pid $($left.OwningProcess)" } else { Write-Output "port 8000 free" }
