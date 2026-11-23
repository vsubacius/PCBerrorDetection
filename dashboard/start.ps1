$ErrorActionPreference = 'Stop'
$dashboardPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $dashboardPython)) {
    Write-Host 'First-time setup: py -3 -m venv dashboard/.venv'
    Write-Host 'Then: dashboard/.venv/Scripts/python.exe -m pip install -r dashboard/requirements.txt'
    exit 1
}
& $dashboardPython (Join-Path $PSScriptRoot 'server.py')
