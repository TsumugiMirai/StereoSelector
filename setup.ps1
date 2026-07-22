$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand -or $PythonCommand.Source -like "*WindowsApps*") {
    $InstalledPython = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $InstalledPython) {
        throw "Python 3.12 is required. Install it and run setup.ps1 again."
    }
    $PythonExe = $InstalledPython.FullName
} else {
    $PythonExe = $PythonCommand.Source
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $PythonExe -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e ".[build,test]"
Write-Host "Environment ready. Run: .\run.ps1"
