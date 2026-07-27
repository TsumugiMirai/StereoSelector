$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $PythonExe -m PyInstaller --noconfirm --clean .\StereoSelector.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
    $OutputExe = Join-Path $ProjectRoot "dist\StereoSelector-v1.1.exe"
    if (-not (Test-Path -LiteralPath $OutputExe)) {
        throw "Build finished without the expected output: $OutputExe"
    }
    Write-Host "Build complete: $OutputExe"
} finally {
    Pop-Location
}
