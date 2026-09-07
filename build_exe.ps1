$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    # Refresh distribution metadata, which is embedded in the frozen app.
    & $PythonExe -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to refresh the project metadata."
    }
    # The output name is derived from pyproject.toml by StereoSelector.spec;
    # ask the package for the same value so this script never drifts.
    $AppName = & $PythonExe -c "import sys; sys.path.insert(0, 'src'); from stereo_selector import release_version; print(f'StereoSelector-v{release_version()}')"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($AppName)) {
        throw "Unable to determine the application version."
    }
    & $PythonExe -m PyInstaller --noconfirm --clean .\StereoSelector.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
    $OutputExe = Join-Path $ProjectRoot "dist\$AppName.exe"
    if (-not (Test-Path -LiteralPath $OutputExe)) {
        throw "Build finished without the expected output: $OutputExe"
    }
    Write-Host "Build complete: $OutputExe"
} finally {
    Pop-Location
}
