param(
    [ValidateSet("low", "medium", "high")]
    [string]$Profile = "low"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Radiosity Renderer - Phase 1 ===" -ForegroundColor Cyan
Write-Host "Profile: $Profile" -ForegroundColor Yellow
Write-Host ""

$exe = "output/bin/radiosity.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Error: radiosity.exe not found. Please build first with build.ps1" -ForegroundColor Red
    exit 1
}

# Run the renderer
& $exe --$Profile

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nExecution complete!" -ForegroundColor Green
    Write-Host "Output files: output\scenes\cornell_phase1_$Profile.obj" -ForegroundColor Yellow
} else {
    Write-Host "`nExecution failed with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
