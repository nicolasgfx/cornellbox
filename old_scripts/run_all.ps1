# Run Phase 2 for all profiles
param([switch]$NoBuild)

if (-not $NoBuild) {
    Write-Host "Building..." -ForegroundColor Cyan
    cd build
    cmake --build . --config Release
    cd ..
}

Write-Host "`n=== Running Phase 2 for all profiles ===" -ForegroundColor Cyan

foreach ($profile in @("low", "medium", "high")) {
    Write-Host "`n--- Profile: $profile ---" -ForegroundColor Green
    & output\bin\radiosity.exe --$profile
}

Write-Host "`n=== Outputs ===" -ForegroundColor Cyan
Write-Host "`nPhase 1 (geometry):" -ForegroundColor Yellow
Get-ChildItem output/scenes/cornell_phase1_*.obj 2>$null | Format-Table Name, Length

Write-Host "Phase 2 (AO):" -ForegroundColor Yellow  
Get-ChildItem output/scenes/cornell_phase2_*.obj 2>$null | Format-Table Name, Length

Write-Host "Visibility caches:" -ForegroundColor Yellow
Get-ChildItem output/cache/visibility_*.bin 2>$null | Format-Table Name, Length
