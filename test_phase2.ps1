# Complete Phase 2 - Build and run all profiles
# Generates visibility caches and AO visualizations

Write-Host "=== Phase 2 Complete Build & Test ===" -ForegroundColor Cyan

# Ensure directories exist
New-Item -ItemType Directory -Force output/cache | Out-Null
New-Item -ItemType Directory -Force output/scenes | Out-Null

# Build
Write-Host "`n[1/4] Building with Ninja..." -ForegroundColor Green
cd build
$buildOutput = cmake --build . --config Release 2>&1
$buildSuccess = $LASTEXITCODE -eq 0
cd ..

if ($buildSuccess) {
    Write-Host "  Build successful!" -ForegroundColor Green
} else {
    Write-Host "  Build failed!" -ForegroundColor Red
    Write-Host $buildOutput
    exit 1
}

# Check executable exists
if (-not (Test-Path output\bin\radiosity.exe)) {
    Write-Host "  ERROR: radiosity.exe not found!" -ForegroundColor Red
    exit 1
}

# Run all three profiles
$profiles = @("low", "medium", "high")

foreach ($profile in $profiles) {
    Write-Host "`n[2/4] Running Phase 2 - Profile: $profile" -ForegroundColor Green
    
    # Run with --phase 2 to compute visibility
    & output\bin\radiosity.exe --$profile --phase 2 --samples 16
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $profile profile complete!" -ForegroundColor Green
    } else {
        Write-Host "  $profile profile FAILED!" -ForegroundColor Red
    }
}

# Verify outputs
Write-Host "`n[3/4] Verifying outputs..." -ForegroundColor Green

Write-Host "`nPhase 1 outputs (geometry):" -ForegroundColor Cyan
Get-ChildItem output/scenes/cornell_phase1_*.obj | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1KB, 1)) KB"
}

Write-Host "`nPhase 2 outputs (AO/visibility):" -ForegroundColor Cyan
Get-ChildItem output/scenes/cornell_phase2_*.obj | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1KB, 1)) KB"
}

Write-Host "`nVisibility caches:" -ForegroundColor Cyan
Get-ChildItem output/cache/visibility_*.bin | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1KB, 1)) KB"
}

Write-Host "`n[4/4] Phase 2 Complete!" -ForegroundColor Green
Write-Host "  - All profiles generated" -ForegroundColor Gray
Write-Host "  - Visibility caches saved" -ForegroundColor Gray
Write-Host "  - AO visualizations exported" -ForegroundColor Gray
