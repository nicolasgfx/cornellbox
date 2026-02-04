#!/usr/bin/env pwsh
# Complete Phase 2 Demo - Builds and runs all profiles

$ErrorActionPreference = "Stop"

Write-Host @"
╔═══════════════════════════════════════════════╗
║   Radiosity Cornell Box - Phase 2 Complete   ║
║   OptiX Visibility + AO Visualization        ║
╚═══════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Step 1: Build
Write-Host "`n[1/4] Building with Ninja..." -ForegroundColor Green
if (-not (Test-Path build/CMakeCache.txt)) {
    Write-Host "  First-time setup..." -ForegroundColor Yellow
    & .\fast_build.ps1 -Clean
} else {
    Write-Host "  Incremental build..." -ForegroundColor Yellow
    & .\fast_build.ps1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Build failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Verify OptiX
Write-Host "`n[2/4] Checking OptiX..." -ForegroundColor Green
if (Test-Path build/optix_kernels.ptx) {
    $ptxSize = (Get-Item build/optix_kernels.ptx).Length
    Write-Host "  ✓ PTX compiled ($ptxSize bytes)" -ForegroundColor Gray
} else {
    Write-Host "  ✗ PTX missing - OptiX may not be enabled" -ForegroundColor Yellow
}

# Step 3: Run all profiles
Write-Host "`n[3/4] Running Phase 2 for all profiles..." -ForegroundColor Green

$profiles = @(
    @{name="low"; desc="~800 triangles, fast"},
    @{name="medium"; desc="~1000 triangles, balanced"},
    @{name="high"; desc="~4000 triangles, quality"}
)

foreach ($p in $profiles) {
    Write-Host "`n  Profile: $($p.name) ($($p.desc))" -ForegroundColor Cyan
    & output\bin\radiosity.exe --$($p.name) 2>&1 | Select-String "Profile|Phase|Complete|triangles|Output"
}

# Step 4: Verify outputs
Write-Host "`n[4/4] Verifying outputs..." -ForegroundColor Green

$phase1Files = Get-ChildItem output/scenes/cornell_phase1_*.obj -ErrorAction SilentlyContinue
$phase2Files = Get-ChildItem output/scenes/cornell_phase2_*.obj -ErrorAction SilentlyContinue
$cacheFiles = Get-ChildItem output/cache/visibility_*.bin -ErrorAction SilentlyContinue

Write-Host "`n  Phase 1 (geometry): " -NoNewline
Write-Host "$($phase1Files.Count)/3 profiles" -ForegroundColor $(if($phase1Files.Count -eq 3){"Green"}else{"Yellow"})

Write-Host "  Phase 2 (AO vis):   " -NoNewline
Write-Host "$($phase2Files.Count)/3 profiles" -ForegroundColor $(if($phase2Files.Count -eq 3){"Green"}else{"Yellow"})

Write-Host "  Visibility cache:   " -NoNewline
Write-Host "$($cacheFiles.Count)/3 profiles" -ForegroundColor $(if($cacheFiles.Count -eq 3){"Green"}else{"Yellow"})

if ($phase1Files.Count -eq 3 -and $phase2Files.Count -eq 3 -and $cacheFiles.Count -eq 3) {
    Write-Host "`n✓ Phase 2 Complete - All outputs generated!" -ForegroundColor Green
    
    Write-Host "`nOutputs:" -ForegroundColor Cyan
    Write-Host "  Geometry:    output/scenes/cornell_phase1_*.obj" -ForegroundColor Gray
    Write-Host "  AO Vis:      output/scenes/cornell_phase2_*.obj" -ForegroundColor Gray
    Write-Host "  Vis Cache:   output/cache/visibility_*.bin" -ForegroundColor Gray
} else {
    Write-Host "`n⚠ Some outputs missing - check logs above" -ForegroundColor Yellow
}

Write-Host ""
