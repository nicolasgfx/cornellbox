# test_ptx_rebuild.ps1 - Verify PTX auto-rebuild functionality
# This script tests that PTX files are automatically rebuilt when CUDA source changes

$ErrorActionPreference = "Stop"

Write-Host "`n=== Testing PTX Auto-Rebuild ===" -ForegroundColor Cyan

$ptxFile = "output/bin/optix_kernels.ptx"
$cudaFile = "src/visibility/optix_kernels.cu"

# Check files exist
if (-not (Test-Path $ptxFile)) {
    Write-Host "✗ PTX file not found. Build the project first with .\build.ps1" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $cudaFile)) {
    Write-Host "✗ CUDA source file not found at $cudaFile" -ForegroundColor Red
    exit 1
}

# Get initial timestamp
$beforeTimestamp = (Get-Item $ptxFile).LastWriteTime
Write-Host "PTX file timestamp before: $beforeTimestamp" -ForegroundColor Gray

# Wait a moment to ensure timestamp difference
Start-Sleep -Milliseconds 100

# Touch CUDA file to trigger rebuild
Write-Host "`nTouching CUDA source file..." -ForegroundColor Yellow
(Get-Item $cudaFile).LastWriteTime = Get-Date
Write-Host "  ✓ CUDA file modified" -ForegroundColor Green

# Rebuild
Write-Host "`nRebuilding project..." -ForegroundColor Yellow
& .\build.ps1 -Verbose:$false

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}

# Check if PTX was rebuilt
if (-not (Test-Path $ptxFile)) {
    Write-Host "✗ PTX file disappeared after rebuild!" -ForegroundColor Red
    exit 1
}

$afterTimestamp = (Get-Item $ptxFile).LastWriteTime
Write-Host "`nPTX file timestamp after: $afterTimestamp" -ForegroundColor Gray

# Compare timestamps
if ($afterTimestamp -gt $beforeTimestamp) {
    Write-Host "`n✓ PTX auto-rebuild works!" -ForegroundColor Green
    Write-Host "PTX was automatically rebuilt when CUDA source changed" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n✗ PTX was NOT rebuilt!" -ForegroundColor Red
    Write-Host "Timestamps are identical - dependency tracking failed" -ForegroundColor Red
    exit 1
}
