# FAST build - uses Ninja (10x faster than Visual Studio)
param([switch]$Clean, [switch]$Fresh)

if ($Clean -and (Test-Path build)) { Remove-Item -Recurse -Force build }
if (-not (Test-Path build)) { mkdir build | Out-Null }

cd build

# Only configure if needed
if ((-not (Test-Path CMakeCache.txt)) -or $Fresh) {
    Write-Host "Configuring..." -ForegroundColor Green
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
}

# Build (Ninja is parallel by default)
Write-Host "Building..." -ForegroundColor Green
$t = Measure-Command { cmake --build . }
Write-Host "Done in $($t.TotalSeconds.ToString('0.0'))s" -ForegroundColor Cyan

cd ..
