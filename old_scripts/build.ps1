param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Building Radiosity Renderer ===" -ForegroundColor Cyan
if (-not (Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
Push-Location build
cmake .. -DCMAKE_BUILD_TYPE=$Config
cmake --build . --config $Config
Pop-Location
Write-Host "Build complete. Output: output/bin/radiosity.exe" -ForegroundColor Green
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Host "  ✓ CMake configuration complete" -ForegroundColor Green
    
    # Build
    Write-Host "`nBuilding radiosity ($Config)..." -ForegroundColor Cyan
    $verbosity = if ($Verbose) { "detailed" } else { "minimal" }
    
    $msbuildPath = 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'
    if (-not (Test-Path $msbuildPath)) {
        # Try to find MSBuild via vswhere
        $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vsWhere) {
            $vsPath = & $vsWhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
            $msbuildPath = Join-Path $vsPath "MSBuild\Current\Bin\MSBuild.exe"
        }
    }
    
    if (-not (Test-Path $msbuildPath)) {
        throw "MSBuild not found. Please install Visual Studio 2022."
    }
    
    & $msbuildPath `
        radiosity.vcxproj `
        /p:Configuration=$Config `
        /v:$verbosity `
        /nologo `
        /m
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    
    Write-Host "`n✓ Build successful!" -ForegroundColor Green
    Write-Host "Executable: " -NoNewline -ForegroundColor Gray
    Write-Host "output/bin/radiosity.exe" -ForegroundColor White
    Write-Host "PTX kernels: " -NoNewline -ForegroundColor Gray
    Write-Host "output/bin/optix_kernels.ptx" -ForegroundColor White

} catch {
    Write-Host "`n✗ Build failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

Write-Host "`nBuild complete! Run with: " -NoNewline -ForegroundColor Gray
Write-Host ".\run.ps1" -ForegroundColor Cyan
Write-Host ""
