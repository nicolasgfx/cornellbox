# Build System Revision Plan

## Critical Issues Identified

### 1. **PTX File Build Issues (SUPERCRITICAL)**
- **Problem**: PTX files not automatically rebuilt when CUDA source changes
- **Current Behavior**: `optix_kernels.ptx` becomes stale, causing crashes
- **Manual Workaround Required**: Copy from `build/` to `build/Release/` after rebuild
- **Root Cause**: CMake custom command doesn't properly track dependencies

### 2. **File Location Chaos**
- **Output Files Scattered**: 
  - `.obj/.mtl` files in: `build/`, `build/Release/`, root directory
  - `.cache` files in: root, `build/`, `build/Release/`
  - `.ptx` files duplicated in: `build/` and `build/Release/`
- **Test Output Files**: 20+ `.txt` log files cluttering `build/Release/`
- **No Clear Output Directory Structure**

### 3. **Build Directory Bloat**
- **30+ files in /build root** (should have ~5)
- **Obsolete files**: `test_visibility_from_cache.vcxproj` still exists
- **CMake generates excessive intermediate files**
- **No `.gitignore` for build artifacts**

### 4. **Slow Build Times**
- **Full rebuild required** when PTX changes (doesn't trigger automatically)
- **No incremental PTX compilation**
- **MSBuild verbosity generates excessive output**

---

## Proposed Solution: Complete Build Restructuring

### Phase 1: PTX Build Pipeline Fix (CRITICAL - DO FIRST)

#### 1.1 Move PTX Generation to Pre-Build Step
```cmake
# CMakeLists.txt changes:

# Define PTX output to same location as executable
set(PTX_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optix_kernels.ptx")

# Make PTX compilation a PROPER dependency
add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${NVCC_EXECUTABLE} 
        -ptx 
        -arch=sm_50
        -use_fast_math
        -I"${OptiX_INCLUDE}"
        -I"${CMAKE_SOURCE_DIR}/src"
        ${CMAKE_SOURCE_DIR}/src/visibility/optix_kernels.cu
        -o ${PTX_OUTPUT}
    DEPENDS ${CMAKE_SOURCE_DIR}/src/visibility/optix_kernels.cu
    COMMENT "Compiling OptiX kernels to PTX"
    VERBATIM
)

# Create custom target that radiosity.exe DEPENDS on
add_custom_target(compile_ptx_auto 
    DEPENDS ${PTX_OUTPUT}
)

# Make radiosity depend on PTX being built
add_dependencies(radiosity compile_ptx_auto)
```

**Result**: PTX automatically rebuilds BEFORE radiosity.exe links, in correct location.

#### 1.2 Remove Separate compile_ptx Project
- Delete standalone `compile_ptx` target (no longer needed)
- PTX now part of main build dependency chain
- No manual copying required

---

### Phase 2: Directory Restructuring

#### 2.1 Create Structured Output Directories
```
radiosity/
├── build/               (CMake-generated only, mostly hidden)
│   ├── CMakeFiles/      (generated, ignored)
│   ├── CMakeCache.txt   (generated, ignored)
│   └── RadiosityRenderer.sln  (VS solution)
│
├── output/              (NEW: All program outputs go here)
│   ├── bin/             (Executables + PTX)
│   │   ├── radiosity.exe
│   │   └── optix_kernels.ptx
│   ├── scenes/          (Generated .obj/.mtl files)
│   │   ├── cornell_box.obj
│   │   ├── cornell_box.mtl
│   │   └── cornell_box_radiosity.obj
│   ├── cache/           (Visibility cache files)
│   │   └── cornell_box_14x7_visibility.cache
│   └── logs/            (Test/debug output)
│       └── radiosity_run.txt
│
└── src/                 (Source files only)
```

#### 2.2 CMakeLists.txt Output Path Configuration
```cmake
# Set all output to output/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/output/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/output/bin)

# PTX to same location as executable
set(PTX_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optix_kernels.ptx)

# Copy necessary runtime files
file(COPY ${CMAKE_SOURCE_DIR}/cornell_box.mtl 
     DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
```

#### 2.3 Update main.cpp Output Paths
```cpp
// Use relative paths from executable location
const std::string OUTPUT_DIR = "../output/scenes/";
const std::string CACHE_DIR = "../output/cache/";
const std::string LOG_DIR = "../output/logs/";

// In exportRadiositySolution():
std::string objFilename = OUTPUT_DIR + "cornell_box_radiosity.obj";
std::string cacheFilename = CACHE_DIR + "cornell_box_14x7_visibility.cache";
```

---

### Phase 3: Build System Cleanup

#### 3.1 Reduce CMake-Generated Files
```cmake
# Use out-of-source build pattern more strictly
# In CMakeLists.txt:
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds not allowed. Create 'build' dir.")
endif()

# Minimize intermediate files
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)  # Don't generate compile_commands.json
set_target_properties(radiosity PROPERTIES 
    INTERMEDIATE_DIR "${CMAKE_BINARY_DIR}/temp"
)
```

#### 3.2 Create Comprehensive .gitignore
```gitignore
# Build artifacts
build/
output/bin/
output/cache/
output/logs/

# Keep scene examples but ignore generated
output/scenes/*.obj
output/scenes/*.mtl
!output/scenes/README.md

# VS/CMake generated
*.vcxproj
*.vcxproj.filters
*.sln
*.user
CMakeCache.txt
CMakeFiles/
cmake_install.cmake

# Runtime outputs
*.exe
*.ptx
*.cache

# Test logs
*.txt
!CMakeLists.txt
!README.txt
```

#### 3.3 Remove Obsolete Files
**Manual cleanup before restructure:**
```bash
# Delete all scattered output files
rm build/*.obj build/*.mtl build/*.txt
rm build/Release/*.txt  # Keep only .exe and .ptx
rm *.cache  # From root

# Remove obsolete project files
rm build/test_visibility_from_cache.*
```

---

### Phase 4: Build Script Improvements

#### 4.1 Create build.ps1 (Improved)
```powershell
# build.ps1 - Unified build script
param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release",
    
    [switch]$Clean,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path output/bin, output/scenes, output/cache, output/logs | Out-Null

if ($Clean) {
    Write-Host "Cleaning build..." -ForegroundColor Yellow
    Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory build | Out-Null
}

# Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Cyan
Push-Location build
cmake .. -DCMAKE_BUILD_TYPE=$Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Build
Write-Host "Building radiosity ($Config)..." -ForegroundColor Cyan
$verbosity = if ($Verbose) { "detailed" } else { "minimal" }
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' `
    radiosity.vcxproj `
    /p:Configuration=$Config `
    /v:$verbosity `
    /nologo

Pop-Location

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Build successful!" -ForegroundColor Green
    Write-Host "Executable: output/bin/radiosity.exe" -ForegroundColor Gray
} else {
    Write-Host "`n✗ Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
```

#### 4.2 Create run.ps1 (Execution Script)
```powershell
# run.ps1 - Run radiosity with proper paths
param(
    [switch]$NoCache,
    [switch]$Verbose
)

$exe = "output/bin/radiosity.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Error: radiosity.exe not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

# Change to output/bin so PTX is found
Push-Location output/bin

if ($NoCache) {
    Remove-Item ../cache/*.cache -ErrorAction SilentlyContinue
    Write-Host "Cache cleared" -ForegroundColor Yellow
}

Write-Host "Running radiosity..." -ForegroundColor Cyan
if ($Verbose) {
    & ./radiosity.exe
} else {
    & ./radiosity.exe | Select-String -Pattern "(STAGE|Visibility|complete|ERROR)"
}

Pop-Location
```

---

### Phase 5: Testing & Validation

#### 5.1 Verify PTX Auto-Rebuild
```powershell
# Test script: test_ptx_rebuild.ps1
# Modify optix_kernels.cu, rebuild, verify PTX timestamp updates
$ptx = "output/bin/optix_kernels.ptx"
$before = (Get-Item $ptx).LastWriteTime

# Touch CUDA file
(Get-Item src/visibility/optix_kernels.cu).LastWriteTime = Get-Date

# Rebuild
& .\build.ps1

$after = (Get-Item $ptx).LastWriteTime
if ($after -gt $before) {
    Write-Host "✓ PTX auto-rebuild works!" -ForegroundColor Green
} else {
    Write-Host "✗ PTX not rebuilt!" -ForegroundColor Red
}
```

#### 5.2 Verify Output Locations
```powershell
# Check all outputs in correct locations
@(
    "output/bin/radiosity.exe",
    "output/bin/optix_kernels.ptx",
    "output/scenes/cornell_box_radiosity.obj",
    "output/cache/cornell_box_14x7_visibility.cache"
) | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "✓ $_" -ForegroundColor Green
    } else {
        Write-Host "✗ $_ MISSING" -ForegroundColor Red
    }
}
```

---

## Implementation Steps (Execute in Order)

### Step 1: Backup Current State
```bash
git commit -am "Pre-build-revision snapshot"
git tag pre-build-revision
```

### Step 2: Create Output Directory Structure
```bash
mkdir -p output/{bin,scenes,cache,logs}
echo "# Generated Scenes" > output/scenes/README.md
echo "# Visibility Cache" > output/cache/README.md
```

### Step 3: Update CMakeLists.txt
- [ ] Set `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `output/bin`
- [ ] Fix PTX generation to use `add_custom_command` with `OUTPUT`
- [ ] Add `add_dependencies(radiosity compile_ptx_auto)`
- [ ] Remove standalone `compile_ptx` target
- [ ] Add in-source build check

### Step 4: Update main.cpp Output Paths
- [ ] Define `OUTPUT_DIR`, `CACHE_DIR`, `LOG_DIR` constants
- [ ] Update all `OBJWriter::write*()` calls to use `OUTPUT_DIR`
- [ ] Update all `DEBUG_CACHE_FILENAME` to use `CACHE_DIR`
- [ ] Ensure working directory is `output/bin` when running

### Step 5: Create Build Scripts
- [ ] Write improved `build.ps1`
- [ ] Write `run.ps1` execution script
- [ ] Write `clean.ps1` to wipe build and output
- [ ] Update existing scripts or delete obsolete ones

### Step 6: Create .gitignore
- [ ] Add comprehensive ignore rules
- [ ] Keep directory structure but ignore generated files
- [ ] Commit `.gitignore` before cleanup

### Step 7: Clean Up Existing Mess
```bash
# Manual cleanup (do AFTER committing changes)
rm build/*.obj build/*.mtl build/*.txt
rm build/Release/*.txt
rm *.cache cornell_box*.obj cornell_box*.mtl
rm -rf build/test_visibility_from_cache.*
```

### Step 8: Test Build
```bash
.\clean.ps1
.\build.ps1 -Clean
.\run.ps1
```

### Step 9: Verify All Tests Pass
```bash
# Run full test suite
.\run.ps1 -Verbose

# Verify PTX auto-rebuild
.\test_ptx_rebuild.ps1

# Check output locations
Get-ChildItem output -Recurse
```

### Step 10: Update Documentation
- [ ] Update README.md with new build instructions
- [ ] Document new directory structure
- [ ] Add "Building" section to main README
- [ ] Remove outdated build notes

---

## Expected Outcomes

### Build Directory (Minimal)
```
build/
├── CMakeCache.txt          (CMake config)
├── cmake_install.cmake     (CMake installer)
├── CMakeFiles/             (CMake internals - hidden in .gitignore)
├── RadiosityRenderer.sln   (VS solution)
├── radiosity.vcxproj       (VS project)
├── radiosity.dir/          (VS intermediate files)
└── ZERO_CHECK.vcxproj      (CMake helper)
```
**File count: ~7 visible files (down from 30+)**

### Output Directory (All Program Outputs)
```
output/
├── bin/
│   ├── radiosity.exe       (Executable)
│   └── optix_kernels.ptx   (GPU kernel)
├── scenes/
│   ├── cornell_box.obj     (Default scene)
│   ├── cornell_box_radiosity.obj  (Solved radiosity)
│   └── *.mtl               (Material files)
├── cache/
│   └── cornell_box_14x7_visibility.cache  (Visibility matrix)
└── logs/
    └── radiosity_run.txt   (Test output)
```

### Build Process Improvements
- ✅ **PTX auto-rebuilds** when CUDA source changes
- ✅ **No manual copying** required
- ✅ **Single command build**: `.\build.ps1`
- ✅ **Clean output structure**: All generated files in `output/`
- ✅ **Fast incremental builds**: Only rebuild what changed
- ✅ **Clear separation**: Source vs Build vs Output

### Developer Experience
- **Before**: 
  - Manual PTX copy after every CUDA change
  - Files scattered across 4 locations
  - 30+ files in build directory
  - Unclear what's generated vs source
  
- **After**:
  - Automatic PTX rebuild
  - All outputs in `output/` tree
  - ~7 files in build directory
  - Clear directory structure
  - One-command build and run

---

## Risk Assessment

### Low Risk
- Creating new `output/` directory structure
- Adding `.gitignore`
- Improving build scripts

### Medium Risk
- Changing CMake output paths (test thoroughly)
- Updating main.cpp file paths (verify all tests)

### High Risk
- PTX build pipeline changes (critical path)
  - **Mitigation**: Test on isolated branch first
  - **Rollback**: Keep `git tag pre-build-revision`
  - **Validation**: Run `test_ptx_rebuild.ps1` after changes

---

## Timeline Estimate

- **Phase 1** (PTX Fix): 1 hour
- **Phase 2** (Directory Structure): 30 minutes  
- **Phase 3** (Build Cleanup): 30 minutes
- **Phase 4** (Build Scripts): 45 minutes
- **Phase 5** (Testing): 1 hour

**Total: ~4 hours** for complete restructure

---

## Success Criteria

1. ✅ PTX automatically rebuilds when `optix_kernels.cu` changes
2. ✅ PTX output to same directory as executable (no copying)
3. ✅ All program outputs in `output/` tree
4. ✅ Build directory has <10 visible files
5. ✅ Single-command build: `.\build.ps1`
6. ✅ Single-command run: `.\run.ps1`
7. ✅ All existing tests pass after restructure
8. ✅ Clean `git status` (no untracked build artifacts)

---

## Notes for Implementation Session

### Critical Order Dependencies
1. **DO FIRST**: Commit current state and tag
2. **DO SECOND**: PTX build fix (most critical)
3. **DO THIRD**: Directory restructure
4. **DO LAST**: Cleanup old files

### Test After Each Phase
- After PTX fix: Modify CUDA file, rebuild, verify auto-rebuild
- After directory restructure: Run full test suite
- After cleanup: Fresh clone and build from scratch

### Common Pitfalls
- Don't clean up files before committing changes
- Don't change working directory assumptions without updating all code
- Test PTX loading from correct location before deleting duplicates
- Verify `.gitignore` doesn't accidentally ignore source files

### Rollback Plan
```bash
# If something breaks:
git reset --hard pre-build-revision
git clean -fdx build/ output/
```

---

## Future Enhancements (Post-Restructure)

1. **CMake Presets** for common build configurations
2. **CTest Integration** for automated testing
3. **CPack Configuration** for distribution packaging
4. **CI/CD Pipeline** (GitHub Actions) for automatic builds
5. **Build Time Profiling** to identify slow compilation units
6. **Unity Build** option to speed up full rebuilds

---

*This plan addresses all identified build system issues with minimal risk and maximum impact on developer productivity.*
