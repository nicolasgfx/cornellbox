@echo off
REM Run Phase 2 for all profiles sequentially

setlocal enabledelayedexpansion

echo.
echo === Building... ===
cd build
cmake --build . --config Release
if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)
cd ..

echo.
echo === Running Phase 2 for all profiles ===
echo.

echo --- Profile: low ---
output\bin\radiosity.exe --low --nocache --phase 3
if errorlevel 1 echo Warning: low profile failed

echo.
echo --- Profile: medium ---
output\bin\radiosity.exe --medium --nocache --phase 3
if errorlevel 1 echo Warning: medium profile failed

echo.
echo --- Profile: high ---
output\bin\radiosity.exe --high --nocache --phase 3
if errorlevel 1 echo Warning: high profile failed

echo.
echo === Outputs ===
echo.
echo Phase 1 (geometry):
dir /B output\scenes\cornell_phase1_*.obj 2>nul
echo.
echo Phase 2 (AO):
dir /B output\scenes\cornell_phase2_*.obj 2>nul
echo.
echo Phase 3 (Radiosity):
dir /B output\scenes\cornell_phase3_*.obj 2>nul
echo.
echo Visibility caches:
dir /B output\cache\visibility_*.bin 2>nul

echo.
echo Done!
