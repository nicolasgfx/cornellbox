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

echo.
echo --- Profile: medium ---
output\bin\radiosity.exe --low --nocache --phase 3
output\bin\radiosity.exe --medium --nocache --phase 3
output\bin\radiosity.exe --high --nocache --phase 3
if errorlevel 1 echo Warning: medium profile failed


echo.
echo Done!
