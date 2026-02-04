@echo off
REM Simple build wrapper using CMake Presets

if "%1"=="clean" (
    echo Cleaning build directory...
    rmdir /s /q build 2>nul
    echo Done.
    exit /b 0
)

if "%1"=="debug" (
    set PRESET=debug
) else (
    set PRESET=release
)

echo Building with preset: %PRESET% (full rebuild)
echo.

REM Always clean and reconfigure to ensure PTX is rebuilt
echo Cleaning build directory...
rmdir /s /q build 2>nul

echo Configuring...
cmake --preset %PRESET%
if errorlevel 1 exit /b 1

REM Build (this will compile PTX and everything)
echo Building...
cmake --build --preset %PRESET%
if errorlevel 1 exit /b 1

echo.
echo ✓ Build successful!
echo Executable: build\bin\Release\radiosity.exe (or Debug)
