@echo off
cmake --preset release
cmake --build --preset release
if %errorlevel% equ 0 (
    build\bin\Release\radiosity.exe %*
)
