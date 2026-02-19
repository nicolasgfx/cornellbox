@echo off
cmake --preset release
cmake --build --preset release
if %errorlevel% equ 0 (
    build\bin\Release\viewer.exe --scene scenes/conference/conference.obj %*
)
