Write-Host "=== Cleaning build and output folders ===" -ForegroundColor Cyan
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path output) { Remove-Item output -Recurse -Force }
Write-Host "Clean complete."
    Write-Host "`nOutput directory preserved (use -All to clean)" -ForegroundColor Gray
}

Write-Host "`n✓ Clean complete!" -ForegroundColor Green
Write-Host "Run .\build.ps1 to rebuild the project`n" -ForegroundColor Gray
