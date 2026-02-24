[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "Installing Neural Network Toolbox" -ForegroundColor Cyan
Write-Host ""

$width = 40

for ($i = 0; $i -le 100; $i++) {
    $filled = [math]::Floor(($i / 100) * $width)
    $empty = $width - $filled

    $bar = ("#" * $filled) + ("-" * $empty)

    Write-Host -NoNewline "`rInstalling Neural Network Toolbox...  $bar $i%"
    Start-Sleep -Milliseconds 25
}

Write-Host "`n"

pip install -e . -q

if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "Run: nntoolbox" -ForegroundColor Yellow
} else {
    Write-Host "Installation failed. Please check your environment." -ForegroundColor Red
}