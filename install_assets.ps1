# Prevent script from continuing if a command fails
$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Starting a FULL CLEAN INSTALL of Piper TTS Assets" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$ProjectRoot = Get-Location
$AssetsDir = Join-Path $ProjectRoot "assets"

# 1. Purge Existing Assets Directory
if (Test-Path $AssetsDir) {
    Write-Host "Wiping existing assets folder completely..." -ForegroundColor Red
    Remove-Item -Path $AssetsDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Creating fresh assets directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $AssetsDir | Out-Null

# 2. Reset Git LFS Index Cache (if working inside a Git repository)
if (Test-Path .git) {
    Write-Host "Cleaning local Git caching records..." -ForegroundColor Yellow
    git lfs install
    git lfs track "*.onnx"
    
    # Fixed syntax to cleanly drop the index files without causing flag switches
    git rm --cached -r -f --ignore-unmatch "assets/*" 2>$null | Out-Null
}

# 3. Base URL and Voice Model Array (Updated to correct HF path)
$BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB"
$Voices = @(
    "cori/medium/en_GB-cori-medium.onnx",
    "cori/medium/en_GB-cori-medium.onnx.json",
    "jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
    "jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
    "cori/high/en_GB-cori-high.onnx",
    "cori/high/en_GB-cori-high.onnx.json"
)

# 4. Download Clean Binary Assets
Write-Host "Downloading fresh voice profiles directly to assets..." -ForegroundColor Cyan

foreach ($FilePath in $Voices) {
    $Filename = Split-Path $FilePath -Leaf
    $TargetFile = Join-Path $AssetsDir $Filename
    $DownloadUrl = "$BaseUrl/$FilePath"

    Write-Host "Downloading fresh copy of $Filename..." -ForegroundColor Yellow
    
    # Direct binary download to guarantee file integrity
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TargetFile -UserAgent "Mozilla/5.0"
}

Write-Host "==================================================" -ForegroundColor Green
Write-Host "Clean install completed successfully!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# 5. Execute your application script to test
Write-Host "Testing fresh build with: uv run main.py..." -ForegroundColor Cyan
uv run main.py
