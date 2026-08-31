# CatalystDesk Deployment Script
# This script automates pushing to GitHub and deploying to Streamlit Cloud

param(
    [string]$GitHubUsername = "kingstonjoeldas",
    [string]$RepoName = "CatalystDesk"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "CatalystDesk Deployment" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git not found. Please install Git for Windows." -ForegroundColor Red
    exit 1
}

# Navigate to project directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommandPath
Set-Location $ProjectDir
Write-Host "📁 Working directory: $ProjectDir" -ForegroundColor Green

# Check if .git exists
if (-not (Test-Path .git)) {
    Write-Host "`n🔄 Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Stage all files
Write-Host "`n📦 Staging files..." -ForegroundColor Yellow
git add .

# Check status (secrets.toml should be ignored)
Write-Host "`n✅ Files to commit:" -ForegroundColor Green
git status --short | ForEach-Object { Write-Host "  $_" }

# Verify secrets.toml is NOT staged
if (git status --short | Select-String ".streamlit/secrets.toml") {
    Write-Host "`n❌ ERROR: secrets.toml is staged! This should never be committed." -ForegroundColor Red
    Write-Host "Run: git reset .streamlit/secrets.toml" -ForegroundColor Yellow
    exit 1
}

# Commit
Write-Host "`n💾 Committing..." -ForegroundColor Yellow
git commit -m "Deploy CatalystDesk: AI trader briefs with RAG, LangGraph, and evals"

# Check if remote exists
$RemoteExists = git remote -v | Select-String "origin"
if (-not $RemoteExists) {
    Write-Host "`n🔗 Adding GitHub remote..." -ForegroundColor Yellow
    $RemoteUrl = "https://github.com/$GitHubUsername/$RepoName.git"
    Write-Host "   Remote: $RemoteUrl" -ForegroundColor Gray
    git remote add origin $RemoteUrl
}

# Set main branch
Write-Host "`n🌳 Setting main branch..." -ForegroundColor Yellow
git branch -M main

# Push to GitHub
Write-Host "`n🚀 Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "   This will prompt for your GitHub credentials." -ForegroundColor Gray
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "`n📍 Repository URL:" -ForegroundColor Cyan
    Write-Host "   https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Green

    Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Go to https://share.streamlit.io" -ForegroundColor Gray
    Write-Host "   2. Click 'New app'" -ForegroundColor Gray
    Write-Host "   3. Select repo: $GitHubUsername/$RepoName" -ForegroundColor Gray
    Write-Host "   4. Main file: app/streamlit_app.py" -ForegroundColor Gray
    Write-Host "   5. Click Deploy" -ForegroundColor Gray
    Write-Host "   6. Once running, add secrets (Settings → Secrets)" -ForegroundColor Gray
    Write-Host "`n🎉 All set! Your app will be live shortly." -ForegroundColor Green
} else {
    Write-Host "`n❌ Push failed. Check your GitHub credentials." -ForegroundColor Red
    exit 1
}
