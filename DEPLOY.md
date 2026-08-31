# CatalystDesk Deployment Guide

## ✅ Setup Complete!

Your HuggingFace token and Streamlit config are ready. Now let's deploy to GitHub and Streamlit Cloud.

---

## Step 1: Create GitHub Repository

Run these commands in PowerShell:

```powershell
cd "d:\AI projects\CatalystDesk"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Check what's being committed (secrets.toml should NOT appear)
git status

# Commit
git commit -m "Initial commit: CatalystDesk - AI-powered trader briefs with RAG, LangGraph agents, and evals"
```

## Step 2: Create GitHub Repo (via Web)

1. **Go to https://github.com/new**
2. **Repository name**: `CatalystDesk`
3. **Description**: `$0 LangGraph + RAG briefing system for traders`
4. **Public** (so Streamlit Cloud can access it)
5. **Click "Create repository"**

## Step 3: Push to GitHub

After creating the repo, GitHub shows commands. Copy and run these:

```powershell
cd "d:\AI projects\CatalystDesk"

# Add remote (replace kingstonjoeldas with your username)
git remote add origin https://github.com/kingstonjoeldas/CatalystDesk.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

**Expected**: Files appear on GitHub at `https://github.com/kingstonjoeldas/CatalystDesk`

## Step 4: Deploy to Streamlit Cloud

1. **Go to https://share.streamlit.io**
   - Login with GitHub (or create account)

2. **Click "New app"** (top right button)

3. **Fill the form:**
   - **GitHub repo**: `kingstonjoeldas/CatalystDesk`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`

4. **Click "Deploy"**
   - App deploys in 1-2 minutes
   - URL: `https://kingstonjoeldas-catalystdesk.streamlit.app` (or similar)

## Step 5: Add Secrets in Streamlit Cloud

After app is running:

1. **Click ☰ (hamburger menu) → Settings**
2. **Click "Secrets" tab**
3. **Paste:**
   ```
   HF_TOKEN = "hf_YOUR_TOKEN_HERE"
   FRED_API_KEY = ""
   ```
   (Replace `hf_YOUR_TOKEN_HERE` with your actual token)
4. **Click "Save"**
5. **App auto-redeploys** (30 seconds)

## Step 6: Test

1. **Wait for green "Running" indicator**
2. **Query**: "What are the risks for AAPL?"
3. **Expect**: Brief in 5-10 seconds (first run downloads HF models)

---

## 🎉 Live!

Your CatalystDesk is now deployed at:
```
https://kingstonjoeldas-catalystdesk.streamlit.app
```

Share this link with traders!

---

## Troubleshooting

### "App is loading... taking longer than usual"
- Normal on first deploy (HF models ~500MB downloading)
- Wait 2-3 minutes
- Refresh page if stuck

### "HF_TOKEN error"
- Verify token in Settings → Secrets
- Make sure it starts with `hf_`
- Re-save and wait for redeploy

### "Module not found"
- Check `requirements.txt` is in repo root
- Click Settings → Reboot app

---

## What Happens to Data?

**Local (your PC):**
- `.streamlit/secrets.toml` ← Your token (stays local, never pushed to git)
- `cache/` ← HF model cache
- `data/chroma/` ← Vector DB
- `data/traces.db` ← Query logs

**On Streamlit Cloud:**
- Chroma DB resets on redeploy (ephemeral storage)
- Traces reset on redeploy
- Models re-download on redeploy (takes time)

**Workaround** (if needed):
- Keep running L1-L4 locally to pre-populate Chroma
- Or accept first query being slow while models download

---

## Next Steps

1. ✅ GitHub repo created and pushed
2. ✅ App deployed to Streamlit Cloud
3. ✅ Secrets configured
4. ✅ Testing complete
5. 📊 Share link with traders
6. 📈 Collect feedback (thumbs up/down)
7. 🔄 Iterate based on L6 metrics

---

## Questions?

Check `INTERVIEW.md` for architecture details, or look at individual phase scripts (L1-L7) for how each component works.

**Happy deploying! 🚀**
