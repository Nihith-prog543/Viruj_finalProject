# 🚀 Step-by-Step Deployment Guide

Follow these steps to deploy your application to Render.com (FREE).

## Step 1: Prepare Your Code ✅

### 1.1 Check if Git is initialized
```powershell
cd C:\Users\HP\Desktop\DOM\synthesis_route_finder
git status
```

If you see "not a git repository", initialize it:
```powershell
git init
```

### 1.2 Create .gitignore (Already created ✅)
The `.gitignore` file is already created to exclude:
- Database files (*.db)
- Virtual environment (venv/)
- Environment variables (.env)
- Python cache files

### 1.3 Verify all files are ready
Make sure you have:
- ✅ `app.py`
- ✅ `requirements.txt` (with gunicorn)
- ✅ `Procfile`
- ✅ `templates/index.html`
- ✅ `synthesis_engine/` folder
- ✅ `API_Manufacturers_List.csv` (in DOM folder or synthesis_route_finder folder)

---

## Step 2: Create GitHub Repository 📦

### 2.1 Create account on GitHub (if you don't have one)
- Go to [github.com](https://github.com) and sign up (FREE)

### 2.2 Create a new repository
1. Click the "+" icon → "New repository"
2. Repository name: `synthesis-route-finder` (or any name you like)
3. Description: "Synthesis Route Finder - API Buyer and Manufacturer Finder"
4. Choose: **Public** (free) or **Private** (if you want it private)
5. **DO NOT** check "Initialize with README" (we already have files)
6. Click "Create repository"

### 2.3 Copy the repository URL
After creating, GitHub will show you a URL like:
```
https://github.com/yourusername/synthesis-route-finder.git
```
**Copy this URL** - you'll need it in the next step!

---

## Step 3: Push Code to GitHub 📤

### 3.1 Add all files to Git
```powershell
cd C:\Users\HP\Desktop\DOM\synthesis_route_finder
git add .
```

### 3.2 Commit your code
```powershell
git commit -m "Initial commit - Synthesis Route Finder app"
```

### 3.3 Connect to GitHub and push
```powershell
# Replace YOUR_USERNAME and REPO_NAME with your actual GitHub username and repo name
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

**Note**: You'll be asked for your GitHub username and password (use a Personal Access Token, not your password)

---

## Step 4: Create Personal Access Token (if needed) 🔑

If Git asks for authentication:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: `deployment-token`
4. Select scopes: Check **"repo"** (full control)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

---

## Step 5: Deploy on Render.com 🌐

### 5.1 Create Render account
1. Go to [render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with GitHub (recommended) or email

### 5.2 Create New Web Service
1. Click "New +" button (top right)
2. Select "Web Service"
3. Click "Connect GitHub" (if not already connected)
4. Authorize Render to access your repositories
5. Find and select your `synthesis-route-finder` repository
6. Click "Connect"

### 5.3 Configure Deployment Settings
Render will auto-detect settings, but verify:

- **Name**: `synthesis-route-finder` (or your choice)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or `master`)
- **Root Directory**: Leave empty (or `synthesis_route_finder` if your repo root is DOM)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Plan**: **Free** (or Starter $7/month for better performance)

### 5.4 Add Environment Variables
Click "Environment" tab and add:

```
OPENAI_API_KEY = your_openai_key_here
GROQ_API_KEY = your_groq_key_here
GOOGLE_API_KEY = your_google_api_key_here
GOOGLE_CSE_ID = your_google_cse_id_here
```

**Important**: 
- Click "Add" after each variable
- Use the exact names shown above
- Get your API keys from:
  - OpenAI: https://platform.openai.com/api-keys
  - Groq: https://console.groq.com/keys
  - Google: https://console.cloud.google.com/apis/credentials

### 5.5 Deploy!
1. Scroll down and click "Create Web Service"
2. Render will start building and deploying
3. Wait 5-10 minutes for first deployment
4. You'll see build logs in real-time

### 5.6 Get Your Live URL
Once deployed, you'll see:
- ✅ "Live" status
- Your URL: `https://synthesis-route-finder.onrender.com` (or similar)

**Share this URL with your team!** 🎉

---

## Step 6: Upload Database File (if needed) 💾

### Option A: Upload via Render Dashboard
1. Go to your service on Render
2. Click "Shell" tab
3. Upload `viruj_local.db` file
4. Or use Git to commit it (if small enough)

### Option B: Use Cloud Database (Recommended)
1. On Render, go to "New +" → "PostgreSQL"
2. Create free PostgreSQL database
3. Update your code to use PostgreSQL instead of SQLite
4. Import your data

---

## Step 7: Upload Excel File 📊

### Option A: Keep in Git (if file is small)
```powershell
# Copy file to synthesis_route_finder folder
copy C:\Users\HP\Desktop\DOM\API_Manufacturers_List.csv C:\Users\HP\Desktop\DOM\synthesis_route_finder\
git add API_Manufacturers_List.csv
git commit -m "Add API Manufacturers CSV file"
git push
```

### Option B: Upload via Render Shell
1. Go to Render → Your Service → Shell
2. Upload `API_Manufacturers_List.csv`
3. Place it in the project root

---

## Step 8: Test Your Deployment ✅

1. Visit your Render URL
2. Test all features:
   - ✅ AI Predicted Route tab
   - ✅ Find API Buyers tab
   - ✅ API Manufacturers tab (should auto-load)
3. Check if all data loads correctly

---

## 🆘 Troubleshooting

### Build Fails
- Check build logs on Render
- Verify `requirements.txt` has all dependencies
- Ensure Python version in `runtime.txt` is supported

### App Crashes
- Check logs in Render dashboard
- Verify all environment variables are set
- Check database file path

### Database Not Found
- Upload database file to Render
- Or switch to PostgreSQL (free on Render)

### Excel File Not Found
- Upload CSV file to project root
- Or update path in `app.py`

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Render Support: support@render.com
- Check deployment logs in Render dashboard

---

## 🎉 Success!

Once deployed, your app will be:
- ✅ Accessible from anywhere
- ✅ HTTPS enabled (secure)
- ✅ Auto-deploys on Git push
- ✅ FREE (with Render free tier)

**Share your URL**: `https://your-app-name.onrender.com`

