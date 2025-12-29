# ⚡ Quick Start - Deploy in 10 Minutes

## Current Status ✅
- ✅ Git initialized
- ✅ All files ready
- ✅ Deployment files created

## Next Steps (Choose One Path)

### 🚀 PATH 1: Deploy to Render.com (Recommended - FREE)

#### Step 1: Create GitHub Repository (5 min)
1. Go to https://github.com and sign up/login
2. Click "+" → "New repository"
3. Name: `synthesis-route-finder`
4. Choose **Public** or **Private**
5. **DO NOT** check "Initialize with README"
6. Click "Create repository"
7. **Copy the repository URL** (e.g., `https://github.com/yourusername/synthesis-route-finder.git`)

#### Step 2: Push Code to GitHub (2 min)
Run these commands (replace YOUR_USERNAME and REPO_NAME):

```powershell
cd C:\Users\HP\Desktop\DOM\synthesis_route_finder
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

**If asked for password**: Use GitHub Personal Access Token (see below)

#### Step 3: Get GitHub Token (if needed) (2 min)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token (classic)"
3. Name: `deployment`
4. Check **"repo"** scope
5. Generate and **copy the token**
6. Use this token as password when pushing

#### Step 4: Deploy on Render (5 min)
1. Go to https://render.com → Sign up (FREE)
2. "New +" → "Web Service"
3. Connect GitHub → Select your repository
4. Settings:
   - Name: `synthesis-route-finder`
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app`
   - Plan: **Free**
5. Add Environment Variables:
   - `OPENAI_API_KEY`
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`
   - `GOOGLE_CSE_ID`
6. Click "Create Web Service"
7. Wait 5-10 minutes
8. **Done!** Your app is live at `https://your-app.onrender.com`

---

### 🏠 PATH 2: Local Network Deployment (FREE - Same Network Only)

#### Step 1: Update app.py (Already done ✅)
The app is already configured to run on `0.0.0.0`

#### Step 2: Run the App
```powershell
cd C:\Users\HP\Desktop\DOM\synthesis_route_finder
python app.py
```

#### Step 3: Find Your IP
```powershell
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)
```

#### Step 4: Share the URL
Others on your network can access:
```
http://YOUR_IP:5000
```

**Note**: Only works on same WiFi/network

---

## 📋 What You Need

### API Keys (for cloud deployment):
- ✅ OpenAI API Key
- ✅ Groq API Key  
- ✅ Google API Key
- ✅ Google CSE ID

### Files Ready:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `Procfile`
- ✅ `templates/index.html`
- ✅ `API_Manufacturers_List.csv` (in DOM folder)

---

## 🎯 Which Path Should You Choose?

**Choose Render.com if:**
- ✅ You want it accessible from anywhere
- ✅ You want HTTPS (secure)
- ✅ You want automatic updates
- ✅ You're okay with free tier limitations

**Choose Local Network if:**
- ✅ Only your team needs access
- ✅ You're on the same network
- ✅ You want zero cost
- ✅ You want full control

---

## 🆘 Need Help?

Check `DEPLOY_STEPS.md` for detailed instructions.

