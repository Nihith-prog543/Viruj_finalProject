# 🚀 Deploy to Render.com - Step by Step

Your code is now on GitHub! Follow these steps to deploy it to Render.com (FREE).

## ✅ Step 1: Create Render Account

1. Go to **https://render.com**
2. Click **"Get Started for Free"** or **"Sign Up"**
3. Choose **"Sign up with GitHub"** (recommended - easier connection)
4. Authorize Render to access your GitHub account

---

## ✅ Step 2: Create New Web Service

1. Once logged in, click the **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see a list of your GitHub repositories
4. Find and click on **"Synthesis-route-finder"**
5. Click **"Connect"**

---

## ✅ Step 3: Configure Deployment Settings

Render will auto-detect most settings, but verify these:

### Basic Settings:
- **Name**: `synthesis-route-finder` (or keep default)
- **Environment**: `Python 3` (should be auto-selected)
- **Region**: Choose closest to your users (e.g., Singapore, Mumbai)
- **Branch**: `main` (should be auto-selected)
- **Root Directory**: Leave **empty** (or `synthesis_route_finder` if needed)

### Build & Deploy:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

### Plan:
- Select **"Free"** (or "Starter" for $7/month - better performance)

---

## ✅ Step 4: Add Environment Variables

**IMPORTANT**: You must add your API keys here!

1. Scroll down to **"Environment Variables"** section
2. Click **"Add Environment Variable"** for each:

### Required Variables:
```
OPENAI_API_KEY = your_openai_key_here
GROQ_API_KEY = your_groq_key_here
GOOGLE_API_KEY = your_google_api_key_here
GOOGLE_CSE_ID = your_google_cse_id_here
```

### Where to get your keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys
- **Google API Key**: https://console.cloud.google.com/apis/credentials
- **Google CSE ID**: https://programmablesearchengine.google.com/controlpanel/create

3. After adding each variable, click **"Add"**

---

## ✅ Step 5: Deploy!

1. Scroll to the bottom
2. Click **"Create Web Service"**
3. Render will start building your app
4. You'll see build logs in real-time
5. **Wait 5-10 minutes** for first deployment

---

## ✅ Step 6: Get Your Live URL

Once deployment completes:
- Status will show **"Live"** ✅
- Your app URL will be: `https://synthesis-route-finder.onrender.com` (or similar)
- **Share this URL with your team!** 🎉

---

## 📋 Important Notes

### Database File:
- Your `viruj_local.db` file is NOT in Git (it's in .gitignore)
- You'll need to either:
  - Upload it via Render Shell after deployment
  - Or use Render's PostgreSQL (free tier available)

### Excel File:
- Your `API_Manufacturers_List.csv` is also not in Git
- Upload it to Render Shell or commit it to Git if it's small

### Auto-Deploy:
- Render automatically deploys when you push to GitHub
- Every `git push` = new deployment

---

## 🆘 Troubleshooting

### Build Fails:
- Check build logs in Render dashboard
- Verify all dependencies in `requirements.txt`
- Check Python version compatibility

### App Crashes:
- Check logs in Render dashboard
- Verify all environment variables are set correctly
- Check database file path

### Database Not Found:
- Upload `viruj_local.db` via Render Shell
- Or switch to PostgreSQL (free on Render)

---

## 🎉 Success!

Once deployed, your app will be:
- ✅ Accessible from anywhere in the world
- ✅ HTTPS enabled (secure connection)
- ✅ Auto-updates on Git push
- ✅ FREE (with Render free tier)

**Your live URL**: `https://your-app-name.onrender.com`

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Render Support: support@render.com
- Check deployment logs in Render dashboard

