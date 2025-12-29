# 🚀 Deploy to Replit - Step by Step Guide

Your Flask app can be easily deployed on Replit! Follow these steps:

## ✅ Step 1: Import Your Code to Replit

### Option A: Import from GitHub (Recommended)
1. In Replit, click **"Import code or design"** (left sidebar)
2. Select **"Import from GitHub"**
3. Enter your repository URL: `https://github.com/Nihith-prog543/Synthesis-route-finder.git`
4. Click **"Import"**
5. Replit will clone your repository

### Option B: Create New Repl and Upload Files
1. Click **"+ Create App"** → **"Web app"**
2. Choose **"Python"** as the language
3. Name it: `synthesis-route-finder`
4. Click **"Create"**
5. Upload your files using the file explorer

---

## ✅ Step 2: Install Dependencies

1. In Replit, open the **Shell** (bottom panel)
2. Run:
```bash
pip install -r requirements.txt
```

---

## ✅ Step 3: Set Up Environment Variables (Secrets)

**IMPORTANT**: Add your API keys as Replit Secrets!

1. Click the **🔒 Lock icon** in the left sidebar (or Tools → Secrets)
2. Add these secrets:

```
OPENAI_API_KEY = your_openai_key_here
GROQ_API_KEY = your_groq_key_here
GOOGLE_API_KEY = your_google_api_key_here
GOOGLE_CSE_ID = your_google_cse_id_here
```

3. Click **"Add secret"** for each one

---

## ✅ Step 4: Upload Required Files

### Database File:
1. Click the **Files** icon in the left sidebar
2. Click **"Upload file"** (or drag and drop)
3. Upload your `viruj_local.db` file
4. Place it in the project root or `synthesis_route_finder/` folder

### Excel File:
1. Upload `API_Manufacturers_List.csv` to the project root
2. Or place it in the `DOM/` folder (as your code expects)

---

## ✅ Step 5: Configure for Web Deployment

### Update .replit file (Already created ✅)
The `.replit` file is already configured with:
- Run command: `python app.py`
- Deploy command: `gunicorn app:app`

### Install Gunicorn (if not in requirements):
```bash
pip install gunicorn
```

---

## ✅ Step 6: Run Your App

1. Click the **"Run"** button (top center)
2. Replit will start your Flask app
3. You'll see the URL in the output (e.g., `https://your-app-name.replit.app`)

---

## ✅ Step 7: Deploy as Web App (Make it Public)

1. Click the **"Deploy"** button (top right, or in the sidebar)
2. Or go to **"Published apps"** in the sidebar
3. Click **"Publish"** or **"Deploy"**
4. Your app will be live at: `https://your-app-name.replit.app`

---

## 📋 Important Notes for Replit

### File Structure:
Make sure your files are in the correct location:
```
/
├── app.py
├── requirements.txt
├── Procfile
├── .replit
├── synthesis_route_finder/
│   ├── app.py (if you have nested structure)
│   ├── synthesis_engine/
│   └── templates/
├── viruj_local.db (upload this)
└── API_Manufacturers_List.csv (upload this)
```

### Database Path:
If your code looks for database in parent directory, you may need to adjust paths in Replit.

### Port Configuration:
Replit uses port `8080` by default. The `.replit` file is configured for this.

---

## 🆘 Troubleshooting

### App Won't Start:
- Check the Shell for error messages
- Verify all dependencies are installed
- Check that environment variables (secrets) are set

### Database Not Found:
- Verify `viruj_local.db` is uploaded
- Check the file path in your code matches Replit's structure

### Excel File Not Found:
- Upload `API_Manufacturers_List.csv` to the project root
- Or update the path in `app.py` to match Replit's file structure

### Port Issues:
- Replit uses port 8080 by default
- Make sure your app uses `os.environ.get('PORT', 8080)`

---

## 🎉 Success!

Once deployed, your app will be:
- ✅ Accessible at `https://your-app-name.replit.app`
- ✅ HTTPS enabled
- ✅ Free on Replit Starter Plan
- ✅ Easy to update (just push to GitHub or edit in Replit)

---

## 💡 Pro Tips

1. **Auto-Deploy**: Replit can auto-deploy from GitHub
2. **Always-On**: Upgrade to Replit Core for always-on apps (free tier may sleep)
3. **File Persistence**: Files uploaded to Replit persist between sessions
4. **Collaboration**: Share your Repl with team members

---

## 📞 Need Help?

- Replit Docs: https://docs.replit.com
- Replit Community: https://ask.replit.com

