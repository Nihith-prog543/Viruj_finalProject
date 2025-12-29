# Deployment Guide - Cost-Effective Options

This guide provides step-by-step instructions for deploying the Synthesis Route Finder application to make it accessible to everyone.

## 🚀 Recommended Deployment Options (Free/Low-Cost)

### Option 1: Render.com (Recommended - FREE Tier Available)
**Best for: Easy deployment, free tier, automatic HTTPS**

#### Steps:
1. **Create Account**: Go to [render.com](https://render.com) and sign up (free)

2. **Prepare Your Code**:
   - Create a GitHub repository and push your code
   - Or use Render's direct deployment

3. **Deploy on Render**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Settings:
     - **Name**: `synthesis-route-finder`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Plan**: Free (or Starter $7/month for better performance)

4. **Environment Variables** (Add in Render Dashboard):
   ```
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   GOOGLE_CSE_ID=your_cse_id_here
   SQLITE_DB_FILENAME=/opt/render/project/src/viruj_local.db
   ```

5. **Database Setup**:
   - Upload your `viruj_local.db` file to Render
   - Or use Render's PostgreSQL (free tier available)

**Cost**: FREE (with limitations) or $7/month for better performance

---

### Option 2: Railway.app (FREE Tier with $5 Credit)
**Best for: Simple deployment, good free tier**

#### Steps:
1. **Sign up**: Go to [railway.app](https://railway.app) (free $5 credit)

2. **Deploy**:
   - Click "New Project" → "Deploy from GitHub"
   - Connect your repository
   - Railway auto-detects Flask and deploys

3. **Environment Variables**: Add in Railway dashboard (same as Render)

4. **Database**: Railway provides PostgreSQL free tier

**Cost**: FREE ($5 credit monthly) or pay-as-you-go

---

### Option 3: PythonAnywhere (FREE Tier)
**Best for: Python-focused, easy setup**

#### Steps:
1. **Sign up**: [pythonanywhere.com](https://www.pythonanywhere.com) (free account)

2. **Upload Files**:
   - Use Files tab to upload your project
   - Or use Git: `git clone your-repo-url`

3. **Configure Web App**:
   - Go to Web tab → "Add a new web app"
   - Choose Flask → Python 3.10
   - Set source code path: `/home/yourusername/synthesis_route_finder`
   - Set WSGI file: `/var/www/yourusername_pythonanywhere_com_wsgi.py`

4. **Update WSGI File**:
   ```python
   import sys
   path = '/home/yourusername/synthesis_route_finder'
   if path not in sys.path:
       sys.path.append(path)
   from app import app as application
   ```

5. **Reload Web App**

**Cost**: FREE (with limitations) or $5/month for better features

---

### Option 4: Fly.io (FREE Tier)
**Best for: Global deployment, good performance**

#### Steps:
1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`

2. **Login**: `fly auth login`

3. **Create App**: `fly launch` (in your project directory)

4. **Deploy**: `fly deploy`

**Cost**: FREE (generous free tier)

---

### Option 5: Local Network Deployment (FREE)
**Best for: Internal company use, no internet access needed**

#### Steps:
1. **Find Your IP**: 
   ```powershell
   ipconfig
   # Look for IPv4 Address (e.g., 192.168.1.100)
   ```

2. **Update Flask App**:
   ```python
   app.run(host='0.0.0.0', port=5000, debug=False)
   ```

3. **Configure Firewall**:
   - Windows Firewall → Allow app through firewall
   - Allow Python/Flask on port 5000

4. **Access**: Others can access via `http://YOUR_IP:5000`

**Cost**: FREE (only accessible on same network)

---

## 📋 Pre-Deployment Checklist

### 1. Update Requirements
Make sure `requirements.txt` includes:
```
flask==3.0.3
gunicorn==21.2.0
pandas==2.1.3
openpyxl==3.1.2
xlrd==2.0.1
requests==2.31.0
beautifulsoup4==4.12.2
groq==0.19.0
sqlalchemy==2.0.23
```

### 2. Create Procfile (for Render/Railway)
```
web: gunicorn app:app
```

### 3. Environment Variables
Set these in your deployment platform:
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`
- `SQLITE_DB_FILENAME` (optional, defaults to viruj_local.db)

### 4. Database File
- Upload `viruj_local.db` to your deployment
- Or use cloud database (PostgreSQL on Render/Railway)

### 5. Excel File Location
- Upload `API_Manufacturers_List.csv` to the deployment
- Or update the path in code to use cloud storage

---

## 🔧 Quick Setup for Render.com (Recommended)

1. **Install Gunicorn** (add to requirements.txt):
   ```
   gunicorn==21.2.0
   ```

2. **Create Procfile**:
   ```
   web: gunicorn app:app
   ```

3. **Push to GitHub**

4. **Deploy on Render**:
   - Connect GitHub repo
   - Render auto-detects and deploys
   - Add environment variables
   - Done! 🎉

---

## 💡 Tips for Cost-Effective Deployment

1. **Use Free Tiers**: All platforms offer free tiers sufficient for testing/small teams
2. **Optimize Database**: SQLite is fine for small-medium datasets
3. **Cache Static Files**: Use CDN for static assets (free on Cloudflare)
4. **Monitor Usage**: Set up alerts to avoid unexpected costs
5. **Use Environment Variables**: Never commit API keys to Git

---

## 🆘 Troubleshooting

### Common Issues:
1. **Database not found**: Ensure database file is uploaded or use cloud DB
2. **Port issues**: Use platform's PORT environment variable
3. **Import errors**: Check all dependencies in requirements.txt
4. **Static files**: Ensure static folder structure is correct

---

## 📞 Need Help?

Check platform-specific documentation:
- Render: https://render.com/docs
- Railway: https://docs.railway.app
- PythonAnywhere: https://help.pythonanywhere.com

