# 🚀 Deploy with Gemini CLI - Step by Step Guide

This guide will help you deploy your Flask application using Google's Gemini CLI.

## ⚠️ Important Notes

- **Gemini CLI** is an AI-powered development tool that can help deploy applications
- It typically deploys to **Google Cloud Platform (GCP)**
- You'll need a **Google Cloud account** (free tier available with $300 credit)
- Gemini CLI uses AI to help automate the deployment process

---

## ✅ Prerequisites

1. **Node.js 18 or higher** installed
   - Check: `node --version`
   - Download: https://nodejs.org/

2. **Google Account** with access to Google Cloud Platform
   - Sign up: https://cloud.google.com/ (free tier available)

3. **Python 3.12** (already installed ✅)

4. **Your Flask app** (ready ✅)

---

## 📦 Step 1: Install Gemini CLI

### Option A: Global Installation (Recommended)
```powershell
npm install -g @google/gemini-cli
```

### Option B: Use npx (No Installation)
```powershell
npx @google/gemini-cli
```

**Verify Installation:**
```powershell
gemini --version
```

---

## 🔐 Step 2: Authenticate with Google

1. **Run Gemini CLI:**
   ```powershell
   gemini
   ```

2. **Sign in with your Google account** when prompted
   - This grants the CLI access to deploy your application
   - You'll get up to 60 model requests per minute and 1,000 requests per day

3. **Verify authentication:**
   ```powershell
   gemini auth status
   ```

---

## 🔧 Step 3: Prepare Your App for Deployment

### 3.1 Ensure Required Files Exist

Your app already has:
- ✅ `requirements.txt` (with all dependencies)
- ✅ `Procfile` (for gunicorn)
- ✅ `app.py` (main Flask application)
- ✅ `runtime.txt` (Python version)

### 3.2 Create `.gcloudignore` (Optional but Recommended)

Create a file named `.gcloudignore` in your project root:

```
venv/
__pycache__/
*.pyc
*.db
*.db-shm
*.db-wal
.env
.git/
*.log
```

### 3.3 Prepare Environment Variables

Gemini CLI will need to know about your environment variables. Create a file `app.yaml` for Google App Engine (if deploying to App Engine):

```yaml
runtime: python312
env: standard

env_variables:
  PORT: 8080
  GROQ_API_KEY: "your_groq_key_here"
  GOOGLE_API_KEY: "your_google_key_here"
  GOOGLE_CSE_ID: "your_cse_id_here"
  GOOGLE_CSE_API_KEY: "your_cse_api_key_here"
  HUGGINGFACE_API_KEY: "your_hf_key_here"
  SERP_API_KEY: "your_serp_key_here"
```

**⚠️ Security Note:** For production, use Google Cloud Secret Manager instead of hardcoding keys in `app.yaml`.

---

## 🚀 Step 4: Deploy with Gemini CLI

### 4.1 Navigate to Your Project Directory
```powershell
cd C:\Users\HP\Desktop\DOM\synthesis_route_finder
```

### 4.2 Run Deployment Command
```powershell
gemini deploy
```

### 4.3 Follow the Interactive Prompts

Gemini CLI will ask you questions like:
- **Deployment target**: Choose Google Cloud Platform
- **Service type**: Web application
- **Region**: Choose closest to your users
- **Project configuration**: It may help create/configure your GCP project

### 4.4 Alternative: Use Gemini CLI for Assistance

If `gemini deploy` doesn't work directly, you can use Gemini CLI to help you:

```powershell
# Ask Gemini CLI for deployment help
gemini "How do I deploy this Flask app to Google Cloud Platform?"

# Or get help creating deployment files
gemini "Create a Dockerfile and app.yaml for this Flask application"
```

---

## 🌐 Step 5: Manual Google Cloud Deployment (If Needed)

If Gemini CLI doesn't handle everything automatically, you can deploy manually:

### 5.1 Install Google Cloud SDK
```powershell
# Download and install from:
# https://cloud.google.com/sdk/docs/install
```

### 5.2 Initialize Google Cloud
```powershell
gcloud init
```

### 5.3 Create App Engine Application
```powershell
gcloud app create --region=us-central
```

### 5.4 Deploy to App Engine
```powershell
gcloud app deploy
```

---

## 🔑 Step 6: Set Environment Variables in Google Cloud

### Option A: Using Google Cloud Console
1. Go to https://console.cloud.google.com
2. Navigate to your App Engine service
3. Go to **Settings** → **Environment Variables**
4. Add all your API keys

### Option B: Using Secret Manager (Recommended for Production)
```powershell
# Create secrets
gcloud secrets create groq-api-key --data-file=- <<< "your_groq_key"
gcloud secrets create google-api-key --data-file=- <<< "your_google_key"
# ... repeat for all keys

# Grant access to App Engine
gcloud secrets add-iam-policy-binding groq-api-key \
  --member="serviceAccount:YOUR_PROJECT@appspot.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 📝 Step 7: Update app.py for Cloud Deployment

Your `app.py` already handles the PORT environment variable correctly:
```python
port = int(os.environ.get('PORT', 5000))
```

This is perfect for Google Cloud Platform! ✅

---

## 🧪 Step 8: Test Your Deployment

After deployment, Gemini CLI or Google Cloud will provide you with a URL like:
```
https://your-project-id.appspot.com
```

Test your endpoints:
- `https://your-project-id.appspot.com/` (main page)
- `https://your-project-id.appspot.com/api/analyze` (API endpoint)

---

## 💰 Cost Considerations

### Google Cloud Platform Free Tier:
- **App Engine**: 28 hours/day free (F1 instances)
- **$300 credit** for new accounts (valid for 90 days)
- After free tier: Pay-as-you-go pricing

### Estimated Monthly Cost (After Free Tier):
- **Small traffic**: ~$5-15/month
- **Medium traffic**: ~$20-50/month

---

## 🆘 Troubleshooting

### Issue: "Command not found: gemini"
**Solution:** Make sure Node.js is installed and Gemini CLI is installed globally:
```powershell
npm install -g @google/gemini-cli
```

### Issue: "Authentication failed"
**Solution:** Re-authenticate:
```powershell
gemini auth login
```

### Issue: "Deployment failed"
**Solution:** 
1. Check Google Cloud SDK is installed
2. Verify you have a GCP project created
3. Check billing is enabled (even for free tier)
4. Review error logs in Google Cloud Console

### Issue: "Environment variables not working"
**Solution:**
1. Verify variables are set in Google Cloud Console
2. Restart the App Engine service
3. Check `app.py` is reading from `os.environ`

---

## 🔄 Alternative: Use Gemini CLI for Deployment Scripts

If direct deployment doesn't work, use Gemini CLI to generate deployment scripts:

```powershell
gemini "Generate a deployment script for this Flask app to deploy to Google Cloud Run"
```

This will help you create:
- `Dockerfile`
- `cloudbuild.yaml`
- Deployment scripts

---

## 📚 Additional Resources

- **Gemini CLI Docs**: https://google-gemini.github.io/gemini-cli/
- **Google Cloud App Engine**: https://cloud.google.com/appengine/docs
- **Google Cloud Run** (Alternative): https://cloud.google.com/run/docs

---

## ✅ Quick Checklist

Before deploying, ensure:
- [ ] Node.js 18+ installed
- [ ] Gemini CLI installed (`npm install -g @google/gemini-cli`)
- [ ] Authenticated with Google (`gemini auth login`)
- [ ] Google Cloud account created
- [ ] Billing enabled (even for free tier)
- [ ] `requirements.txt` up to date
- [ ] `Procfile` exists
- [ ] Environment variables documented
- [ ] Database files handled (upload or use Cloud SQL)

---

## 🎉 Success!

Once deployed, your app will be live on Google Cloud Platform and accessible worldwide!

**Next Steps:**
1. Monitor usage in Google Cloud Console
2. Set up custom domain (optional)
3. Configure auto-scaling
4. Set up monitoring and alerts

