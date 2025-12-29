# 🔑 API Keys Location in Code

## ✅ Good News: Keys are NOT hardcoded!

Your API keys are loaded from **environment variables** for security. They are NOT stored in the code files.

---

## 📍 Where API Keys are Loaded:

### 1. **File: `synthesis_engine/api_buyer_finder.py`**
**Lines 55-59:**
```python
# API keys loaded from environment variables (required - no defaults for security)
self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "1475eceacf9eb4d06")
```

**Used for:**
- Finding API Buyers
- Google Custom Search
- OpenAI/Groq AI analysis

---

### 2. **File: `synthesis_engine/analysis.py`**
**Lines 49-54:**
```python
# API keys loaded from environment variables (required - no defaults for security)
self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
self.GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
self.SERP_API_KEY = os.getenv("SERP_API_KEY")
```

**Used for:**
- AI Predicted Route analysis
- Groq AI models
- Google Custom Search
- HuggingFace models
- SerpAPI (alternative search)

---

## 🔐 Required Environment Variables:

### For API Buyers Feature:
- `OPENAI_API_KEY` - OpenAI API key
- `GROQ_API_KEY` - Groq API key
- `GOOGLE_API_KEY` - Google Custom Search API key
- `GOOGLE_CSE_ID` - Google Custom Search Engine ID

### For AI Predicted Route Feature:
- `GROQ_API_KEY` - Groq API key
- `GOOGLE_CSE_API_KEY` or `GOOGLE_API_KEY` - Google API key
- `GOOGLE_CSE_ID` - Google Custom Search Engine ID
- `HUGGINGFACE_API_KEY` - (Optional) HuggingFace API key
- `SERP_API_KEY` - (Optional) SerpAPI key

---

## 🚀 How to Set API Keys:

### **Option 1: Local Development (Windows PowerShell)**
```powershell
$env:OPENAI_API_KEY = "your-key-here"
$env:GROQ_API_KEY = "your-key-here"
$env:GOOGLE_API_KEY = "your-key-here"
$env:GOOGLE_CSE_ID = "your-key-here"
```

### **Option 2: Create .env file** (Recommended for local)
Create a file named `.env` in `synthesis_route_finder/` folder:
```
OPENAI_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
GOOGLE_CSE_ID=your-key-here
```

Then load it in your code (add to `app.py`):
```python
from dotenv import load_dotenv
load_dotenv()
```

### **Option 3: Replit Secrets**
1. Click 🔒 Lock icon (Secrets)
2. Add each key as a secret

### **Option 4: Render.com Environment Variables**
1. Go to your service → Environment
2. Add each variable

---

## ⚠️ Important Security Notes:

1. ✅ **Never commit API keys to Git** - They are in `.gitignore`
2. ✅ **Use environment variables** - Already implemented!
3. ✅ **No hardcoded keys** - We removed them for security
4. ⚠️ **Keep keys secret** - Don't share them publicly

---

## 🔍 Where to Get Your Keys:

- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys
- **Google API Key**: https://console.cloud.google.com/apis/credentials
- **Google CSE ID**: https://programmablesearchengine.google.com/controlpanel/create
- **HuggingFace**: https://huggingface.co/settings/tokens (optional)
- **SerpAPI**: https://serpapi.com/dashboard (optional)

---

## 📝 Summary:

- **Keys are NOT in code files** ✅
- **Keys are loaded from environment variables** ✅
- **You need to set them in your deployment platform** ✅
- **Safe to push to GitHub** ✅ (no keys exposed)

