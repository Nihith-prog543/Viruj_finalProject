# 🔧 Fix Render Root Directory Error

## Problem
Render is looking for `/opt/render/project/src/synthesis_route_finder` but it doesn't exist.

## Solution: Clear Root Directory in Render Dashboard

### Step 1: Go to Render Dashboard
1. Log in to https://render.com
2. Click on your service: `synthesis-route-finder`

### Step 2: Open Settings
1. Click on the **"Settings"** tab (top navigation)

### Step 3: Find Root Directory Field
1. Scroll down to **"Build & Deploy"** section
2. Look for **"Root Directory"** field
3. **DELETE** any value in this field (make it empty/blank)
4. Click **"Save Changes"** button

### Step 4: Verify Build & Start Commands
Make sure these are set correctly:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

### Step 5: Deploy
1. Go to **"Manual Deploy"** tab
2. Click **"Clear build cache & deploy"** button
3. Wait for deployment to complete

## Why This Works
- Your GitHub repo `Synthesis-route-finder` has `app.py` at the **root** level
- Render was looking for it in a subdirectory `synthesis_route_finder/` that doesn't exist
- By clearing Root Directory, Render will use the repository root where `app.py` actually is

## If Root Directory Field is Not Visible
If you can't find the Root Directory field:
1. Try deleting and recreating the service
2. Or contact Render support

## Alternative: If You Must Use Root Directory
If for some reason you need to keep a Root Directory setting, you would need to:
1. Restructure your GitHub repo to have files in a subdirectory
2. But this is NOT recommended - just clear the Root Directory instead!

