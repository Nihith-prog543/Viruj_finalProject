# 🚀 Database Migration to Supabase (PostgreSQL) - Complete Guide

## Why Supabase?
- ✅ **Free tier** with 500MB database (plenty for your data)
- ✅ **Persistent storage** - data survives app restarts
- ✅ **Easy setup** - 5 minutes to get started
- ✅ **PostgreSQL** - industry standard, works great with SQLAlchemy
- ✅ **Dashboard** - visual interface to manage data
- ✅ **Works on Render** - no additional disk space needed

## Step 1: Create Supabase Account & Project

1. Go to https://supabase.com
2. Click "Start your project" → Sign up (free)
3. Click "New Project"
4. Fill in:
   - **Name**: `pharmaceutical-db` (or any name)
   - **Database Password**: Create a strong password (save it!)
   - **Region**: Choose closest to you
   - Click "Create new project"
5. Wait 2-3 minutes for project to initialize

## Step 2: Get Database Connection String

1. In your Supabase project dashboard, go to **Settings** → **Database**
2. Scroll down to **Connection string**
3. Under **URI**, copy the connection string
   - It looks like: `postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres`
4. Replace `[YOUR-PASSWORD]` with your actual database password
5. **Save this connection string** - you'll need it!

## Step 3: Create Tables in Supabase

1. In Supabase dashboard, go to **SQL Editor**
2. Click "New query"
3. Copy and paste the SQL below, then click "Run":

```sql
-- Create viruj table (API Buyers)
CREATE TABLE IF NOT EXISTS viruj (
    id SERIAL PRIMARY KEY,
    company TEXT NOT NULL,
    form TEXT,
    strength TEXT,
    verification_source TEXT,
    confidence INTEGER DEFAULT 0,
    url TEXT,
    additional_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    api TEXT NOT NULL,
    country TEXT NOT NULL,
    UNIQUE(api, country, company)
);

-- Create API_manufacturers table
CREATE TABLE IF NOT EXISTS API_manufacturers (
    id SERIAL PRIMARY KEY,
    api_name TEXT NOT NULL,
    manufacturer TEXT NOT NULL,
    country TEXT,
    usdmf TEXT,
    cep TEXT,
    source_file TEXT,
    imported_at TEXT,
    source_url TEXT,
    source_name TEXT,
    UNIQUE(api_name, manufacturer, country)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_viruj_api_country ON viruj(api, country);
CREATE INDEX IF NOT EXISTS idx_viruj_company ON viruj(company);
CREATE INDEX IF NOT EXISTS idx_manufacturers_api_country ON API_manufacturers(api_name, country);
CREATE INDEX IF NOT EXISTS idx_manufacturers_manufacturer ON API_manufacturers(manufacturer);
```

4. You should see "Success. No rows returned"

## Step 4: Import Your Data

Run the migration script (see next section)

## Step 5: Set Environment Variables

Add to your `.env` file or Render environment variables:

```
# Use PostgreSQL (Supabase) when available, fallback to SQLite for local dev
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres

# Or set these separately:
DB_HOST=db.xxxxx.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASS=[YOUR-PASSWORD]
```

**For Render deployment:**
- Go to your Render dashboard → Your service → Environment
- Add `DATABASE_URL` with your Supabase connection string
- Make sure to replace `[YOUR-PASSWORD]` with actual password

## Step 6: Update requirements.txt

Make sure you have PostgreSQL driver:

```
psycopg2-binary==2.9.9
```

## That's it! 🎉

Your app will now:
- Use PostgreSQL (Supabase) when `DATABASE_URL` is set
- Fallback to SQLite for local development
- All new entries will be saved to Supabase
- Data persists across deployments!

