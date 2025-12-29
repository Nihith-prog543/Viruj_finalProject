# 🚀 Quick Migration Steps to Supabase

## Prerequisites
- Python 3.8+
- Your current SQLite database with data
- A Supabase account (free)

## Step-by-Step Migration

### 1. Export Your Current Data (Already Done ✅)
```bash
python export_sqlite_data.py
```
This creates:
- `viruj_export.csv` (API Buyers)
- `api_manufacturers_export.csv` (API Manufacturers)

### 2. Create Supabase Project
1. Go to https://supabase.com → Sign up (free)
2. Click "New Project"
3. Fill in:
   - Name: `pharmaceutical-db`
   - Password: (save this!)
   - Region: Choose closest
4. Wait 2-3 minutes for setup

### 3. Get Connection String
1. In Supabase dashboard → **Settings** → **Database**
2. Scroll to **Connection string** → **URI**
3. Copy the string (looks like):
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with your actual password

### 4. Create Tables in Supabase
1. In Supabase dashboard → **SQL Editor** → **New query**
2. Copy and paste this SQL:

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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_viruj_api_country ON viruj(api, country);
CREATE INDEX IF NOT EXISTS idx_viruj_company ON viruj(company);
CREATE INDEX IF NOT EXISTS idx_manufacturers_api_country ON API_manufacturers(api_name, country);
CREATE INDEX IF NOT EXISTS idx_manufacturers_manufacturer ON API_manufacturers(manufacturer);
```

3. Click **Run** (should see "Success")

### 5. Set Environment Variable
Create/update `.env` file in your project root:
```env
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
```
**Replace `[YOUR-PASSWORD]` with your actual password!**

### 6. Install PostgreSQL Driver
```bash
pip install psycopg2-binary
```
(Already added to requirements.txt ✅)

### 7. Run Migration
```bash
python migrate_to_supabase.py
```

This will:
- Connect to Supabase
- Import all your data
- Show progress and results

### 8. Verify Migration
1. In Supabase dashboard → **Table Editor**
2. Check `viruj` table (should have ~16,507 records)
3. Check `API_manufacturers` table (should have ~555 records)

### 9. Test Locally
```bash
python app.py
```
- Search for an API buyer
- Verify new entries are saved
- Check Supabase dashboard to see new entries appear

### 10. Deploy to Render
1. In Render dashboard → Your service → **Environment**
2. Add environment variable:
   - Key: `DATABASE_URL`
   - Value: `postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres`
3. **Redeploy** your service

## ✅ That's It!

Your app now:
- ✅ Uses PostgreSQL (Supabase) when `DATABASE_URL` is set
- ✅ Falls back to SQLite for local dev (if `DATABASE_URL` not set)
- ✅ All new entries persist across deployments
- ✅ Works on Render free tier (no disk space needed!)

## Troubleshooting

**Connection failed?**
- Check your password is correct in `DATABASE_URL`
- Make sure Supabase project is active
- Check firewall/network settings

**Migration failed?**
- Make sure CSV files exist (`viruj_export.csv`, `api_manufacturers_export.csv`)
- Check Supabase tables were created
- Look at error messages for specific issues

**Data not showing?**
- Verify `DATABASE_URL` is set correctly
- Check app logs for database connection messages
- Test connection: `python -c "from sqlalchemy import create_engine; engine = create_engine('YOUR_DATABASE_URL'); print('Connected!' if engine.connect() else 'Failed')"`

