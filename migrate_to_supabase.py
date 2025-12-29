"""
Migrate SQLite data to Supabase PostgreSQL
"""
import os
import pandas as pd
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("MIGRATING DATA TO SUPABASE POSTGRESQL")
print("="*80)

# Get Supabase connection string
database_url = os.getenv("DATABASE_URL")

if not database_url:
    print("\n❌ DATABASE_URL not found in environment variables!")
    print("\nPlease set DATABASE_URL in your .env file:")
    print("DATABASE_URL=postgresql://postgres:[PASSWORD]@db.xxxxx.supabase.co:5432/postgres")
    print("\nOr set it as an environment variable before running this script.")
    sys.exit(1)

print(f"\n✅ Connecting to Supabase...")
print(f"   Database: {database_url.split('@')[1] if '@' in database_url else 'Hidden'}")

try:
    # Connect to Supabase
    engine = create_engine(database_url, echo=False)
    
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("   ✅ Connected successfully!")
    
    # Read exported CSV files
    print("\n📊 Reading exported data...")
    
    # Import viruj data
    if os.path.exists("viruj_export.csv"):
        print("\n📥 Importing viruj table (API Buyers)...")
        df_viruj = pd.read_csv("viruj_export.csv")
        print(f"   Found {len(df_viruj)} records")
        
        # Clean data - remove id column (will be auto-generated)
        if 'id' in df_viruj.columns:
            df_viruj = df_viruj.drop(columns=['id'])
        
        # Ensure all required columns exist
        required_cols = ['company', 'form', 'strength', 'verification_source', 'confidence', 
                        'url', 'additional_info', 'created_at', 'updated_at', 'api', 'country']
        for col in required_cols:
            if col not in df_viruj.columns:
                df_viruj[col] = ''
        
        # Fill NaN values
        df_viruj = df_viruj.fillna('')
        
        # Insert data (using INSERT ... ON CONFLICT to avoid duplicates)
        inserted_count = 0
        skipped_count = 0
        
        with engine.begin() as conn:
            for idx, row in df_viruj.iterrows():
                try:
                    insert_stmt = text("""
                        INSERT INTO viruj (company, form, strength, verification_source, confidence, 
                                         url, additional_info, created_at, updated_at, api, country)
                        VALUES (:company, :form, :strength, :verification_source, :confidence,
                                :url, :additional_info, :created_at, :updated_at, :api, :country)
                        ON CONFLICT (api, country, company) DO NOTHING
                    """)
                    
                    result = conn.execute(insert_stmt, {
                        'company': str(row.get('company', '')),
                        'form': str(row.get('form', '')),
                        'strength': str(row.get('strength', '')),
                        'verification_source': str(row.get('verification_source', '')),
                        'confidence': int(row.get('confidence', 0)) if str(row.get('confidence', '')).isdigit() else 0,
                        'url': str(row.get('url', '')),
                        'additional_info': str(row.get('additional_info', '')),
                        'created_at': str(row.get('created_at', '')),
                        'updated_at': str(row.get('updated_at', '')),
                        'api': str(row.get('api', '')),
                        'country': str(row.get('country', ''))
                    })
                    if result.rowcount > 0:
                        inserted_count += 1
                    else:
                        skipped_count += 1
                    
                    if (idx + 1) % 1000 == 0:
                        print(f"   Processed {idx + 1}/{len(df_viruj)} records...")
                        
                except Exception as e:
                    print(f"   ⚠️ Error on row {idx + 1}: {str(e)[:100]}")
                    skipped_count += 1
        
        print(f"   ✅ Imported: {inserted_count} records")
        print(f"   ⚠️ Skipped (duplicates): {skipped_count} records")
    else:
        print("\n⚠️ viruj_export.csv not found. Skipping viruj import.")
    
    # Import API_manufacturers data
    if os.path.exists("api_manufacturers_export.csv"):
        print("\n📥 Importing API_manufacturers table...")
        df_manufacturers = pd.read_csv("api_manufacturers_export.csv")
        print(f"   Found {len(df_manufacturers)} records")
        
        # Clean data - remove id column
        if 'id' in df_manufacturers.columns:
            df_manufacturers = df_manufacturers.drop(columns=['id'])
        
        # Ensure all required columns exist
        required_cols = ['api_name', 'manufacturer', 'country', 'usdmf', 'cep', 
                        'source_file', 'imported_at', 'source_url', 'source_name']
        for col in required_cols:
            if col not in df_manufacturers.columns:
                df_manufacturers[col] = ''
        
        # Fill NaN values
        df_manufacturers = df_manufacturers.fillna('')
        
        # Insert data
        inserted_count = 0
        skipped_count = 0
        
        with engine.begin() as conn:
            for idx, row in df_manufacturers.iterrows():
                try:
                    insert_stmt = text("""
                        INSERT INTO API_manufacturers (api_name, manufacturer, country, usdmf, cep,
                                                      source_file, imported_at, source_url, source_name)
                        VALUES (:api_name, :manufacturer, :country, :usdmf, :cep,
                                :source_file, :imported_at, :source_url, :source_name)
                        ON CONFLICT (api_name, manufacturer, country) DO NOTHING
                    """)
                    
                    result = conn.execute(insert_stmt, {
                        'api_name': str(row.get('api_name', '')),
                        'manufacturer': str(row.get('manufacturer', '')),
                        'country': str(row.get('country', '')),
                        'usdmf': str(row.get('usdmf', '')),
                        'cep': str(row.get('cep', '')),
                        'source_file': str(row.get('source_file', '')),
                        'imported_at': str(row.get('imported_at', '')),
                        'source_url': str(row.get('source_url', '')),
                        'source_name': str(row.get('source_name', ''))
                    })
                    if result.rowcount > 0:
                        inserted_count += 1
                    else:
                        skipped_count += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"   Processed {idx + 1}/{len(df_manufacturers)} records...")
                        
                except Exception as e:
                    print(f"   ⚠️ Error on row {idx + 1}: {str(e)[:100]}")
                    skipped_count += 1
        
        print(f"   ✅ Imported: {inserted_count} records")
        print(f"   ⚠️ Skipped (duplicates): {skipped_count} records")
    else:
        print("\n⚠️ api_manufacturers_export.csv not found. Skipping manufacturers import.")
    
    # Verify migration
    print("\n📊 Verifying migration...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM viruj"))
        viruj_count = result.fetchone()[0]
        print(f"   viruj table: {viruj_count} records")
        
        result = conn.execute(text("SELECT COUNT(*) FROM API_manufacturers"))
        manufacturers_count = result.fetchone()[0]
        print(f"   API_manufacturers table: {manufacturers_count} records")
    
    print("\n" + "="*80)
    print("✅ Migration complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Set DATABASE_URL in your .env file or Render environment variables")
    print("2. Restart your application")
    print("3. All new entries will now be saved to Supabase!")
    
except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

