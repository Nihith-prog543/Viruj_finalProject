"""
Import end_manufacturers_list.csv into the API_manufacturers table
"""
import pandas as pd
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the synthesis_engine to path
sys.path.insert(0, os.path.dirname(__file__))
from synthesis_engine.api_manufacturer_service import ApiManufacturerService

# CSV file path
CSV_PATH = Path(r"C:\Users\HP\Desktop\end_manufacturers_list.csv")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found at: {CSV_PATH}")
    sys.exit(1)

print(f"📂 Loading CSV from: {CSV_PATH}")
print(f"📊 Reading CSV file...")

# Read CSV
df = pd.read_csv(CSV_PATH)

print(f"✅ Loaded {len(df)} rows from CSV")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Rename columns to match database schema
rename_map = {
    "API NAME": "api_name",
    "Manufacturers (API suppliers)": "manufacturer",
    "Country": "country",
    "USDMF": "usdmf",
    "CEP": "cep",
}

df = df.rename(columns=rename_map)

# Check for missing columns
missing_cols = [col for col in rename_map.values() if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns after renaming: {missing_cols}")
    sys.exit(1)

# Select only the columns we need
df = df[list(rename_map.values())]

# Clean data
df = df.fillna("")
df = df.drop_duplicates(subset=["api_name", "manufacturer", "country"])

# Add metadata columns
df["source_file"] = CSV_PATH.name
df["imported_at"] = datetime.now().isoformat()
df["source_url"] = ""
df["source_name"] = "end_manufacturers_list.csv"

print(f"\n📊 After cleaning: {len(df)} unique records")
print(f"\nSample data:")
print(df.head())

# Initialize service (this will use the correct database)
print(f"\n🔌 Connecting to database...")
service = ApiManufacturerService()
print(f"✅ Connected to: {service.db_path}")
print(f"✅ Table: {service.table_name}")

# Import data
print(f"\n📥 Importing data into database...")
try:
    from sqlalchemy import text
    
    imported_count = 0
    skipped_count = 0
    
    with service.engine.begin() as conn:
        for idx, row in df.iterrows():
            try:
                insert_stmt = text(f"""
                    INSERT INTO {service.table_name} 
                    (api_name, manufacturer, country, usdmf, cep, source_file, imported_at, source_url, source_name)
                    VALUES (:api_name, :manufacturer, :country, :usdmf, :cep, :source_file, :imported_at, :source_url, :source_name)
                """)
                
                conn.execute(insert_stmt, {
                    'api_name': str(row['api_name']).strip(),
                    'manufacturer': str(row['manufacturer']).strip(),
                    'country': str(row['country']).strip(),
                    'usdmf': str(row['usdmf']).strip(),
                    'cep': str(row['cep']).strip(),
                    'source_file': str(row['source_file']),
                    'imported_at': str(row['imported_at']),
                    'source_url': str(row['source_url']),
                    'source_name': str(row['source_name'])
                })
                imported_count += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"  Imported {idx + 1}/{len(df)} records...")
                    
            except Exception as e:
                # Likely a duplicate (UNIQUE constraint)
                skipped_count += 1
                if skipped_count <= 5:  # Only show first 5 errors
                    print(f"  ⚠️ Skipped row {idx + 1}: {str(e)[:100]}")
    
    print(f"\n✅ Import complete!")
    print(f"  - Imported: {imported_count} records")
    print(f"  - Skipped (duplicates): {skipped_count} records")
    print(f"  - Total: {imported_count + skipped_count} records processed")
    
    # Verify import
    with service.engine.begin() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {service.table_name}"))
        total_count = result.fetchone()[0]
        print(f"\n📊 Total records in database: {total_count}")
        
        # Show sample of imported data
        result = conn.execute(text(f"""
            SELECT api_name, manufacturer, country 
            FROM {service.table_name} 
            WHERE source_file = :source_file 
            LIMIT 5
        """), {'source_file': CSV_PATH.name})
        
        print(f"\n📋 Sample imported records:")
        for row in result.fetchall():
            print(f"  - {row[1]} ({row[0]}, {row[2]})")
            
except Exception as e:
    print(f"\n❌ Error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n✅ Done!")

