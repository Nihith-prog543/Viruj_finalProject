"""
Export SQLite data to CSV files for migration to PostgreSQL
"""
import sqlite3
import pandas as pd
import os
from pathlib import Path

# Database paths
viruj_db = r"C:\Users\HP\Desktop\manufactures_api\viruj_local.db"
manufacturers_db = r"C:\Users\HP\Desktop\manufactures_api\viruj_local.db"

print("="*80)
print("EXPORTING SQLITE DATA FOR MIGRATION")
print("="*80)

# Export viruj table (API Buyers)
if os.path.exists(viruj_db):
    print(f"\n📊 Exporting viruj table from: {viruj_db}")
    conn = sqlite3.connect(viruj_db)
    
    # Read all data
    df_viruj = pd.read_sql_query("SELECT * FROM viruj", conn)
    print(f"   Found {len(df_viruj)} records")
    
    # Export to CSV
    export_file = "viruj_export.csv"
    df_viruj.to_csv(export_file, index=False)
    print(f"   ✅ Exported to: {export_file}")
    
    # Show sample
    if not df_viruj.empty:
        print(f"\n   Sample columns: {list(df_viruj.columns)}")
        print(f"   Sample data:")
        print(df_viruj.head(3).to_string())
    
    conn.close()
else:
    print(f"\n⚠️ Database not found: {viruj_db}")

# Export API_manufacturers table
if os.path.exists(manufacturers_db):
    print(f"\n📊 Exporting API_manufacturers table from: {manufacturers_db}")
    conn = sqlite3.connect(manufacturers_db)
    
    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='API_manufacturers'")
    if cursor.fetchone():
        # Read all data
        df_manufacturers = pd.read_sql_query("SELECT * FROM API_manufacturers", conn)
        print(f"   Found {len(df_manufacturers)} records")
        
        # Export to CSV
        export_file = "api_manufacturers_export.csv"
        df_manufacturers.to_csv(export_file, index=False)
        print(f"   ✅ Exported to: {export_file}")
        
        # Show sample
        if not df_manufacturers.empty:
            print(f"\n   Sample columns: {list(df_manufacturers.columns)}")
            print(f"   Sample data:")
            print(df_manufacturers.head(3).to_string())
    else:
        print(f"   ⚠️ API_manufacturers table not found")
    
    conn.close()
else:
    print(f"\n⚠️ Database not found: {manufacturers_db}")

print("\n" + "="*80)
print("✅ Export complete! Files ready for migration.")
print("="*80)

