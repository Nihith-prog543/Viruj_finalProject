import sqlite3
import os

# Check manufacturers database
if os.path.exists('new_manufacturers.db'):
    conn = sqlite3.connect('new_manufacturers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print("Manufacturers DB tables:", tables)
    
    if 'API_manufacturers' in tables:
        cursor.execute("SELECT COUNT(*) FROM API_manufacturers WHERE LOWER(api_name) LIKE ? AND LOWER(country)=?", ('%rocuronium%', 'china'))
        count = cursor.fetchone()[0]
        print(f"Rocuronium Bromide + China in manufacturers DB: {count}")
        
        if count > 0:
            cursor.execute("SELECT api_name, manufacturer, country FROM API_manufacturers WHERE LOWER(api_name) LIKE ? AND LOWER(country)=? LIMIT 5", ('%rocuronium%', 'china'))
            print("Sample rows:", cursor.fetchall())
    conn.close()

