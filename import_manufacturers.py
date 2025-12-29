import pandas as pd
import sqlite3
from pathlib import Path

EXCEL_PATH = Path(r"C:\Users\HP\Desktop\DOM\API_Manufacturers_List.csv")
DB_PATH = Path(__file__).resolve().parent / "viruj_local.db"

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Excel file not found at {EXCEL_PATH}")

print(f"Loading data from {EXCEL_PATH}...")

if EXCEL_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(EXCEL_PATH)
else:
    df = pd.read_excel(EXCEL_PATH)

rename_map = {
    "API NAME": "api_name",
    "Manufacturers (API suppliers)": "manufacturer",
    "Country": "country",
    "USDMF": "usdmf",
    "CEP": "cep",
}
df = df.rename(columns=rename_map)
missing_cols = [col for col in rename_map.values() if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns after renaming: {missing_cols}")

df = df[list(rename_map.values())]
df = df.fillna("")
df = df.drop_duplicates(subset=["api_name", "manufacturer", "country"])
df["source_file"] = EXCEL_PATH.name
df["imported_at"] = ""
df["source_url"] = ""
df["source_name"] = ""

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS API_manufacturers;")
cursor.execute(
    """
    CREATE TABLE API_manufacturers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """
)

df.to_sql("API_manufacturers", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print(f"Imported {len(df)} rows into {DB_PATH}")

