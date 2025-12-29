import os
import pandas as pd
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


class ApiManufacturerService:
    """
    Service to synchronize API manufacturer data from Excel/CSV files into SQLite
    and to query manufacturers by API name & country.
    """

    def __init__(self, db_filename: str | None = None, table_name: str = "api_manufacturers"):
        self.table_name = table_name
        self.db_path = None
        self.is_postgresql = False
        
        # Priority 1: Check for PostgreSQL connection (Supabase/cloud)
        database_url = os.getenv("DATABASE_URL")
        if database_url and database_url.startswith("postgresql://"):
            self.is_postgresql = True
            self.engine = create_engine(
                database_url,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                echo=False,
                pool_pre_ping=True,
            )
            print("✅ ApiManufacturerService: Using Supabase PostgreSQL database")
        else:
            # Priority 2: Fallback to SQLite (local development)
            self.db_path = self._determine_db_path(db_filename)
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                connect_args={"check_same_thread": False, "timeout": 60},
                echo=False,
                pool_pre_ping=True,
            )
            self._enable_wal_mode()
        
        self._ensure_table()
    
    def _enable_wal_mode(self):
        """Enable WAL mode for better concurrency (SQLite only)"""
        if self.is_postgresql:
            return  # WAL mode is SQLite-specific
        try:
            with self.engine.begin() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA busy_timeout=10000;"))
        except Exception:
            pass  # Ignore if already in WAL mode or can't set it

    def _determine_db_path(self, override_path: str | None = None):
        """
        Reuse the same SQLite database resolution logic as ApiBuyerFinder.
        """
        if override_path:
            db_filename = os.path.abspath(override_path)
        else:
            db_filename = os.getenv("SQLITE_DB_FILENAME")

        if db_filename:
            if not os.path.isabs(db_filename):
                db_filename = os.path.abspath(db_filename)
        else:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            
            # Priority 1: Check the manufactures_api location (same as ApiBuyerFinder)
            manufactures_api_path = r"C:\Users\HP\Desktop\manufactures_api\viruj_local.db"
            if os.path.exists(manufactures_api_path):
                db_filename = manufactures_api_path
            else:
                # Priority 2: Look in project root
                preferred_project_db = os.path.join(project_root, "viruj_local.db")
                if os.path.exists(preferred_project_db):
                    db_filename = preferred_project_db
                else:
                    # Default to manufactures_api location
                    db_filename = manufactures_api_path

        os.makedirs(os.path.dirname(db_filename), exist_ok=True)
        return db_filename

    def _ensure_table(self):
        """Create table with appropriate syntax for PostgreSQL or SQLite"""
        if self.is_postgresql:
            # PostgreSQL syntax
            # PostgreSQL syntax - matches Supabase schema
            # Note: Table already exists in Supabase, so this is just for local dev
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                CONSTRAINT {self.table_name}_api_name_manufacturer_country_key UNIQUE (api_name, manufacturer, country)
            );
            """
        else:
            # SQLite syntax
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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
        
        with self.engine.begin() as conn:
            conn.execute(text(create_sql))

            # Check and add missing columns
            if self.is_postgresql:
                # PostgreSQL: Check columns using information_schema (case-insensitive)
                table_name_lower = self.table_name.lower()
                columns_query = text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND lower(table_name) = :table_name
                    """
                )
                existing_columns = {
                    row[0].lower()
                    for row in conn.execute(columns_query, {"table_name": table_name_lower})
                }

                if "source_url" not in existing_columns:
                    conn.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN source_url TEXT"))
                if "source_name" not in existing_columns:
                    conn.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN source_name TEXT"))
            else:
                # SQLite: Use PRAGMA
                columns = [row[1] for row in conn.execute(text(f"PRAGMA table_info({self.table_name})"))]
                if "source_url" not in columns:
                    conn.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN source_url TEXT"))
                if "source_name" not in columns:
                    conn.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN source_name TEXT"))

    # ---------- Excel Synchronization ----------

    def sync_from_excel(self):
        """
        Attempt to ingest manufacturers from the Excel/CSV file used previously.
        This method is idempotent thanks to INSERT OR IGNORE.
        """
        excel_path = self._find_excel_source()
        if not excel_path:
            return {
                "synced": False,
                "message": "Excel source not found.",
                "source_file": None,
                "added_rows": 0,
            }

        try:
            df = self._read_excel(excel_path)
        except Exception as exc:
            return {
                "synced": False,
                "message": f"Failed to read Excel: {exc}",
                "source_file": excel_path,
                "added_rows": 0,
            }

        if df.empty:
            return {
                "synced": False,
                "message": "Excel source is empty.",
                "source_file": excel_path,
                "added_rows": 0,
            }

        normalized = self._normalize_dataframe(df)
        if normalized.empty:
            return {
                "synced": False,
                "message": "Excel did not contain the expected columns.",
                "source_file": excel_path,
                "added_rows": 0,
            }

        added, _ = self._bulk_insert(normalized, excel_path)
        return {
            "synced": True,
            "message": f"Synchronized {added} rows from Excel.",
            "source_file": excel_path,
            "added_rows": added,
        }

    def _find_excel_source(self):
        """
        Reuse the same search paths as the legacy Excel loader.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(project_root)

        possible_files = [
            os.path.join(parent_dir, "API_Manufacturers_List.csv"),
            os.path.join(parent_dir, "api_manufacturers_list.csv"),
            os.path.join(parent_dir, "api_manufacturers.xlsx"),
            os.path.join(parent_dir, "api_manufacturers.xls"),
            os.path.join(parent_dir, "api_manufacturers.csv"),
            os.path.join(project_root, "API_Manufacturers_List.csv"),
            os.path.join(project_root, "api_manufacturers_list.csv"),
            os.path.join(project_root, "api_manufacturers.xlsx"),
            os.path.join(project_root, "api_manufacturers.xls"),
            os.path.join(project_root, "api_manufacturers.csv"),
            os.path.join(project_root, "manufacturers.xlsx"),
            os.path.join(project_root, "manufacturers.xls"),
            os.path.join(project_root, "manufacturers.csv"),
        ]

        for path in possible_files:
            if os.path.exists(path):
                return path
        return None

    def _read_excel(self, path):
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        rename_map = {}
        for col in df.columns:
            col_clean = col.strip().lower()
            if col_clean in ("apiname", "api name", "api"):
                rename_map[col] = "api_name"
            elif col_clean in ("manufacturer", "manufacturers", "company"):
                rename_map[col] = "manufacturer"
            elif col_clean in ("country", "country name", "region"):
                rename_map[col] = "country"
            elif col_clean == "usdmf":
                rename_map[col] = "usdmf"
            elif col_clean == "cep":
                rename_map[col] = "cep"

        df = df.rename(columns=rename_map)

        required_cols = ["api_name", "manufacturer", "country"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        for optional in ["usdmf", "cep", "source_url", "source_name"]:
            if optional not in df.columns:
                df[optional] = ""

        subset = df[required_cols + ["usdmf", "cep", "source_url", "source_name"]].copy()
        subset = subset.fillna("")
        subset["api_name"] = subset["api_name"].astype(str).str.strip()
        subset["manufacturer"] = subset["manufacturer"].astype(str).str.strip()
        subset["country"] = subset["country"].astype(str).str.strip()
        subset["usdmf"] = subset["usdmf"].astype(str).str.strip()
        subset["cep"] = subset["cep"].astype(str).str.strip()
        subset["source_url"] = subset["source_url"].astype(str).str.strip()
        subset["source_name"] = subset["source_name"].astype(str).str.strip()

        subset = subset[subset["api_name"] != ""]
        subset = subset[subset["manufacturer"] != ""]

        return subset

    def _bulk_insert(self, df: pd.DataFrame, source_file: str):
        if df.empty:
            return 0, []

        # Use PostgreSQL-compatible syntax if using PostgreSQL, otherwise SQLite
        if self.is_postgresql:
            insert_sql = text(
                f"""
                INSERT INTO {self.table_name}
                (api_name, manufacturer, country, usdmf, cep, source_file, imported_at, source_url, source_name)
                VALUES (:api_name, :manufacturer, :country, :usdmf, :cep, :source_file, :imported_at, :source_url, :source_name)
                ON CONFLICT ON CONSTRAINT api_manufacturers_api_name_manufacturer_country_key DO NOTHING;
                """
            )
        else:
            insert_sql = text(
                f"""
                INSERT OR IGNORE INTO {self.table_name}
                (api_name, manufacturer, country, usdmf, cep, source_file, imported_at, source_url, source_name)
                VALUES (:api_name, :manufacturer, :country, :usdmf, :cep, :source_file, :imported_at, :source_url, :source_name);
                """
            )

        rows_to_insert = [
            {
                "api_name": row["api_name"],
                "manufacturer": row["manufacturer"],
                "country": row.get("country", ""),
                "usdmf": row.get("usdmf", ""),
                "cep": row.get("cep", ""),
                "source_file": source_file,
                "imported_at": datetime.utcnow().isoformat(),
                "source_url": row.get("source_url", ""),
                "source_name": row.get("source_name", ""),
            }
            for _, row in df.iterrows()
        ]

        added = 0
        inserted_rows = []
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with self.engine.begin() as conn:
                    for payload in rows_to_insert:
                        result = conn.execute(insert_sql, payload)
                        if result.rowcount:
                            added += result.rowcount
                            inserted_rows.append(
                                {
                                    "api_name": payload["api_name"],
                                    "manufacturer": payload["manufacturer"],
                                    "country": payload["country"],
                                    "usdmf": payload["usdmf"],
                                    "cep": payload["cep"],
                                    "source_file": source_file,
                                    "imported_at": payload["imported_at"],
                                    "source_url": payload.get("source_url", ""),
                                    "source_name": payload.get("source_name", ""),
                                }
                            )
                break  # Success, exit retry loop
            except OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise  # Re-raise if not a lock error or max retries reached

        return added, inserted_rows

    # ---------- Query ----------

    def query(self, api_name: str, country: str):
        """
        Return manufacturers matching the given API & country (case-insensitive substring match).
        """
        api_filter = (api_name or "").strip()
        country_filter = (country or "").strip()

        if not api_filter or not country_filter:
            return []

        # Exclude the first batch of hallucinated rows inserted during initial testing
        bad_import_timestamp = "2025-11-18T08:11:35.863619"

        if self.is_postgresql:
            order_clause = "ORDER BY LOWER(manufacturer)"
        else:
            order_clause = "ORDER BY manufacturer COLLATE NOCASE"

        query_sql = text(
            f"""
            SELECT api_name, manufacturer, country, usdmf, cep, imported_at, source_url, source_name
            FROM {self.table_name}
            WHERE LOWER(api_name) LIKE LOWER(:api_pattern)
              AND LOWER(country) LIKE LOWER(:country_pattern)
              AND (imported_at IS NULL OR imported_at = '' OR imported_at != :bad_ts)
            {order_clause};
            """
        )

        api_pattern = f"%{api_filter}%"
        country_pattern = f"%{country_filter}%"

        with self.engine.begin() as conn:
            rows = conn.execute(
                query_sql,
                {
                    "api_pattern": api_pattern,
                    "country_pattern": country_pattern,
                    "bad_ts": bad_import_timestamp,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_skip_list(self, api_name: str, country: str):
        records = self.query(api_name, country)
        return {rec["manufacturer"].strip().lower() for rec in records}

    def insert_records(self, records, source_label: str = "discovery"):
        if not records:
            return {"inserted": 0, "rows": []}

        df = pd.DataFrame(records)
        expected_cols = {"api_name", "manufacturer", "country", "usdmf", "cep", "source_url", "source_name"}
        missing = expected_cols - set(df.columns)
        for col in missing:
            df[col] = ""

        df = df[["api_name", "manufacturer", "country", "usdmf", "cep", "source_url", "source_name"]]
        df = df.fillna("")
        df["api_name"] = df["api_name"].astype(str).str.strip()
        df["manufacturer"] = df["manufacturer"].astype(str).str.strip()
        df["country"] = df["country"].astype(str).str.strip()
        df["usdmf"] = df["usdmf"].astype(str).str.strip()
        df["cep"] = df["cep"].astype(str).str.strip()
        df["source_url"] = df["source_url"].astype(str).str.strip()
        df["source_name"] = df["source_name"].astype(str).str.strip()
        df = df[(df["api_name"] != "") & (df["manufacturer"] != "")]

        inserted_count, inserted_rows = self._bulk_insert(df, source_label)
        return {"inserted": inserted_count, "rows": inserted_rows}

    def delete_by_source(
        self,
        source_name: str,
        api_name: str | None = None,
        country: str | None = None,
        use_like: bool = False,
    ) -> int:
        """
        Delete rows originating from a specific source (and optionally api/country).
        Returns the number of deleted rows.
        """
        if not source_name:
            return 0

        comparator = "LIKE" if (use_like or "%" in source_name or "_" in source_name) else "="
        clauses = [f"LOWER(source_name) {comparator} LOWER(:source_name)"]
        params = {"source_name": source_name}

        if api_name:
            clauses.append("LOWER(api_name) = LOWER(:api_name)")
            params["api_name"] = api_name
        if country:
            clauses.append("LOWER(country) = LOWER(:country)")
            params["country"] = country

        where_sql = " AND ".join(clauses)
        delete_sql = text(f"DELETE FROM {self.table_name} WHERE {where_sql}")

        with self.engine.begin() as conn:
            result = conn.execute(delete_sql, params)
            return result.rowcount

