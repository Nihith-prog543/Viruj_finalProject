import os
import sys
import time
import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

from .tools import Crawl4aiTools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load environment variables ===
load_dotenv()

# Groq API Key - must be set via environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠ GROQ_API_KEY is not set in environment variables!")

# Supabase (PostgreSQL) connection string
SUPABASE_DB_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    print("❌ Missing DATABASE_URL (Supabase connection string) in environment.")
    sys.exit(1)

CSV_DATA_PATH = os.getenv("MANUFACTURERS_CSV_PATH", r"/Users/prabhas/Desktop/prabhas5.csv")
# Note: If you encounter "tool_use_failed" errors, try switching to a different Groq model
# that better supports function calling, such as "llama-3.1-70b-versatile" or "mixtral-8x7b-32768"
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")


def load_dataset() -> pd.DataFrame:
    try:
        logger.info(f"📂 Loading dataset from: {CSV_DATA_PATH}")
        df = pd.read_csv(CSV_DATA_PATH, encoding="latin1")
        df["apiname"] = df["apiname"].str.strip().str.lower()
        df["country"] = df["country"].str.strip().str.lower()
        logger.info(f"✅ Loaded {len(df)} records from CSV")
        return df
    except Exception as exc:
        logger.warning(f"⚠️ CSV Load Failed: {exc}")
        logger.warning(f"⚠️ Using empty dataset. Set MANUFACTURERS_CSV_PATH environment variable if you have a CSV file.")
        return pd.DataFrame(columns=["apiname", "manufacturers", "country", "usdmf", "cep"])


# === Helper: Extract Manufacturer Info from Markdown ===
def extract_manufacturers(markdown_output, api_name, country_input, existing_manufacturers):
    manufacturers = []
    if not markdown_output:
        return manufacturers

    lines = markdown_output.splitlines()
    for line in lines:
        if "|" in line and not line.lower().startswith("| manufacturers") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5:
                manu_name = parts[0]
                country = parts[1].lower()
                usdmf_status = "Yes" if parts[2].strip().lower() in ["yes", "t"] else "No"
                cep_status = "Yes" if parts[3].strip().lower() in ["yes", "t"] else "No"
                source = parts[4]

                if manu_name.lower() not in existing_manufacturers and (country_input in country):
                    manufacturers.append(
                        {
                            "apiname": api_name,
                            "manufacturers": manu_name,
                            "country": country,
                            "usdmf": usdmf_status,
                            "cep": cep_status,
                            "source": source,
                        }
                    )

    return manufacturers


def create_pharma_agent(api_name, country_input, skip_list):
    skip_clause = ", ".join(skip_list) if skip_list else "None"
    return Agent(
        name="Pharma Agent",
        role="Crawl pharma directories & FDA Orange Book for new manufacturers of API.",
        model=Groq(id=GROQ_MODEL_ID, api_key=GROQ_API_KEY),
        tools=[Crawl4aiTools()],
        instructions=f"""
            You are a pharmaceutical research agent. Your task is to find API manufacturers from regulatory sources.
            
            IMPORTANT TOOL USAGE:
            - You have access to a web_crawler tool that can crawl regulatory websites
            - When using the web_crawler tool, call it with proper JSON format: {{"url": "https://example.com", "max_length": 10000}}
            - DO NOT use XML-like syntax or malformed function calls
            - The tool accepts two parameters: url (required) and max_length (optional, defaults to 2000)
            
            TASK:
            Crawl FDA Orange Book, EMA, PMDA, DCGI, MHRA, and other regulator-backed sources for {api_name} API in {country_input}.
            Skip known manufacturers: {skip_clause}.
            
            REGULATORY SOURCES TO CHECK:
            - FDA Orange Book: https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book
            - EMA: https://www.ema.europa.eu/
            - PMDA: https://www.pmda.go.jp/
            - DCGI/CDSCO: https://cdsco.gov.in/
            - MHRA: https://www.mhra.gov.uk/
            - Health Canada: https://www.canada.ca/en/health-canada.html
            
            OUTPUT FORMAT:
            Output strictly in Markdown table format:
            | manufacturers | country | usdmf | cep | source |
            |---------------|---------|-------|-----|--------|
            
            Where:
            - manufacturers: Company name
            - country: Country of manufacturer
            - usdmf: Yes/No for US DMF status
            - cep: Yes/No for CEP status
            - source: Regulator acronym (FDA, EMA, PMDA, DCGI, MHRA, Health Canada)
            
            Exclude any manufacturer you cannot tie to one of those regulators.
        """,
        show_tool_calls=True,
        markdown=True,
    )


def run_discovery(
    api_name: str,
    country_input: str,
    persist: bool = True,
    existing_manufacturers: Optional[Set[str]] = None,
) -> Dict[str, object]:
    api_name = api_name.strip().lower()
    country_input = country_input.strip().lower()
    if not api_name or not country_input:
        raise ValueError("API name and country are required.")

    logger.info(f"🔍 Looking up: API = '{api_name}' | Country = '{country_input}'")

    df = load_dataset()
    existing_df = df[
        (df["apiname"].str.contains(api_name, na=False))
        & (df["country"].str.contains(country_input, na=False))
    ]
    csv_existing_manufacturers = set(
        existing_df["manufacturers"].dropna().str.strip().str.lower().unique()
    )

    if existing_manufacturers is None:
        existing_manufacturers = csv_existing_manufacturers
    else:
        existing_manufacturers = {name.strip().lower() for name in existing_manufacturers if name}

    logger.info(f"📊 Found {len(existing_manufacturers)} existing manufacturers to skip")

    batch_size = 30
    existing_list = sorted(existing_manufacturers)
    batches = [existing_list[i : i + batch_size] for i in range(0, len(existing_list), batch_size)] or [[]]

    logger.info(f"🚀 Starting discovery with {len(batches)} batch(es)")
    pharma_rows: List[Dict[str, str]] = []

    for batch_idx, skip_batch in enumerate(batches, 1):
        max_retries = 3
        retry_delay = 2
        
        for retry_attempt in range(max_retries):
            try:
                logger.info(f"🤖 Running Pharma Agent batch {batch_idx}/{len(batches)} (skipping {len(skip_batch)} manufacturers)... [Attempt {retry_attempt + 1}/{max_retries}]")
                pharma_agent = create_pharma_agent(api_name, country_input, skip_batch)
                logger.info(f"🔍 Agent created. Starting crawl for {api_name} in {country_input}...")
                
                pharma_result = pharma_agent.run(f"Crawl for {api_name} API manufacturers in {country_input}.")
                
                if pharma_result:
                    logger.info(f"✅ Agent returned result. Content length: {len(pharma_result.content) if pharma_result.content else 0} chars")
                    extracted = extract_manufacturers(
                        pharma_result.content, api_name, country_input, existing_manufacturers
                    )
                    logger.info(f"📋 Extracted {len(extracted)} manufacturers from batch {batch_idx}")
                    for row in extracted:
                        row["source"] = row.get("source") or "regulator"
                    pharma_rows.extend(extracted)
                    break  # Success, exit retry loop
                else:
                    logger.warning(f"⚠️ Agent returned no result for batch {batch_idx}")
                    break  # No result but no error, exit retry loop
                    
            except Exception as exc:
                error_str = str(exc)
                is_tool_use_error = "tool_use_failed" in error_str or "failed_generation" in error_str
                
                if is_tool_use_error and retry_attempt < max_retries - 1:
                    logger.warning(f"⚠️ Tool use error detected (attempt {retry_attempt + 1}/{max_retries}): {error_str[:200]}")
                    logger.info(f"🔄 Retrying with delay of {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Pharma Agent failed for batch {batch_idx}: {exc}")
                    if is_tool_use_error:
                        logger.error("❌ This appears to be a Groq API function calling issue. The model may be generating malformed function calls.")
                        logger.error("💡 Suggestions:")
                        logger.error("   1. Try using a different Groq model")
                        logger.error("   2. Check if the Crawl4aiTools are properly configured")
                        logger.error("   3. The model may need more explicit instructions about tool usage")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    break  # Exit retry loop on final failure
                    
        time.sleep(2)  # Delay between batches

    combined_scraped_rows = pharma_rows

    logger.info(f"📊 Total manufacturers found: {len(combined_scraped_rows)}")

    if not combined_scraped_rows:
        message = f"❌ No API manufacturers found for '{api_name}' in '{country_input}'."
        logger.warning(message)
        logger.warning("💡 This could mean:")
        logger.warning("   1. The agent didn't find any new manufacturers")
        logger.warning("   2. The agent encountered errors (check logs above)")
        logger.warning("   3. All found manufacturers were already in the skip list")
        return {"success": False, "message": message, "new_records": []}

    # Note: Data insertion is handled by ApiManufacturerDiscoveryService.discover()
    # which uses manufacturer_service.insert_records() to store in Supabase
    # This ensures all data goes to the API_manufacturers table in Supabase
    
    return {
        "success": True,
        "new_records": combined_scraped_rows,
        "inserted_count": len(combined_scraped_rows),
        "pharma_records": pharma_rows,
    }


# === Insert into Supabase PostgreSQL ===
def insert_into_supabase(fresh_df):
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)

        cursor = conn.cursor()
        combined_df = fresh_df.drop_duplicates()

        insert_query = """
        INSERT INTO manufacturers (apiname, manufacturers, country, usdmf, cep)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (apiname,manufacturers, country) DO NOTHING;
        """

        for _, row in combined_df.iterrows():
            cursor.execute(
                insert_query,
                (row["apiname"], row["manufacturers"], row["country"], row["usdmf"], row["cep"]),
            )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Data inserted into Supabase successfully! ({len(combined_df)} records)")
    except Exception as e:
        logger.error(f"❌ Error inserting into Supabase: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")


class ApiManufacturerDiscoveryService:
    """
    Wraps the discovery workflow so it can be reused inside the Flask app.
    Normalizes newly scraped records and stores them through ApiManufacturerService.
    """

    def __init__(self, manufacturer_service=None, source_label: str = "web_discovery"):
        self.manufacturer_service = manufacturer_service
        self.source_label = source_label

    def _get_skip_set(self, api_name: str, country: str) -> Tuple[List[Dict[str, str]], Set[str]]:
        if not self.manufacturer_service:
            return [], set()
        existing_records = self.manufacturer_service.query(api_name, country)
        skip = {
            (record.get("manufacturer") or "").strip().lower()
            for record in existing_records
            if record.get("manufacturer")
        }
        return existing_records, skip

    def discover(self, api_name: str, country: str) -> Dict[str, object]:
        api_name = (api_name or "").strip()
        country = (country or "").strip()

        if not api_name or not country:
            return {"success": False, "error": "API name and country are required"}

        existing_records, skip_set = self._get_skip_set(api_name, country)
        
        logger.info(f"📊 Found {len(existing_records)} existing records in Supabase")
        logger.info(f"🚫 Skipping {len(skip_set)} known manufacturers")

        discovery_result = run_discovery(
            api_name,
            country,
            persist=False,  # Don't persist here - we'll use the service to insert into Supabase
            existing_manufacturers=skip_set if skip_set else None,
        )

        if not discovery_result.get("success"):
            return discovery_result

        pharma_rows = discovery_result.get("pharma_records", [])
        normalized_rows = []
        for row in pharma_rows:
            manufacturer_name = row.get("manufacturers") or row.get("manufacturer")
            if not manufacturer_name:
                continue
            regulator_source = row.get("source") or self.source_label
            normalized_rows.append(
                {
                    "api_name": row.get("apiname", api_name),
                    "manufacturer": manufacturer_name,
                    "country": row.get("country", country),
                    "usdmf": row.get("usdmf", ""),
                    "cep": row.get("cep", ""),
                    "source_name": regulator_source,
                    "source_url": row.get("source_url", ""),
                }
            )

        inserted_rows = []
        inserted_count = 0
        if self.manufacturer_service:
            logger.info(f"💾 Inserting {len(normalized_rows)} new manufacturer records into Supabase via manufacturer_service...")
            insert_result = self.manufacturer_service.insert_records(
                normalized_rows, source_label=self.source_label
            )
            inserted_rows = insert_result.get("rows", [])
            inserted_count = insert_result.get("inserted", 0)
            logger.info(f"✅ Successfully inserted {inserted_count} new records into Supabase")
            if inserted_count < len(normalized_rows):
                logger.warning(f"⚠️ Only {inserted_count} of {len(normalized_rows)} records were inserted (some may be duplicates)")
        else:
            logger.warning("⚠️ No manufacturer_service configured - records will not be saved to database")
            inserted_rows = normalized_rows
            inserted_count = len(inserted_rows)

        all_records = existing_records + inserted_rows

        return {
            "success": True,
            "existing_records": existing_records,
            "new_records": inserted_rows,
            "all_records": all_records,
            "inserted_count": inserted_count,
        }

    def purge_discovery_results(
        self,
        source_name: str,
        api_name: Optional[str] = None,
        country: Optional[str] = None,
        use_like: bool = True,
    ) -> Dict[str, object]:
        if not self.manufacturer_service:
            return {"success": False, "error": "Manufacturer service not configured"}

        deleted = self.manufacturer_service.delete_by_source(
            source_name=source_name,
            api_name=api_name,
            country=country,
            use_like=use_like,
        )
        return {"success": True, "deleted": deleted}


def main():
    if len(sys.argv) < 3:
        print("Usage: python api_manufacturer_discovery.py <api_name> <country>")
        sys.exit(1)

    _, api_arg, country_arg = sys.argv[:3]
    result = run_discovery(api_arg, country_arg)
    if not result.get("success"):
        sys.exit(0)


if __name__ == "__main__":
    main()

