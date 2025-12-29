"""
CLI helper to run the user's Groq/DuckDuckGo/Crawl4AI workflow for the
New Manufacturers tab, but persist discoveries directly into the same table
the app uses (Supabase via DATABASE_URL).

Usage:
    python manual_manufacturer_discovery.py <api_name> <country> [--csv ./path.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Dict

import pandas as pd
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

from synthesis_route_finder.synthesis_engine.tools import Crawl4aiTools
from sqlalchemy import create_engine, text

# Ensure env vars are loaded (works both locally and on Railway)
load_dotenv()


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Environment variable {var_name} is required for this script.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover manufacturers and push to Supabase/Postgres.")
    parser.add_argument("api_name", help="API name to search for (case-insensitive).")
    parser.add_argument("country", help="Country to focus on (case-insensitive).")
    parser.add_argument(
        "--csv-path",
        default=os.getenv("MANUFACTURER_CSV_PATH", "./API_Manufacturers_List.csv"),
        help="Seed CSV file path (default: %(default)s).",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("MANUFACTURER_TABLE_NAME", "API_manufacturers"),
        help="Target table in Supabase/Postgres.",
    )
    return parser.parse_args()


def load_seed_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="latin1")
        for col in ("apiname", "country", "manufacturers"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        return df
    except Exception as exc:
        print(f"⚠️ CSV Load Failed ({csv_path}): {exc}")
        return pd.DataFrame(columns=["apiname", "manufacturers", "country", "usdmf", "cep"])


def extract_manufacturers(markdown_output: str, skip_set: set[str], api_name: str, country_filter: str) -> List[Dict]:
    rows = []
    if not markdown_output:
        return rows

    for raw_line in markdown_output.splitlines():
        if "|" not in raw_line or raw_line.lower().startswith("| manufacturers") or raw_line.startswith("|---"):
            continue

        parts = [p.strip() for p in raw_line.split("|") if p.strip()]
        if len(parts) < 4:
            continue

        manu_name = parts[0]
        country = parts[1].lower()
        usdmf_status = "Yes" if parts[2].lower() in ("yes", "y", "t") else "No"
        cep_status = "Yes" if parts[3].lower() in ("yes", "y", "t") else "No"

        if manu_name.lower() in skip_set:
            continue
        if country_filter not in country:
            continue

        rows.append(
            {
                "api_name": api_name,
                "manufacturer": manu_name,
                "country": country,
                "usdmf": usdmf_status,
                "cep": cep_status,
                "source_name": "GroqAgent",
                "source_url": "N/A",
            }
        )

    return rows


def create_agent(role: str, instructions: str, tools: list) -> Agent:
    groq_model = Groq(id=os.getenv("GROQ_MODEL_ID", "qwen-2.5-32b"))
    return Agent(
        name=role,
        role=role,
        model=groq_model,
        tools=tools,
        instructions=instructions,
        show_tool_calls=True,
        markdown=True,
    )


def run_agents(api_name: str, country: str, existing: List[str]) -> List[Dict]:
    skip_clause = ", ".join(existing) or "None"
    instructions = f"""
Search FDA Orange Book, Pharmaoffer, Pharmacompass for manufacturers of {api_name} API in {country}.
Avoid listing known manufacturers: {skip_clause}.
Return ONLY a Markdown table with columns:
| manufacturers | country | usdmf | cep |
"""

    batches = [existing[i : i + 30] for i in range(0, len(existing), 30)] or [[]]
    discovered: List[Dict] = []

    for batch in batches:
        batch_skip = set(batch)
        try:
            web_agent = create_agent(
                role="Web Agent",
                instructions=instructions,
                tools=[DuckDuckGoTools()],
            )
            result = web_agent.run(f"Find {api_name} API manufacturers in {country}.")
            if result and result.content:
                discovered.extend(extract_manufacturers(result.content, batch_skip, api_name, country))
            time.sleep(1)
        except Exception as exc:
            print(f"⚠️ Web Agent failed: {exc}")

        try:
            pharma_agent = create_agent(
                role="Pharma Agent",
                instructions=instructions,
                tools=[Crawl4aiTools()],
            )
            result = pharma_agent.run(f"Crawl for {api_name} API manufacturers in {country}.")
            if result and result.content:
                discovered.extend(extract_manufacturers(result.content, batch_skip, api_name, country))
            time.sleep(1)
        except Exception as exc:
            print(f"⚠️ Pharma Agent failed: {exc}")

        if discovered:
            break  # stop once we have fresh rows

    return discovered


def insert_into_supabase(table_name: str, rows: List[Dict]):
    database_url = _require_env("DATABASE_URL")
    engine = create_engine(database_url, pool_pre_ping=True)

    insert_sql = text(
        f"""
        INSERT INTO {table_name} (api_name, manufacturer, country, usdmf, cep, source_name, source_url, imported_at)
        VALUES (:api_name, :manufacturer, :country, :usdmf, :cep, :source_name, :source_url, :imported_at)
        ON CONFLICT (api_name, manufacturer, country) DO NOTHING;
        """
    )

    now_iso = pd.Timestamp.utcnow().isoformat()
    with engine.begin() as conn:
        for row in rows:
            payload = dict(row)
            payload.setdefault("source_name", "GroqAgent")
            payload.setdefault("source_url", "N/A")
            payload["imported_at"] = now_iso
            conn.execute(insert_sql, payload)


def main():
    args = parse_args()
    api_name = args.api_name.strip().lower()
    country = args.country.strip().lower()

    if not api_name or not country:
        print("❌ API name and country are required.")
        sys.exit(1)

    print(f"🔍 Looking up manufacturers for '{api_name}' in '{country}'")

    seed_df = load_seed_dataframe(args.csv_path)
    existing_df = seed_df[
        (seed_df.get("apiname", "").str.contains(api_name, na=False))
        & (seed_df.get("country", "").str.contains(country, na=False))
    ]
    skip_list = sorted(existing_df.get("manufacturers", pd.Series(dtype=str)).dropna().unique())
    skip_set = {item.strip().lower() for item in skip_list}

    new_rows = run_agents(api_name, country, list(skip_set))
    if not new_rows:
        print(f"❌ No manufacturers found for '{api_name}' in '{country}'.")
        sys.exit(0)

    print(f"✅ Discovered {len(new_rows)} candidate manufacturers. Inserting into Supabase...")
    insert_into_supabase(args.table, new_rows)
    print("🎉 Done! Rows inserted (duplicates skipped).")


if __name__ == "__main__":
    main()

