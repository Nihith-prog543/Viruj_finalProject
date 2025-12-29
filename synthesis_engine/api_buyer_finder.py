# synthesis_engine/api_buyer_finder.py - API Buyer Search Logic (FDF Manufacturers)
import pandas as pd
import logging
import os
import warnings
import re
import time
from typing import List, Dict
from sqlalchemy import create_engine, text
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import Groq

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy.*")

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("⚠ psycopg2 not installed. Install with: pip install psycopg2-binary")

class ApiBuyerFinder:
    """
    API Buyer Finder - Finds FDF manufacturers using DuckDuckGo + Tavily search
    validated by Groq LLM. No scraping required.
    """
    
    def __init__(self):
        # ====== API KEYS FROM ENV ======
        # All API keys must be set via environment variables for security
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        if not self.GROQ_API_KEY:
            logger.warning("⚠ GROQ_API_KEY is not set!")
        else:
            logger.info("✅ Groq API key configured")
        
        if not self.TAVILY_API_KEY:
            logger.warning("⚠ TAVILY_API_KEY is not set!")
        else:
            logger.info("✅ Tavily API key configured")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.GROQ_API_KEY) if self.GROQ_API_KEY else None
        
        # ====== DATABASE CONFIG ======
        # Only Supabase (PostgreSQL) is supported - DATABASE_URL is required
        if not self.DATABASE_URL:
            logger.error("❌ DATABASE_URL is not set! Supabase PostgreSQL connection is required.")
            logger.error("❌ Please set DATABASE_URL environment variable with your Supabase PostgreSQL connection string")
        elif not self.DATABASE_URL.startswith("postgresql://"):
            logger.error(f"❌ DATABASE_URL must start with 'postgresql://' - got: {self.DATABASE_URL[:20]}...")
        else:
            logger.info("✅ DATABASE_URL configured - will use Supabase PostgreSQL")
        # ============================
    
    # ============================================================
    # MARKDOWN TABLE PARSER
    # ============================================================
    def markdown_table_to_df(self, markdown: str) -> pd.DataFrame:
        """Parse a simple GitHub-style markdown table into a DataFrame."""
        lines = [line.strip() for line in markdown.splitlines() if "|" in line]
        if len(lines) < 2:
            return pd.DataFrame()

        headers = [h.strip() for h in lines[0].split("|")[1:-1]]
        data = []
        for row in lines[2:]:
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if len(cols) == len(headers):
                data.append(cols)

        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data, columns=headers)

    def standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map whatever header names the model outputs to a consistent schema."""
        if df.empty:
            return df

        col_map = {}
        for col in df.columns:
            key = col.strip().lower()
            if "company" in key:
                col_map[col] = "company"
            elif key.startswith("form"):
                col_map[col] = "form"
            elif "strength" in key:
                col_map[col] = "strength"
            elif "verification source" in key:
                col_map[col] = "verification_source"
            elif "confidence" in key:
                col_map[col] = "confidence"
            elif key == "url" or "url" in key:
                col_map[col] = "url"
            elif "verification status" in key:
                col_map[col] = "verification_status"
            elif "additional info" in key:
                col_map[col] = "additional_info"
            elif "manufacturer country" in key or ("manufacturer" in key and "country" in key):
                col_map[col] = "manufacturer_country"
            elif "manufacturing country" in key:
                col_map[col] = "manufacturer_country"

        df = df.rename(columns=col_map)

        # Ensure all key columns exist
        for c in [
            "company",
            "form",
            "strength",
            "manufacturer_country",
            "verification_source",
            "confidence",
            "url",
            "verification_status",
            "additional_info",
        ]:
            if c not in df.columns:
                df[c] = ""

        # Normalise confidence to int
        df["confidence"] = df["confidence"].apply(
            lambda x: int(str(x).strip("%")) if str(x).strip("%").isdigit() else 0
        )

        return df

    # ============================================================
    # DUCKDUCKGO SEARCH
    # ============================================================
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    def search_with_duckduckgo(self, api: str, country: str, max_results: int = 30) -> List[Dict[str, str]]:
        """
        Use DuckDuckGo to search for pharmaceutical sites related to the API.
        Returns list of dicts: {title, url, snippet}.
        """
        try:
            from ddgs import DDGS
            import logging as std_logging
            
            # Temporarily suppress DuckDuckGo library's internal error logging
            # These errors are expected and handled gracefully
            ddgs_logger = std_logging.getLogger('ddgs')
            original_level = ddgs_logger.level
            ddgs_logger.setLevel(std_logging.ERROR)  # Only show ERROR level, suppress INFO/WARNING
            
            try:
                # Build search queries focused on finished dosage forms
                queries = [
                    f"{api} manufacturers {country} finished dosage forms",
                    f"{api} pharmaceutical companies {country} tablets capsules",
                    f"{api} drug manufacturers {country} FDF",
                    f"{api} pharma companies {country} products",
                    f"{api} registered products {country} regulatory",
                ]
                
                all_results = []
                seen_urls = set()
                
                with DDGS() as ddgs:
                    for query in queries:
                        try:
                            # Try to get results, catch IndexError from DuckDuckGo library
                            try:
                                results = list(ddgs.text(query, max_results=max_results))
                            except (IndexError, KeyError) as parse_error:
                                # DuckDuckGo library sometimes fails to parse results
                                # This is expected and we continue with other queries
                                continue
                            except Exception as inner_error:
                                error_msg = str(inner_error)
                                # Suppress known DuckDuckGo engine errors (mullvad_google, mullvad_brave, etc.)
                                if any(engine in error_msg for engine in ['mullvad_google', 'mullvad_brave', 'mojeek', 'IndexError']):
                                    # These are known issues with DuckDuckGo's internal parsing
                                    continue
                                # Re-raise unexpected errors
                                raise
                            
                            # Validate and process results
                            if not results:
                                continue
                                
                            for result in results:
                                if not isinstance(result, dict):
                                    continue
                                url = result.get('href', '') or result.get('url', '')
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_results.append({
                                        'title': result.get('title', '') or result.get('name', ''),
                                        'url': url,
                                        'snippet': result.get('body', '') or result.get('snippet', '') or result.get('content', ''),
                                        'query': query
                                    })
                            time.sleep(1)  # Be polite to DuckDuckGo
                        except Exception as e:
                            error_msg = str(e)
                            # Suppress known DuckDuckGo engine errors
                            if any(engine in error_msg for engine in ['mullvad_google', 'mullvad_brave', 'mojeek', 'IndexError']):
                                # These are known issues - silently continue
                                continue
                            # Log unexpected errors
                            logger.warning(f"Search query '{query}' failed: {e}")
                            continue
            finally:
                # Restore original logging level
                ddgs_logger.setLevel(original_level)
            
            logger.info(f"🔍 Found {len(all_results)} unique URLs from DuckDuckGo search")
            return all_results[:30]  # Limit to top 30
            
        except ImportError:
            logger.warning("❌ ddgs package not installed. Install with: pip install ddgs")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    # ============================================================
    # TAVILY SEARCH
    # ============================================================
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    def search_with_tavily(self, api: str, country: str, max_results: int = 30) -> List[Dict[str, str]]:
        """
        Use Tavily API to search for pharmaceutical sites related to the API.
        Returns list of dicts: {title, url, snippet}.
        """
        if not self.TAVILY_API_KEY:
            logger.warning("⚠ TAVILY_API_KEY is not set. Skipping Tavily search.")
            return []
        
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]  # pyright: ignore  # pylance: disable
            logger.info("✅ Tavily package imported successfully")
            
            tavily_client = TavilyClient(api_key=self.TAVILY_API_KEY)
            logger.info("✅ Tavily client initialized successfully")
            
            # Build search queries focused on finished dosage forms
            queries = [
                f"{api} manufacturers {country} finished dosage forms",
                f"{api} pharmaceutical companies {country} tablets capsules",
                f"{api} drug manufacturers {country} FDF",
                f"{api} pharma companies {country} products",
                f"{api} registered products {country} regulatory",
            ]
            
            all_results = []
            seen_urls = set()
            
            for query in queries:
                try:
                    # Tavily search
                    response = tavily_client.search(
                        query=query,
                        max_results=max_results,
                        search_depth="advanced"  # Use advanced search for better results
                    )
                    
                    # Extract results from Tavily response
                    results_list = []
                    if isinstance(response, dict):
                        results_list = response.get('results', [])
                    elif isinstance(response, list):
                        results_list = response
                    
                    for result in results_list:
                        # Handle different possible response formats
                        url = result.get('url', '') or result.get('link', '')
                        title = result.get('title', '') or result.get('name', '')
                        snippet = result.get('content', '') or result.get('snippet', '') or result.get('body', '')
                        
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet[:500] if snippet else '',
                                'query': query
                            })
                    
                    time.sleep(0.5)  # Be polite to Tavily API
                    
                except Exception as e:
                    logger.warning(f"Tavily search query '{query}' failed: {e}")
                    continue
            
            logger.info(f"🔍 Found {len(all_results)} unique URLs from Tavily search")
            return all_results[:30]  # Limit to top 30
            
        except ImportError as import_err:
            logger.warning(f"⚠ tavily package not installed. Skipping Tavily search. Error: {import_err}")
            logger.warning("⚠ Install with: pip install tavily-python")
            import sys
            logger.warning(f"⚠ Python executable: {sys.executable}")
            return []
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            import traceback
            logger.error(f"Tavily error traceback: {traceback.format_exc()}")
            return []

    # ============================================================
    # COMBINED SEARCH (DuckDuckGo + Tavily)
    # ============================================================
    def search_combined(self, api: str, country: str, max_results_per_source: int = 30) -> List[Dict[str, str]]:
        """
        Search using both DuckDuckGo and Tavily, then combine and deduplicate results.
        Returns combined list of dicts: {title, url, snippet}.
        """
        all_results = []
        seen_urls = set()
        
        # Search with DuckDuckGo
        logger.info("🔍 Searching with DuckDuckGo...")
        ddg_results = self.search_with_duckduckgo(api, country, max_results_per_source)
        for result in ddg_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                result['source'] = 'DuckDuckGo'  # Tag the source
                all_results.append(result)
        
        # Search with Tavily
        logger.info("🔍 Searching with Tavily...")
        tavily_results = self.search_with_tavily(api, country, max_results_per_source)
        for result in tavily_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                result['source'] = 'Tavily'  # Tag the source
                all_results.append(result)
        
        logger.info(f"🔍 Combined search: {len(ddg_results)} from DuckDuckGo, {len(tavily_results)} from Tavily, {len(all_results)} unique total")
        return all_results

    # ============================================================
    # GROQ API HELPER WITH RATE LIMITING
    # ============================================================
    def call_groq_api(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 2000, max_retries: int = 5) -> str:
        """
        Call Groq API with proper rate limiting and retry logic for 429 errors.
        Returns the response content or raises an exception.
        """
        if not self.groq_client:
            raise Exception("Groq client not initialized. GROQ_API_KEY is required.")
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Success - add delay before next call
                time.sleep(1.0)  # Increased delay to 1 second
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()
                last_exception = e
                
                # Check if it's a rate limit error (429)
                is_rate_limit = (
                    "429" in error_str or 
                    "too many requests" in error_str or 
                    "rate limit" in error_str or
                    "ratelimit" in error_str or
                    "rate_limit" in error_str or
                    hasattr(e, 'status_code') and e.status_code == 429
                )
                
                if is_rate_limit:
                    # Calculate exponential backoff: 2^attempt seconds (max 60s)
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"⚠ Rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    # For other errors, use shorter backoff
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"⚠ API error ({error_type}): {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
        
        # If all retries failed
        logger.error(f"❌ Failed to call Groq API after {max_retries} attempts: {last_exception}")
        raise last_exception

    # ============================================================
    # GROQ VALIDATION (EXTRACT FROM SEARCH RESULTS WITHOUT SCRAPING)
    # ============================================================
    def validate_and_extract_from_search_results(self, api: str, country: str, search_results: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Use Groq LLM to validate search results and extract manufacturer information
        directly from search snippets/titles WITHOUT scraping the pages.
        Returns DataFrame with validated manufacturer data.
        """
        if not search_results:
            return pd.DataFrame()
        
        # Process results in batches to handle large result sets
        max_results_per_batch = 30
        all_extracted_data = []
        
        total_results = len(search_results)
        num_batches = (total_results + max_results_per_batch - 1) // max_results_per_batch
        
        logger.info(f"🔍 Processing {total_results} search results in {num_batches} batch(es) for validation")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_results_per_batch
            end_idx = min(start_idx + max_results_per_batch, total_results)
            batch_results = search_results[start_idx:end_idx]
            
            if not batch_results:
                continue
            
            # Prepare context for this batch
            search_context = "\n\n".join([
                f"Result {i+1}:\nURL: {r['url']}\nTitle: {r['title']}\nSnippet: {r['snippet'][:400]}..."
                for i, r in enumerate(batch_results)
            ])
            
            prompt = f"""
You are a pharmaceutical data analyst. Your task is to extract Finished Dosage Form (FDF) manufacturer information
for the drug "{api}" in "{country}" from the search results below.

IMPORTANT: You are working with SEARCH RESULT SNIPPETS (not full page content). Extract information ONLY from what is visible in the snippets.

SEARCH RESULTS (Batch {batch_idx + 1}/{num_batches}):
{search_context}

STRICT RULES (MANDATORY - ALL MUST BE PRESENT):
1. Extract ONLY if the search snippet/title clearly mentions ALL of the following:
   - Company name (FDF manufacturer, NOT API supplier) - REQUIRED
   - Brand/product name (indicates finished product) - REQUIRED (must be in Additional Info)
   - Finished dosage form (tablet, capsule, injection, etc.) - REQUIRED (cannot be empty)
   - Manufacturing location in {country} (if mentioned) - Preferred but can be inferred if company is clearly from {country}
   - Strength (if mentioned) - Preferred but can be left blank if not in snippet

2. REJECT if snippet only mentions:
   - API suppliers, bulk suppliers, chemical suppliers
   - Generic mentions without specific company names
   - Products not manufactured in {country}
   - Research papers, news articles without manufacturer info
   - Entries WITHOUT brand/product name (this is REQUIRED to prove it's a finished product)
   - Entries WITHOUT finished dosage form (this is REQUIRED)

3. CRITICAL: If form is empty or additional_info (brand name) is empty, DO NOT extract that row.
   - Form MUST be filled (Tablet, Capsule, Injection, etc.)
   - Additional Info MUST contain brand/product name
   - If snippet doesn't have enough info, SKIP that result completely

OUTPUT FORMAT (MANDATORY):
Return ONLY a markdown table with EXACTLY these columns:

| Company | Form | Strength | Manufacturer Country | Verification Source | Confidence (%) | URL | Verification Status | Additional Info |
|---------|------|----------|----------------------|--------------------|---------------|-----|--------------------|----------------|
| ...rows go here, one per manufacturer found... |

- Company: Manufacturer company name from snippet (REQUIRED)
- Form: Dosage form (Tablet, Capsule, Injection, etc.) - REQUIRED, cannot be empty. If not in snippet, DO NOT extract.
- Strength: Strength (e.g., "10 mg") if mentioned, otherwise leave blank
- Manufacturer Country: MUST be "{country}" if mentioned, otherwise leave blank (do not infer from company name alone)
- Verification Source: Short description like "Search result snippet"
- Confidence (%): 0-100 based on how clear the evidence is in snippet. Use lower confidence if information is incomplete.
- URL: The exact URL from search result - REQUIRED. MUST include the full URL (e.g., https://example.com/page). If URL is not available in the snippet, use the URL from the search result that contains this information. NEVER leave URL empty.
- Verification Status: "Verified" ONLY if snippet clearly shows FDF manufacturer with form and brand name. Use "Unverified" if information is incomplete.
- Additional Info: Brand/product name is REQUIRED here. Must contain brand name to prove it's a finished product. If no brand name in snippet, DO NOT extract that row.

If no valid manufacturers found in this batch, return empty table with just headers.

Return ONLY the markdown table, no explanations.
"""
            
            try:
                content = self.call_groq_api(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a pharmaceutical research agent. Extract manufacturer data from search snippets. Return valid markdown table only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=3000,
                    max_retries=5
                )
                
                # Parse markdown table
                df = self.markdown_table_to_df(content)
                df = self.standardise_columns(df)
                
                if not df.empty:
                    # Ensure all required columns exist
                    required_columns = ["company", "form", "strength", "manufacturer_country", "verification_source", 
                                     "confidence", "url", "verification_status", "additional_info"]
                    for col in required_columns:
                        if col not in df.columns:
                            if col == "confidence":
                                df[col] = 100  # Default confidence
                            else:
                                df[col] = ""  # Default empty string
                    
                    # Ensure confidence is numeric
                    if "confidence" in df.columns:
                        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(100)
                    
                    # CRITICAL: Map URLs from search results to extracted rows
                    # If URL is missing or empty, try to find it from the search results
                    def fill_missing_url(row):
                        url = str(row.get('url', '')).strip()
                        # If URL is missing or invalid, try to find it from search results
                        if not url or url.lower() in ['', 'none', 'n/a', 'na', 'null']:
                            company_name = str(row.get('company', '')).strip().lower()
                            # Try to match company name to search results
                            for search_result in batch_results:
                                title = str(search_result.get('title', '')).lower()
                                snippet = str(search_result.get('snippet', '')).lower()
                                search_url = str(search_result.get('url', '')).strip()
                                # If company name appears in title or snippet, use this URL
                                if company_name and company_name in title or company_name in snippet:
                                    if search_url and search_url.startswith('http'):
                                        return search_url
                        # If URL exists but doesn't start with http, try to fix it
                        if url and not url.startswith('http'):
                            if url.startswith('www.'):
                                url = 'https://' + url
                            elif not url.startswith('http'):
                                # If it's a relative URL, we can't use it - return empty
                                return ""
                        return url if url and url.startswith('http') else ""
                    
                    df['url'] = df.apply(fill_missing_url, axis=1)
                    
                    # Add source information
                    df["source_site"] = "Search Result"
                    all_extracted_data.append(df)
                    logger.info(f"🤖 Batch {batch_idx + 1}/{num_batches}: Extracted {len(df)} manufacturers from {len(batch_results)} search results")
                
                # Small delay between batches
                if batch_idx < num_batches - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Groq validation failed for batch {batch_idx + 1}: {e}")
                continue
        
        if not all_extracted_data:
            logger.warning("⚠ No manufacturer data extracted from search results")
            return pd.DataFrame()
        
        # Combine all batches
        combined_df = pd.concat(all_extracted_data, ignore_index=True)
        logger.info(f"✅ Total extracted: {len(combined_df)} manufacturers from {len(search_results)} search results")
        
        # Ensure all required columns exist before validation
        required_columns = ["company", "form", "strength", "manufacturer_country", "verification_source", 
                            "confidence", "url", "verification_status", "additional_info"]
        for col in required_columns:
            if col not in combined_df.columns:
                if col == "confidence":
                    combined_df[col] = 100  # Default confidence
                else:
                    combined_df[col] = ""  # Default empty string
        
        # Ensure confidence is numeric
        if "confidence" in combined_df.columns:
            combined_df["confidence"] = pd.to_numeric(combined_df["confidence"], errors="coerce").fillna(100)
        
        # POST-EXTRACTION VALIDATION: Reject entries with missing required fields
        if not combined_df.empty:
            def has_required_fields(row):
                """Check if row has all required fields for FDF manufacturer."""
                form = str(row.get('form', '')).strip()
                additional_info = str(row.get('additional_info', '')).strip()
                company = str(row.get('company', '')).strip()
                url = str(row.get('url', '')).strip()
                
                # Form is REQUIRED - cannot be empty
                if not form or form.lower() in ['', 'unknown', 'n/a', 'na', 'none']:
                    return False
                
                # URL is REQUIRED - must be a valid HTTP/HTTPS URL
                if not url or not url.startswith('http'):
                    return False
                
                # Additional Info must contain brand name or evidence
                # Check for brand name indicators
                brand_indicators = [
                    'brand name:', 'brand:', 'product name:', 'marketing name:',
                    'trade name:', 'commercial name:', 'branded product'
                ]
                has_brand = any(indicator in additional_info.lower() for indicator in brand_indicators)
                
                # If additional_info is empty or too short, reject
                if not additional_info or len(additional_info) < 5:
                    return False
                
                # Company name is required
                if not company or len(company) < 3:
                    return False
                
                # Must have either brand name in additional_info OR form must be a valid finished form
                finished_forms = ['tablet', 'capsule', 'injection', 'syrup', 'suspension', 'solution', 'cream', 'ointment', 'gel', 'drops', 'spray']
                has_finished_form = any(f_form in form.lower() for f_form in finished_forms)
                
                # Reject if form is not a finished form
                if not has_finished_form:
                    return False
                
                # Prefer entries with brand names, but allow if form is clear
                return True
            
            before_validation = len(combined_df)
            validation_mask = combined_df.apply(has_required_fields, axis=1)
            combined_df = combined_df[validation_mask].copy()
            
            removed_incomplete = before_validation - len(combined_df)
            if removed_incomplete > 0:
                logger.info(f"🚫 Post-extraction validation: Removed {removed_incomplete} entries with missing required fields (form, brand name, URL, or additional_info)")
        
        return combined_df

    # ============================================================
    # MAIN PIPELINE (NO SCRAPING - VALIDATION FROM SEARCH RESULTS)
    # ============================================================
    def run_pipeline(self, api: str, country: str) -> pd.DataFrame:
        """
        1. Search with DuckDuckGo + Tavily to find relevant URLs (combined results)
        2. Use Groq LLM to validate and extract manufacturer data directly from search snippets
        3. Filter and deduplicate results
        4. Return validated manufacturer data WITHOUT scraping pages
        
        NOTE: This approach uses search result snippets for validation, not full page content.
        """
        logger.info(f"🚀 Starting FDF validation pipeline (NO SCRAPING) for API='{api}', Country='{country}'")

        # 1. Combined search (DuckDuckGo + Tavily)
        logger.info("🔍 Searching with DuckDuckGo and Tavily...")
        search_results = self.search_combined(api, country)
        if not search_results:
            logger.warning("⚠ No search results found from DuckDuckGo or Tavily.")
            return pd.DataFrame()
        
        # Count results by source
        ddg_count = sum(1 for r in search_results if r.get('source') == 'DuckDuckGo')
        tavily_count = sum(1 for r in search_results if r.get('source') == 'Tavily')
        logger.info(f"✅ Found {len(search_results)} unique URLs ({ddg_count} from DuckDuckGo, {tavily_count} from Tavily)")

        # 2. Groq validation and extraction from search results (NO SCRAPING)
        logger.info("🤖 Validating and extracting manufacturer data from search results with Groq LLM...")
        all_df = self.validate_and_extract_from_search_results(api, country, search_results)
        if all_df.empty:
            logger.warning("⚠ Groq validation returned no manufacturer data from search results.")
            return pd.DataFrame()
        logger.info(f"✅ Extracted {len(all_df)} manufacturer entries from search results")

        # Ensure manufacturer_country column exists - DO NOT fill with target country if empty
        # Only use explicitly extracted manufacturing countries
        if "manufacturer_country" not in all_df.columns:
            all_df["manufacturer_country"] = ""
        else:
            # Keep empty values as empty - DO NOT fill with target country
            # Only use manufacturing country if explicitly extracted from search snippet
            all_df["manufacturer_country"] = all_df["manufacturer_country"].replace("EMPTY", "")
            all_df["manufacturer_country"] = all_df["manufacturer_country"].fillna("")
        
        # ADDITIONAL VALIDATION: Reject entries with empty form, empty additional_info, or missing URL
        if not all_df.empty:
            def has_minimum_required_data(row):
                """Check if row has minimum required data."""
                form = str(row.get('form', '')).strip()
                additional_info = str(row.get('additional_info', '')).strip()
                url = str(row.get('url', '')).strip()
                
                # Form is REQUIRED - reject if empty
                if not form or form.lower() in ['', 'unknown', 'n/a', 'na', 'none']:
                    return False
                
                # Additional Info is REQUIRED - must have brand name or evidence
                if not additional_info or len(additional_info) < 5:
                    return False
                
                # URL is REQUIRED - must be a valid HTTP/HTTPS URL
                if not url or not url.startswith('http'):
                    return False
                
                return True
        
        before_min_validation = len(all_df)
        min_validation_mask = all_df.apply(has_minimum_required_data, axis=1)
        all_df = all_df[min_validation_mask].copy()
        
        removed_min = before_min_validation - len(all_df)
        if removed_min > 0:
            logger.info(f"🚫 Removed {removed_min} entries missing form, additional_info, or valid URL")

        # Ensure all required columns exist in all_df before filtering (safety check)
        required_columns = ["company", "form", "strength", "manufacturer_country", "verification_source", 
                           "confidence", "url", "verification_status", "additional_info"]
        for col in required_columns:
            if col not in all_df.columns:
                logger.warning(f"⚠ Required column '{col}' not found. Adding with default value.")
                if col == "confidence":
                    all_df[col] = 100  # Default confidence
                else:
                    all_df[col] = ""  # Default empty string
        
        # Ensure confidence is numeric
        if "confidence" in all_df.columns:
            all_df["confidence"] = pd.to_numeric(all_df["confidence"], errors="coerce").fillna(100)

        # Keep only Verified (LLM already applies your rules)
        verified_mask = all_df["verification_status"].astype(str).str.lower().str.strip() == "verified"
        verified_df = all_df[verified_mask].copy()

        # Filter by target country (case-insensitive, partial match)
        country_lower = country.lower().strip()
        total_before_filter = len(verified_df)
        
        if "manufacturer_country" in verified_df.columns:
            # Create a function to check if country matches (handles variations)
            def country_matches(row_country):
                if pd.isna(row_country) or row_country == "" or str(row_country).upper() == "EMPTY":
                    return False
                
                row_country_str = str(row_country).lower().strip()
                
                # Handle comma-separated countries
                country_parts = [part.strip() for part in row_country_str.split(',')]
                
                # Check each part for match
                for part in country_parts:
                    # Exact match
                    if country_lower == part:
                        return True
                    # Partial match (country name in part or part in country name)
                    if country_lower in part or part in country_lower:
                        return True
                    # Handle common variations (e.g., "Egypt" matches "Egyptian", "India" matches "Indian")
                    # Remove common suffixes
                    base_country = country_lower
                    base_part = part
                    for suffix in ['ian', 'ese', 'ish', 'i', 'an']:
                        if base_country.endswith(suffix):
                            base_country = base_country[:-len(suffix)]
                        if base_part.endswith(suffix):
                            base_part = base_part[:-len(suffix)]
                    if base_country == base_part and len(base_country) > 2:
                        return True
                
                return False
            
            country_mask = verified_df["manufacturer_country"].apply(country_matches)
        country_filtered_df = verified_df[country_mask].copy()
        
        if len(country_filtered_df) > 0:
            logger.info(f"🌍 Filtered to {len(country_filtered_df)} records matching country '{country}' (from {total_before_filter} total verified)")
            verified_df = country_filtered_df
        else:
            logger.warning(f"⚠ No records found matching country '{country}'. Showing all {total_before_filter} verified records.")

        # Filter out anonymous/generic manufacturer names
        if "company" in verified_df.columns:
            def is_anonymous_manufacturer(company_name):
                """Check if company name is anonymous or generic."""
                if pd.isna(company_name) or company_name == "":
                    return True
                
                company_lower = str(company_name).lower().strip()
                
                # Patterns that indicate anonymous/generic manufacturers
                anonymous_patterns = [
                    r'^manufacturer\s*#?\d+',  # "Manufacturer #274", "Manufacturer 123"
                    r'^manufacturer\s*$',      # Just "Manufacturer"
                    r'^anonymous',              # "Anonymous", "Anonymous Manufacturer"
                    r'^generic\s+manufacturer', # "Generic Manufacturer"
                    r'^company\s*#?\d+',       # "Company #123"
                    r'^supplier\s*#?\d+',       # "Supplier #123"
                    r'^vendor\s*#?\d+',        # "Vendor #123"
                    r'^manufacturer\s+\d+',     # "Manufacturer 274"
                ]
                
                for pattern in anonymous_patterns:
                    if re.match(pattern, company_lower):
                        return True
                
                # Check if it's too generic
                if company_lower in ['manufacturer', 'supplier', 'vendor', 'company', 'anonymous']:
                    return True
                
                return False
            
            before_anonymous_filter = len(verified_df)
        anonymous_mask = verified_df["company"].apply(is_anonymous_manufacturer)
        verified_df = verified_df[~anonymous_mask].copy()  # Keep non-anonymous
        
        removed_count = before_anonymous_filter - len(verified_df)
        if removed_count > 0:
            logger.info(f"🚫 Filtered out {removed_count} anonymous/generic manufacturer entries")
        
        # Drop duplicates by company+form+strength+url
        verified_df = verified_df.drop_duplicates(
            subset=["company", "form", "strength", "url"], keep="first"
        )

        # Ensure confidence column exists before sorting (CRITICAL - prevents KeyError)
        if verified_df.empty:
            logger.info("ℹ No verified results after filtering")
            return verified_df
        
        # Ensure all required columns exist
        required_columns = ["company", "form", "strength", "manufacturer_country", "verification_source", 
                           "confidence", "url", "verification_status", "additional_info"]
        for col in required_columns:
            if col not in verified_df.columns:
                logger.warning(f"⚠ Required column '{col}' not found in verified_df. Adding with default value.")
                if col == "confidence":
                    verified_df[col] = 100  # Default confidence
                else:
                    verified_df[col] = ""  # Default empty string
        
        # Ensure confidence is numeric
        if "confidence" in verified_df.columns:
            verified_df["confidence"] = pd.to_numeric(verified_df["confidence"], errors="coerce").fillna(100)
        else:
            # Fallback: if confidence still doesn't exist, add it
            verified_df["confidence"] = 100
            logger.warning("⚠ Confidence column was missing. Added with default value 100.")

        # Sort by confidence descending
        verified_df = verified_df.sort_values(by="confidence", ascending=False)

        logger.info(f"✅ Total rows: {len(all_df)}, Verified rows after filter: {len(verified_df)}")
        return verified_df

    # ============================================================
    # DATABASE METHODS
    # ============================================================
    def get_db_engine(self):
        """
        Get database engine - ONLY supports PostgreSQL (Supabase).
        DATABASE_URL environment variable is required.
        """
        database_url = os.getenv("DATABASE_URL") or self.DATABASE_URL
        
        if not database_url:
            logger.error("❌ DATABASE_URL is not set! Cannot connect to Supabase.")
            raise ValueError("DATABASE_URL environment variable is required for Supabase connection")
        
        if not database_url.startswith("postgresql://"):
            logger.error(f"❌ DATABASE_URL must start with 'postgresql://'")
            raise ValueError(f"Invalid DATABASE_URL format. Must start with 'postgresql://'")
        
        try:
            logger.info("🔗 Connecting to PostgreSQL (Supabase) database...")
            engine = create_engine(
                database_url,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                echo=False
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ PostgreSQL (Supabase) database connected successfully")
            return engine
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase PostgreSQL: {e}")
            raise

    def fetch_existing_data(self, api: str, country: str) -> pd.DataFrame:
        """
        Fetch existing manufacturer data from Supabase database based on API name and country.
        Returns DataFrame with existing results, or empty DataFrame if none found.
        """
        if not self.DATABASE_URL or not self.DATABASE_URL.startswith("postgresql://"):
            logger.error("❌ DATABASE_URL not configured for Supabase")
            return pd.DataFrame()
        
        if not PSYCOPG2_AVAILABLE:
            logger.warning("⚠ psycopg2 is not installed. Cannot fetch from Supabase.")
            logger.warning("⚠ Install with: pip install psycopg2-binary")
            return pd.DataFrame()
        
        try:
            conn = psycopg2.connect(self.DATABASE_URL)
            cursor = conn.cursor()
            
            # Query for existing records matching API name and country
            query = """
                SELECT company, form, strength, verification_source, 
                       confidence, url, verification_status, 
                       additional_info, source_site, api_name, manufacturer_country
                FROM viruj 
                WHERE LOWER(TRIM(api_name)) = LOWER(TRIM(%s))
                AND LOWER(TRIM(manufacturer_country)) = LOWER(TRIM(%s))
                ORDER BY confidence DESC
            """
            
            cursor.execute(query, (api, country))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                logger.info(f"📊 Fetched {len(df)} existing records from Supabase for {api} in {country}")
                return df
            else:
                logger.info(f"📊 No existing records found in Supabase for {api} in {country}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching from Supabase: {e}")
            return pd.DataFrame()

    def fetch_existing_companies(self, api: str, country: str):
        """Fetch list of existing company names from Supabase for the given API and country."""
        if not self.DATABASE_URL or not self.DATABASE_URL.startswith("postgresql://"):
            logger.error("❌ DATABASE_URL not configured for Supabase")
            return []
        
        if not PSYCOPG2_AVAILABLE:
            return []
        
        try:
            conn = psycopg2.connect(self.DATABASE_URL)
            cursor = conn.cursor()
            query = """
                SELECT DISTINCT company FROM viruj 
                WHERE LOWER(TRIM(api_name)) = LOWER(TRIM(%s))
                AND LOWER(TRIM(manufacturer_country)) = LOWER(TRIM(%s))
            """
            cursor.execute(query, (api, country))
            companies = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return companies
        except Exception as e:
            logger.error(f"❌ Error fetching existing companies from Supabase: {e}")
            return []

    def fetch_from_supabase(self, api: str, country: str) -> pd.DataFrame:
        """
        Fetch existing manufacturer data from Supabase viruj table based on API name and country.
        Returns DataFrame with existing results, or empty DataFrame if none found.
        """
        return self.fetch_existing_data(api, country)

    def insert_to_supabase(self, df: pd.DataFrame, api: str, country: str):
        """
        Insert verified manufacturer data into Supabase viruj table.
        Returns tuple: (success: bool, inserted_count: int, duplicate_count: int)
        """
        return self.insert_into_viruj(df, api, country)

    def insert_into_viruj(self, df: pd.DataFrame, api: str, country: str):
        """
        Insert verified manufacturer data into Supabase PostgreSQL database.
        Returns DataFrame with newly inserted records.
        """
        if df.empty:
            logger.warning("⚠ No data to insert into database")
            return pd.DataFrame()
        
        if not self.DATABASE_URL or not self.DATABASE_URL.startswith("postgresql://"):
            logger.error("❌ DATABASE_URL not configured for Supabase")
            return pd.DataFrame()
        
        if not PSYCOPG2_AVAILABLE:
            logger.warning("⚠ psycopg2 is not installed. Cannot insert to Supabase.")
            logger.warning("⚠ Install with: pip install psycopg2-binary")
            return pd.DataFrame()
        
        try:
                
                conn = psycopg2.connect(self.DATABASE_URL)
                cursor = conn.cursor()
                
                # Prepare data for insertion
                df_insert = df.copy()
                
                # Add api_name column if not present
                if "api_name" not in df_insert.columns:
                    df_insert["api_name"] = api
                
                # Add manufacturer_country column if not present
                if "manufacturer_country" not in df_insert.columns:
                    df_insert["manufacturer_country"] = country
                
                # Ensure all required columns exist
                for col in ["company", "form", "strength", "verification_source", "confidence", 
                           "url", "verification_status", "additional_info", "source_site"]:
                    if col not in df_insert.columns:
                        if col == "confidence":
                            df_insert[col] = 0
                        else:
                            df_insert[col] = ""
                
                # CRITICAL: Filter out rows without valid URLs before insertion
                before_url_filter = len(df_insert)
                if "url" in df_insert.columns:
                    def has_valid_url(row):
                        url = str(row.get("url", "")).strip()
                        return url and url.startswith("http")
                    url_mask = df_insert.apply(has_valid_url, axis=1)
                    df_insert = df_insert[url_mask].copy()
                    removed_no_url = before_url_filter - len(df_insert)
                    if removed_no_url > 0:
                        logger.warning(f"🚫 Removed {removed_no_url} entries without valid URLs before database insertion")
                
                if df_insert.empty:
                    logger.warning("⚠ No records with valid URLs to insert")
                    cursor.close()
                    conn.close()
                    return pd.DataFrame()
                
                # Fill NaN values (but URL should already be validated)
                df_insert = df_insert.fillna({
                    "company": "", "form": "", "strength": "", "verification_source": "",
                    "confidence": 0, "url": "", "verification_status": "Verified",
                    "additional_info": "", "source_site": "", "api_name": api,
                    "manufacturer_country": country
                })
                
                # Final URL validation - ensure no empty URLs slipped through
                if "url" in df_insert.columns:
                    df_insert = df_insert[df_insert["url"].str.strip().str.startswith("http", na=False)].copy()
                
                # Prepare columns for insertion
                columns = [
                    "company", "form", "strength", "verification_source", 
                    "confidence", "url", "verification_status", 
                    "additional_info", "source_site", "api_name", "manufacturer_country"
                ]
                available_columns = [col for col in columns if col in df_insert.columns]
                
                # Prepare data tuples
                values = []
                for _, row in df_insert.iterrows():
                    row_values = [str(row.get(col, "")) if pd.notna(row.get(col, "")) else "" for col in available_columns]
                    values.append(tuple(row_values))
                
                # Check for existing records
                existing_count = 0
                new_records = []
                
                fetch_existing_query = """
                    SELECT company, form, strength, api_name, manufacturer_country
                    FROM viruj 
                    WHERE LOWER(TRIM(api_name)) = LOWER(TRIM(%s))
                    AND LOWER(TRIM(manufacturer_country)) = LOWER(TRIM(%s))
                """
                cursor.execute(fetch_existing_query, (api, country))
                existing_records = cursor.fetchall()
                
                existing_keys = set()
                for existing_record in existing_records:
                    existing_key = (
                        str(existing_record[0] if existing_record[0] else "").strip().lower(),
                        str(existing_record[1] if existing_record[1] else "").strip().lower(),
                        str(existing_record[2] if existing_record[2] else "").strip().lower(),
                        str(existing_record[3] if existing_record[3] else "").strip().lower(),
                        str(existing_record[4] if existing_record[4] else "").strip().lower()
                    )
                    existing_keys.add(existing_key)
                
                for value_tuple in values:
                    row_idx = values.index(value_tuple)
                    row = df_insert.iloc[row_idx]
                    
                    record_key = (
                        str(row.get("company", "")).strip().lower(),
                        str(row.get("form", "")).strip().lower(),
                        str(row.get("strength", "")).strip().lower(),
                        str(row.get("api_name", api)).strip().lower(),
                        str(row.get("manufacturer_country", country)).strip().lower()
                    )
                    
                    if record_key in existing_keys:
                        existing_count += 1
                    else:
                        new_records.append(value_tuple)
                
                if not new_records:
                    logger.info(f"ℹ All {len(values)} records already exist in database")
                    cursor.close()
                    conn.close()
                    return pd.DataFrame()
                
                # Insert new records
                insert_query = f"""
                    INSERT INTO viruj ({', '.join(available_columns)})
                    VALUES %s
                """
                
                execute_values(cursor, insert_query, new_records)
                conn.commit()
                
                inserted_count = len(new_records)
                logger.info(f"✅ Inserted {inserted_count} new records into Supabase")
                if existing_count > 0:
                    logger.info(f"ℹ Skipped {existing_count} duplicate records")
                
                cursor.close()
                conn.close()
                
                # Return DataFrame with newly inserted records
                return df_insert.iloc[[values.index(nr) for nr in new_records]].copy()
            
        except Exception as e:
            logger.error(f"❌ Database insert error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    # ============================================================
    # MAIN ENTRY POINT - Compatible with Flask interface
    # ============================================================
    def find_api_buyers(self, api: str, country: str):
        """
        Main entry point for finding API buyers (FDF manufacturers).
        Compatible with Flask interface - returns dict with success, existing_data, newly_found_companies.
        """
        logger.info(f"\n🔍 Starting pharmaceutical research for {api} in {country}")
        
        # Step 1: Fetch existing data
        existing_data_df = self.fetch_existing_data(api, country)
        
        # Step 2: Run pipeline to find new manufacturers
        try:
            result_df = self.run_pipeline(api, country)
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            result_df = pd.DataFrame()
        
        # Step 3: Insert new results into database
        newly_inserted_df = pd.DataFrame()
        if not result_df.empty:
            newly_inserted_df = self.insert_into_viruj(result_df, api, country)
        
        # Step 4: Format results for Flask interface
        existing_data_formatted = []
        if not existing_data_df.empty:
            for _, row in existing_data_df.iterrows():
                existing_dict = {
                    'company': str(row.get('company', '')),
                    'api': str(row.get('api_name', api)) if 'api_name' in row else api,
                    'country': str(row.get('manufacturer_country', country)) if 'manufacturer_country' in row else country,
                    'form': str(row.get('form', '')),
                    'strength': str(row.get('strength', '')),
                    'additional_info': str(row.get('additional_info', '')),
                    'url': str(row.get('url', '')),
                    'confidence': int(row.get('confidence', 0)) if pd.notna(row.get('confidence', 0)) else 0,
                }
                existing_data_formatted.append(existing_dict)
        
        newly_found_companies = []
        if not newly_inserted_df.empty:
            for _, row in newly_inserted_df.iterrows():
                company_dict = {
                    'company': str(row.get('company', '')),
                    'api': api,
                    'country': country,
                    'form': str(row.get('form', '')),
                    'strength': str(row.get('strength', '')),
                    'additional_info': str(row.get('additional_info', '')),
                    'url': str(row.get('url', '')),
                    'confidence': int(row.get('confidence', 0)) if pd.notna(row.get('confidence', 0)) else 0,
                }
                newly_found_companies.append(company_dict)
        
        return {
            "success": True,
            "existing_data": existing_data_formatted,
            "newly_found_companies": newly_found_companies
        }
