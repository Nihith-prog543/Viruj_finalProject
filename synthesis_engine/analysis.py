# synthesis_engine/analysis.py - Core synthesis analysis logic for Flask backend
import requests
import time
import re
from collections import Counter
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from bs4 import BeautifulSoup
import uuid
from datetime import datetime
import json
import os
from PIL import Image
import io
import pandas as pd
from urllib.parse import urljoin
import base64
from googleapiclient.discovery import build
try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except ImportError:
    Chem = AllChem = Draw = None
    rdMolDraw2D = None
    RDKit_AVAILABLE = False
import pubchempy as pcp
import dataclasses

@dataclasses.dataclass
class GoogleCSESearchTool:
    synthesis_analyzer: "SynthesisAnalyzer"
    def run(self, query: str) -> list[tuple[str, str]]:
        return self.synthesis_analyzer.google_cse_search(query)

@dataclasses.dataclass
class SerpAPISearchTool:
    synthesis_analyzer: "SynthesisAnalyzer"
    def run(self, query: str) -> list[tuple[str, str]]:
        return self.synthesis_analyzer.serpapi_search(query)

def _get_smiles(name):
    try:
        compound = pcp.get_compounds(name, 'name')
        return compound[0].isomeric_smiles if compound else None
    except Exception as e:
        print(f"[DEBUG] Error getting SMILES for '{name}': {e}")
        return None

class SynthesisAnalyzer:
    def __init__(self):
        # API keys loaded from environment variables (required - no defaults for security)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
        self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
        self.SERP_API_KEY = os.getenv("SERP_API_KEY")

        # Constants previously global, now instance attributes
        self.TRUSTED_LITERATURE_DOMAINS = ["sciencedirect.com", "acs.org", "pubs.acs.org", "springer.com", "nature.com"]
        self.TRUSTED_PATENT_DOMAINS = ["patents.google.com", "espacenet.com", "wipo.int", "google.com/patents"]
        
        self.SYNTHESIS_SECTION_PATTERNS = [
            r'EXAMPLE\s+\d+[:\-\s]*(.*?)(?=EXAMPLE\s+\d+|CLAIMS|BRIEF DESCRIPTION|$)',
            r'Example\s+\d+[:\-\s]*(.*?)(?=Example\s+\d+|Claims|Brief Description|$)', 
            r'PREPARATION\s+\d*[:\-\s]*(.*?)(?=PREPARATION|EXAMPLE|CLAIMS|$)',
            r'Preparation\s+\d*[:\-\s]*(.*?)(?=Preparation|Example|Claims|$)',
            r'SYNTHESIS[:\-\s]*(.*?)(?=EXAMPLE|CLAIMS|BRIEF DESCRIPTION|$)',
            r'Synthesis[:\-\s]*(.*?)(?=Example|Claims|Brief Description|$)',
            r'METHOD\s+\d*[:\-\s]*(.*?)(?=METHOD|EXAMPLE|CLAIMS|$)',
            r'PROCEDURE[:\-\s]*(.*?)(?=PROCEDURE|EXAMPLE|CLAIMS|$)',
            r'Step\s+\d+[:\-\s]*(.*?)(?=Step\s+\d+|EXAMPLE|CLAIMS|$)',
            r'Stage\s+\d+[:\-\s]*(.*?)(?=Stage\s+\d+|EXAMPLE|CLAIMS|$)',
        ]
        
        self.COMMERCIAL_VIABILITY_INDICATORS = {
            'positive': [
                'high yield', 'good yield', 'excellent yield', 'quantitative yield',
                'mild conditions', 'room temperature', 'ambient temperature',
                'water soluble', 'green chemistry', 'environmentally friendly',
                'scalable', 'large scale', 'industrial scale', 'commercial scale',
                'cost effective', 'economical', 'inexpensive', 'cheap reagents',
                'readily available', 'commercially available', 'standard conditions',
                'simple purification', 'crystallization', 'precipitation'
            ],
            'negative': [
                'low yield', 'poor yield', 'moderate yield',
                'harsh conditions', 'extreme temperature', 'high pressure',
                'toxic reagents', 'dangerous', 'hazardous', 'explosive',
                'expensive', 'costly', 'precious metal', 'rare reagent',
                'difficult purification', 'complex separation', 'chromatography required',
                'unstable', 'sensitive to air', 'sensitive to moisture',
                'long reaction time', 'multiple steps', 'tedious workup'
            ]
        }
        
        self.BERT_THRESHOLD = 0.35
        self.SYNTHESIS_KEYWORD_THRESHOLD = 3  # Minimum keyword hits to treat as synthesis
        self.SYNTHESIS_FORMULATION_RATIO = 1.2  # Synthesis keywords must beat formulation keywords by this factor
    
    def normalize_api_name(self, api_name):
        normalized = api_name.lower()
        salt_suffixes = [' hcl', ' hydrochloride', ' sulfate', ' phosphate', ' tartrate', 
                        ' maleate', ' fumarate', ' citrate', ' succinate', ' acetate']
        
        base_name = normalized
        for suffix in salt_suffixes:
            if normalized.endswith(suffix):
                base_name = normalized.replace(suffix, '').strip()
                break
        
        return base_name, normalized

    def _generate_api_variants(self, api_name):
        """
        Create multiple variations of the API name so we can detect references
        to parent/child compounds (e.g., ravuconazole vs isavuconazole).
        """
        base_name, full_name = self.normalize_api_name(api_name)
        variants = {base_name, full_name}
        
        cleaned = re.sub(r'[^a-z0-9]', '', base_name)
        if cleaned:
            variants.add(cleaned)
        
        # Remove common prefixes (iso-, des-, levo-, nor-, de-)
        prefix_candidates = ['iso', 'des', 'levo', 'd-', 'l-', 'nor', 'de', 'di']
        for prefix in prefix_candidates:
            stripped = cleaned.removeprefix(prefix.replace('-', ''))
            if stripped and stripped != cleaned:
                variants.add(stripped)
        
        # Capture shared cores (e.g., "avuconazole")
        if cleaned.endswith('azole') and len(cleaned) > 6:
            variants.add(cleaned[-8:])
            variants.add(cleaned.replace('azole', 'conazole'))
        
        if len(cleaned) > 6:
            variants.add(cleaned[:len(cleaned)-2])
            variants.add(cleaned[-6:])
        
        # Remove empty strings
        return {variant for variant in variants if variant}
    
    def generate_enhanced_search_queries(self, api_name):
        base_name, full_name = self.normalize_api_name(api_name)
        
        variations = [base_name, full_name]
        
        if 'iso' in base_name:
            variations.append(base_name.replace('iso', ''))
        if 'des' in base_name:
            variations.append(base_name.replace('des', ''))
        if base_name.endswith('azole'):
            root = base_name[:-5]
            variations.extend([f"{root}conazole", f"{root}fluconazole"])
        
        queries = []
        for variation in variations:
            queries.extend([
                f"synthesis of {variation}",
                f"preparation of {variation}",
                f"chemical synthesis {variation}",
                f"synthetic route {variation}",
                f"manufacturing process {variation}",
                f"process for preparing {variation}",
                f"method of making {variation}",
                f"synthetic pathway {variation}",
                f"chemical process {variation}",
                f"{variation} synthesis patent",
                f"{variation} manufacturing patent",
                f"{variation} preparation method",
                f"active ingredient {variation} synthesis",
                f"pharmaceutical synthesis {variation}"
            ])
        
        return list(set(queries))
    
    def google_cse_search(self, query):
        try:
            service = build("customsearch", "v1", developerKey=self.GOOGLE_CSE_API_KEY)
            results = service.cse().list(q=query, cx=self.GOOGLE_CSE_ID, num=10).execute()
            found_links = [(item["link"], "Google CSE") for item in results.get("items", [])]
            print(f"[DEBUG] Google CSE search for '{query}' found {len(found_links)} results.")
            return found_links
        except Exception as e:
            print(f"[DEBUG] Google CSE search error for '{query}': {e}")
            return []
    
    def serpapi_search(self, query):
        try:
            url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={self.SERP_API_KEY}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return [(res['link'], "SerpAPI") for res in data.get("organic_results", []) if 'link' in res]
            return []
        except Exception as e:
            print(f"[DEBUG] SerpAPI search error: {e}")
            return []
    
    def extract_detailed_patent_content(self, url, api_name):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        try:
            print(f"[DEBUG] 🔍 Enhanced extraction from: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"[DEBUG] ❌ HTTP {response.status_code} for {url}")
                return None
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            title = self._extract_patent_title(soup)
            abstract = self._extract_patent_abstract(soup)
            synthesis_content = self._extract_synthesis_sections(soup, api_name)
            claims = self._extract_patent_claims(soup)
            detailed_description = self._extract_detailed_description(soup, api_name)
            patent_images = self._extract_patent_images(url, "https://patents.google.com", api_name)
            
            full_content = f"""
PATENT TITLE: {title}

ABSTRACT:
{abstract}

SYNTHESIS EXAMPLES AND PROCEDURES:
{synthesis_content}

DETAILED DESCRIPTION:
{detailed_description}

RELEVANT CLAIMS:
{claims}
"""
            
            patent_data = {
                'title': title,
                'abstract': abstract,
                'synthesis_content': synthesis_content,
                'detailed_description': detailed_description,
                'claims': claims,
                'full_content': full_content,
                'images': patent_images,
                'content_length': len(full_content),
                'synthesis_sections_found': len(re.findall(r'EXAMPLE|Example|PREPARATION|Preparation', synthesis_content)),
                'url': url
            }
            
            print(f"[DEBUG] 📊 Enhanced extraction stats:")
            print(f"  Title: {title[:60]}...")
            print(f"  Abstract length: {len(abstract)}")
            print(f"  Synthesis content length: {len(synthesis_content)}")
            print(f"  Synthesis sections found: {patent_data['synthesis_sections_found']}")
            print(f"  Images found: {len(patent_images)}")
            print(f"  Total content length: {len(full_content)}")
            
            return patent_data
            
        except Exception as e:
            print(f"[DEBUG] ❌ Enhanced patent extraction error: {e}")
            return None
    
    def _extract_patent_title(self, soup):
        title_selectors = [
            'span[itemprop="title"]',
            'h1[itemprop="title"]', 
            'meta[name="DC.title"]',
            'meta[property="og:title"]',
            'title',
            'h1',
            '.patent-title',
            '#title',
            '[data-patent-title]'
        ]
        
        for selector in title_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    title = element.get('content', '').strip() if hasattr(element, 'content') else element.get_text().strip()
                    if title and len(title) > 10 and 'Google Patents' not in title:
                        return title
            except:
                continue
        
        return "Unknown Patent Title"
    
    def _extract_patent_abstract(self, soup):
        abstract_selectors = [
            'meta[name="DC.description"]',
            'meta[name="description"]',
            'div.abstract',
            'section[itemprop="abstract"]',
            '[data-abstract]',
            '.patent-abstract',
            '#abstract',
            'div[class*="abstract"]'
        ]
        
        for selector in abstract_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    abstract = element.get('content', '').strip() if hasattr(element, 'content') else element.get_text().strip()
                    if abstract and len(abstract) > 50:
                        return abstract
            except:
                continue
        
        return "Abstract not found"
    
    def _extract_synthesis_sections(self, soup, api_name):
        full_text = soup.get_text(separator="\n")
        synthesis_sections = []
        
        for pattern in self.SYNTHESIS_SECTION_PATTERNS:
            matches = re.finditer(pattern, full_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section = match.group(1).strip()
                if len(section) > 100:
                    synthesis_sections.append(section)
        
        if not synthesis_sections:
            synthesis_sections = self._extract_procedural_text(full_text, api_name)
        
        combined_synthesis = "\n\n".join(synthesis_sections[:10])
        combined_synthesis = re.sub(r'\n\s*\n\s*\n', '\n\n', combined_synthesis)
        combined_synthesis = re.sub(r'[^\S\n]+', ' ', combined_synthesis)
        
        return combined_synthesis[:8000]

    def _extract_procedural_text(self, text, api_name):
        base_name, full_name = self.normalize_api_name(api_name)
        
        paragraphs = text.split('\n\n')
        synthesis_paragraphs = []
        
        synthesis_keywords = [
            'prepared', 'preparation', 'synthesis', 'reaction', 'treated with',
            'dissolved in', 'added to', 'heated to', 'cooled to', 'stirred',
            'reflux', 'temperature', 'reagent', 'solvent', 'catalyst',
            'yield', 'product', 'compound', 'mixture', 'solution'
        ]
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 50:
                continue
                
            keyword_count = sum(1 for keyword in synthesis_keywords if keyword.lower() in paragraph.lower())
            api_mentioned = base_name.lower() in paragraph.lower() or full_name.lower() in paragraph.lower()
            
            if keyword_count >= 3 or (keyword_count >= 2 and api_mentioned):
                synthesis_paragraphs.append(paragraph)
        
        return synthesis_paragraphs[:20]

    def _extract_patent_claims(self, soup):
        claims_text = ""
        
        claims_selectors = [
            'div[class*="claims"]',
            'section[class*="claims"]', 
            '#claims',
            '.patent-claims',
            '[data-claims]'
        ]
        
        for selector in claims_selectors:
            try:
                claims_element = soup.select_one(selector)
                if claims_element:
                    claims_text = claims_element.get_text().strip()
                    break
            except:
                continue
        
        if not claims_text:
            full_text = soup.get_text()
            claims_pattern = r'(?:CLAIMS?|What is claimed)[:\s]*(.*?)(?=BRIEF DESCRIPTION|DETAILED DESCRIPTION|ABSTRACT|$)'
            match = re.search(claims_pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                claims_text = match.group(1).strip()
        
        return claims_text[:3000]

    def _extract_detailed_description(self, soup, api_name):
        base_name, full_name = self.normalize_api_name(api_name)
        
        description_selectors = [
            'div[class*="description"]',
            'section[class*="description"]',
            '#detailed-description',
            '.patent-description'
        ]
        
        description_text = ""
        for selector in description_selectors:
            try:
                desc_element = soup.select_one(selector)
                if desc_element:
                    description_text = desc_element.get_text().strip()
                    break
            except:
                continue
        
        if not description_text:
            full_text = soup.get_text()
            desc_pattern = r'(?:DETAILED DESCRIPTION|DESCRIPTION)[:\s]*(.*?)(?=CLAIMS|BRIEF DESCRIPTION|ABSTRACT|$)'
            match = re.search(desc_pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                description_text = match.group(1).strip()
            else:
                description_text = full_text
        
        synthesis_relevant_text = self._extract_synthesis_relevant_portions(description_text, api_name)
        
        return synthesis_relevant_text[:5000]

    def _extract_synthesis_relevant_portions(self, text, api_name):
        base_name, full_name = self.normalize_api_name(api_name)
        
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        synthesis_indicators = [
            'prepared', 'preparation', 'synthesis', 'synthetic', 'reaction',
            'treated', 'added', 'dissolved', 'heated', 'cooled', 'stirred',
            'reagent', 'catalyst', 'solvent', 'temperature', 'yield',
            'product', 'compound', 'intermediate', 'starting material'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            
            synthesis_score = sum(1 for indicator in synthesis_indicators if indicator in sentence_lower)
            api_mentioned = base_name.lower() in sentence_lower or full_name.lower() in sentence_lower
            
            if synthesis_score >= 2 or (synthesis_score >= 1 and api_mentioned):
                relevant_sentences.append(sentence)
        
        return ". ".join(relevant_sentences[:50])

    def _is_synthesis_patent_enhanced(self, text, api_name):
        text_lower = text.lower()
        base_name, full_name = self.normalize_api_name(api_name)
        api_variants = self._generate_api_variants(api_name)
        
        synthesis_keywords = [
            'synthesis', 'preparation', 'synthetic route', 'chemical process', 
            'manufacturing process', 'reaction', 'intermediate', 'coupling', 
            'reagent', 'procedure', 'example', 'step', 'stage', 'method'
        ]
        
        formulation_keywords = [
            'tablet', 'capsule', 'formulation', 'pharmaceutical composition', 
            'solid dosage', 'excipient', 'binder', 'disintegrant', 'lubricant', 
            'coating', 'granulation', 'compression', 'drug delivery', 'dosage form'
        ]
        
        synthesis_score = sum(text_lower.count(keyword) for keyword in synthesis_keywords)
        formulation_score = sum(text_lower.count(keyword) for keyword in formulation_keywords)
        
        api_present = any(variant in text_lower for variant in api_variants)
        
        fuzzy_api_present = False
        if not api_present:
            for variant in api_variants:
                if len(variant) < 6:
                    continue
                core_parts = {variant[:len(variant)-2], variant[-6:]}
                if any(part and part in text_lower for part in core_parts):
                    fuzzy_api_present = True
                    break
        
        decision_criteria = [
            synthesis_score >= self.SYNTHESIS_KEYWORD_THRESHOLD,
            api_present or fuzzy_api_present,
            formulation_score == 0 or synthesis_score >= (formulation_score * self.SYNTHESIS_FORMULATION_RATIO)
        ]
        
        is_synthesis = all(decision_criteria)
        
        print(f"[DEBUG] Patent Classification:")
        print(f"  Synthesis Score: {synthesis_score}")
        print(f"  Formulation Score: {formulation_score}")
        print(f"  API Variants Checked: {sorted(api_variants)}")
        print(f"  API Present: {api_present}")
        print(f"  Fuzzy API Present: {fuzzy_api_present}")
        print(f"  Decision: {'SYNTHESIS' if is_synthesis else 'FORMULATION/OTHER'}")
        
        metadata = {
            'synthesis_score': synthesis_score,
            'formulation_score': formulation_score,
            'api_present': api_present,
            'fuzzy_api_present': fuzzy_api_present,
            'api_variants': sorted(api_variants)
        }
        
        return is_synthesis, metadata

    def _enhanced_bert_similarity_multi_query(self, text, api_name):
        base_name, full_name = self.normalize_api_name(api_name)
        
        queries = [
            f"chemical synthesis of {base_name}",
            f"synthesis route for {base_name}",
            f"preparation of {base_name} compound",
            f"making {base_name} molecule",
            f"synthesis of {full_name}",
            f"chemical process to make {base_name}",
            f"organic synthesis of {base_name}",
            f"pharmaceutical synthesis {base_name}",
            f"manufacturing process {base_name}"
        ]
        
        max_score = 0
        best_query = ""
        
        if not self.HUGGINGFACE_API_KEY:
            return True, 0.0
        
        for query in queries:
            try:
                headers = {"Authorization": f"Bearer {self.HUGGINGFACE_API_KEY}"}
                payload = {"inputs": {"source_sentence": text, "sentences": [query]}}
                response = requests.post(
                    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2", 
                    headers=headers, json=payload, timeout=30
                )
                if response.status_code == 200 and isinstance(response.json(), list):
                    score = response.json()[0]
                    if score > max_score:
                        max_score = score
                        best_query = query
            except Exception as e:
                print(f"[DEBUG] BERT API error for query '{query}': {e}")
                continue
        
        print(f"[DEBUG] Best BERT query: '{best_query}' → Score: {max_score:.3f}")
        return max_score >= self.BERT_THRESHOLD, max_score

    def is_url_valid(self, url):
        try:
            resp = requests.get(url, timeout=10, allow_redirects=True)
            return resp.status_code == 200
        except:
            return False

    def _classify_source(self, url):
        for domain in self.TRUSTED_PATENT_DOMAINS:
            if domain in url:
                return "patent"
        for domain in self.TRUSTED_LITERATURE_DOMAINS:
            if domain in url:
                return "literature"
        return "other"

    def _extract_relevant_text(self, html):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return title_text + " " + text[:2000]
        except:
            return html[:1000]

    def _is_source_relevant_enhanced(self, url, api_name):
        try:
            print(f"[DEBUG] Analyzing relevance: {url}")
            
            if "patents.google.com" in url or "google.com/patents" in url:
                patent_data = self.extract_detailed_patent_content(url, api_name)
                if not patent_data:
                    print(f"[DEBUG] Patent data extraction failed")
                    return False, 0.0, None
                
                analysis_text = patent_data['full_content']
            else:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, timeout=15, headers=headers)
                analysis_text = self._extract_relevant_text(response.text)
                patent_data = {
                    'title': 'Unknown', 
                    'abstract': analysis_text[:500], 
                    'full_content': analysis_text, 
                    'synthesis_content': analysis_text,
                    'images': [],
                    'content_length': len(analysis_text),
                    'synthesis_sections_found': 0
                }
            
            is_synthesis, synth_meta = self._is_synthesis_patent_enhanced(analysis_text, api_name)
            if not is_synthesis:
                print(f"[DEBUG] Failed enhanced synthesis patent check")
                return False, 0.0, None
            
            relevant, bert_score = self._enhanced_bert_similarity_multi_query(analysis_text, api_name)
            
            content_score = min(100, len(patent_data.get('synthesis_content', '')) / 50)
            sections_score = patent_data.get('synthesis_sections_found', 0) * 20
            images_score = min(20, len(patent_data.get('images', [])) * 5)
            
            final_score = (bert_score * 0.4) + (content_score/100 * 0.3) + (sections_score/100 * 0.2) + (images_score/100 * 0.1)
            
            print(f"[DEBUG] SCORING BREAKDOWN:")
            print(f"  BERT Score: {bert_score:.3f}")
            print(f"  Content Score: {content_score:.1f}/100")
            print(f"  Sections Score: {sections_score:.1f}/100") 
            print(f"  Images Score: {images_score:.1f}/100")
            print(f"  FINAL SCORE: {final_score:.3f}")
            
            is_relevant = (final_score >= 0.35 and bert_score >= 0.2) or (final_score >= 0.5 and bert_score >= 0.1)
            fallback_reason = None
            
            if not is_relevant:
                strong_content = content_score >= 40 and sections_score >= 20
                if strong_content:
                    is_relevant = True
                    fallback_reason = "content_override"
                elif synth_meta.get('synthesis_score', 0) >= self.SYNTHESIS_KEYWORD_THRESHOLD and (synth_meta.get('api_present') or synth_meta.get('fuzzy_api_present')):
                    is_relevant = True
                    fallback_reason = "keyword_override"
            
            print(f"[DEBUG] {'ACCEPTED' if is_relevant else 'REJECTED'} - Final Score: {final_score:.3f}")
            if fallback_reason:
                print(f"[DEBUG] Override applied: {fallback_reason}")
            
            patent_data['confidence'] = 'low' if fallback_reason else 'high'
            
            return is_relevant, final_score, patent_data
            
        except Exception as e:
            print(f"[DEBUG] Enhanced analysis error: {e}")
            return False, 0.0, None
    
    def assess_commercial_viability(self, synthesis_content, api_name):
        content_lower = synthesis_content.lower()
        
        positive_score = 0
        negative_score = 0
        found_indicators = {'positive': [], 'negative': []}
        
        for indicator in self.COMMERCIAL_VIABILITY_INDICATORS['positive']:
            if indicator in content_lower:
                positive_score += 1
                found_indicators['positive'].append(indicator)
        
        for indicator in self.COMMERCIAL_VIABILITY_INDICATORS['negative']:
            if indicator in content_lower:
                negative_score += 1
                found_indicators['negative'].append(indicator)
        
        yield_info = self._extract_yield_information(synthesis_content)
        conditions_info = self._extract_reaction_conditions(synthesis_content)
        
        total_indicators = positive_score + negative_score
        viability_score = (positive_score / total_indicators) * 100 if total_indicators else 50
        
        if yield_info['max_yield'] > 85:
            viability_score += 15
        elif yield_info['max_yield'] > 70:
            viability_score += 10
        elif yield_info['max_yield'] > 50:
            viability_score += 5
        elif yield_info['max_yield'] < 30 and yield_info['max_yield'] > 0:
            viability_score -= 20
        
        viability_score = min(100, max(0, viability_score))
        
        if viability_score >= 75:
            viability_level = "HIGH - Commercially Viable"
        elif viability_score >= 50:
            viability_level = "MEDIUM - Potentially Viable with Optimization"
        else:
            viability_level = "LOW - Requires Significant Optimization"
        
        assessment = {
            'score': round(viability_score, 1),
            'level': viability_level,
            'positive_indicators': found_indicators['positive'],
            'negative_indicators': found_indicators['negative'],
            'yield_info': yield_info,
            'conditions_info': conditions_info,
            'recommendations': self._generate_viability_recommendations(viability_score, found_indicators, yield_info)
        }
        
        return assessment
    
    def _extract_yield_information(self, text):
        yield_patterns = [
            r'yield[:\s]*(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*yield',
            r'in\s*(\d+(?:\.\d+)?)\s*%\s*yield',
            r'obtained\s*in\s*(\d+(?:\.\d+)?)\s*%',
            r'giving\s*(\d+(?:\.\d+)?)\s*%',
            r'afforded\s*(\d+(?:\.\d+)?)\s*%',
            r'was\s*obtained\s*in\s*(\d+(?:\.\d+)?)\s*%',
            r'to\s*give\s*(\d+(?:\.\d+)?)\s*%',
            r'reported\s*(\d+(?:\.\d+)?)\s*%'
        ]
        
        yields = []
        for pattern in yield_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    yield_value = float(match)
                    if 0 <= yield_value <= 100:
                        yields.append(yield_value)
                except:
                    continue
        
        print(f"[DEBUG] Extracted yields: {yields}")
        return {
            'yields_found': yields,
            'max_yield': max(yields) if yields else 0,
            'avg_yield': sum(yields) / len(yields) if yields else 0,
            'yield_count': len(yields)
        }
    
    def _extract_reaction_conditions(self, text):
        conditions = {'temperatures': [], 'solvents': [], 'catalysts': [], 'reaction_times': []}
        
        temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*°?[CF]',
            r'heated?\s*to\s*(\d+(?:\.\d+)?)\s*°?[CF]',
            r'temperature\s*of\s*(\d+(?:\.\d+)?)\s*°?[CF]'
        ]
        
        for pattern in temp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    temp = float(match)
                    if -100 <= temp <= 500:
                        conditions['temperatures'].append(temp)
                except:
                    continue
        
        solvents = ['water', 'ethanol', 'methanol', 'acetone', 'toluene', 'benzene', 
                   'dichloromethane', 'chloroform', 'ether', 'THF', 'DMF', 'DMSO']
        
        for solvent in solvents:
            if solvent in text.lower():
                conditions['solvents'].append(solvent)
        
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*hours?',
            r'(\d+(?:\.\d+)?)\s*hrs?',
            r'(\d+(?:\.\d+)?)\s*minutes?',
            r'(\d+(?:\.\d+)?)\s*mins?'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions['reaction_times'].extend([float(m) for m in matches])
        
        return conditions

    def _generate_viability_recommendations(self, score, indicators, yield_info):
        recommendations = []
        
        if score < 50:
            recommendations.append("❗ CRITICAL: Current route shows low commercial viability")
            recommendations.append("🔄 Consider alternative synthesis routes")
            
        if yield_info['max_yield'] < 70:
            recommendations.append("📈 Optimize reaction conditions to improve yield")
            recommendations.append("🧪 Investigate catalyst alternatives")
            
        if 'harsh conditions' in indicators['negative'] or 'extreme temperature' in indicators['negative']:
            recommendations.append("🌡️ Optimize reaction conditions for milder temperatures")
            
        if 'expensive' in indicators['negative'] or 'costly' in indicators['negative']:
            recommendations.append("💰 Evaluate alternative, cost-effective reagents")
            
        if 'toxic reagents' in indicators['negative']:
            recommendations.append("☣️ Replace toxic reagents with safer alternatives")
            
        if not recommendations:
            recommendations.append("✅ Route shows good commercial potential")
            recommendations.append("🔧 Consider minor optimizations for scale-up")
        
        return recommendations

    def _download_image(self, url, headers=None):
        try:
            if headers is None:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            response = requests.get(url, headers=headers, timeout=15, stream=True)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            return None
        except Exception as e:
            print(f"[DEBUG] Error downloading image {url}: {e}")
            return None

    def _is_relevant_patent_image(self, img_url, img_alt="", img_title=""):
        relevant_keywords = [
            "formula", "structure", "reaction", "scheme", "synthesis", "chemical",
            "compound", "molecule", "fig", "figure", "diagram", "process", "step"
        ]
        
        irrelevant_keywords = [
            "logo", "banner", "advertisement", "profile", "avatar", "icon",
            "button", "menu", "navigation", "footer", "header"
        ]
        
        text_to_check = f"{img_url} {img_alt} {img_title}".lower()
        
        relevant_count = sum(1 for keyword in relevant_keywords if keyword in text_to_check)
        irrelevant_count = sum(1 for keyword in irrelevant_keywords if keyword in text_to_check)
        
        if irrelevant_count > 0:
            return False
        if relevant_count > 0:
            return True
        
        if any(ext in img_url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
            if any(ui_term in img_url.lower() for ui_term in ['logo', 'icon', 'button', 'banner']):
                return False
            return True
        
        return False

    def _extract_patent_images(self, url, base_url, api_name):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, "html.parser")
            images = []
            img_tags = soup.find_all("img")
            
            print(f"[DEBUG] Found {len(img_tags)} total images on page")
            
            for img in img_tags:
                img_src = img.get("src", "")
                img_alt = img.get("alt", "")
                img_title = img.get("title", "")
                
                if not img_src:
                    continue
                
                if img_src.startswith("//"):
                    img_src = "https:" + img_src
                elif img_src.startswith("/"):
                    img_src = urljoin(base_url, img_src)
                elif not img_src.startswith("http"):
                    img_src = urljoin(url, img_src)
                
                if self._is_relevant_patent_image(img_src, img_alt, img_title):
                    image_obj = self._download_image(img_src, headers)
                    if image_obj:
                        width, height = image_obj.size
                        if width >= 100 and height >= 100:
                            buffered = io.BytesIO()
                            image_obj.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            images.append({
                                'url': img_src, 'alt': img_alt, 'title': img_title,
                                'width': width, 'height': height, 'base64': img_str
                            })
            return images[:10]
        except Exception as e:
            print(f"[DEBUG] Image extraction error: {e}")
            return []

    def _generate_reaction_image(self, reaction_smiles):
        if not RDKit_AVAILABLE:
            print("[DEBUG] RDKit not available; skipping reaction image generation.")
            return None
        try:
            # Attempt to parse as a reaction SMARTS first
            if '>>' in reaction_smiles:
                rxn = AllChem.ReactionFromSmarts(reaction_smiles)
                img = rdMolDraw2D.MolDraw2DCairo(500, 200)  # Adjust dimensions as needed
                img.drawReaction(rxn)
                img.finishDrawing()
                buffered = io.BytesIO(img.GetDrawingText())
            else:
                # Otherwise, assume it's a comma-separated list of compound names
                compound_names = [name.strip() for name in reaction_smiles.split(',')]
                mols = []
                for name in compound_names:
                    smiles = _get_smiles(name)
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mols.append(mol)
                
                if not mols:
                    print(f"[DEBUG] No valid molecules could be generated from: {reaction_smiles}")
                    return None
                
                # Use MolsToGridImage for individual molecules
                # Assuming a reasonable grid for visualization
                img_grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=min(len(mols), 4), subImgSize=(200, 200), 
                                                  legends=[name for name in compound_names if _get_smiles(name)])
                buffered = io.BytesIO()
                img_grid.save(buffered, format="PNG")
            
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            print(f"[DEBUG] Error generating RDKit image for SMILES '{reaction_smiles}': {e}")

    def _create_synthesis_agent(self, selected_model, api_name, country_input, viability_assessment):
        viability_context = f"""
COMMERCIAL VIABILITY ASSESSMENT:
- Viability Score: {viability_assessment['score']}/100
- Level: {viability_assessment['level']}
- Positive Indicators: {', '.join(viability_assessment['positive_indicators'])}
- Negative Indicators: {', '.join(viability_assessment['negative_indicators'])}
- Maximum Yield Found: {viability_assessment['yield_info']['max_yield']}%
- Recommendations: {'; '.join(viability_assessment['recommendations'])}
"""
        enhanced_instructions = f"""
You are a world-class pharmaceutical process chemist analyzing VERIFIED SYNTHESIS PATENTS for {api_name}.

{viability_context}

🎯 PRIMARY OBJECTIVES:
- Extract EXACT synthesis steps from verified patents
- Assess and improve commercial viability based on the analysis above
- If current route is not commercially viable (score < 75), propose optimized alternatives
- Provide comprehensive supplier analysis with global focus

🔬 DETAILED ANALYSIS REQUIREMENTS:

*Patent Synthesis Extraction:*
- Extract exact reaction conditions, yields, purities from patent
- Reference specific examples (Example 1, Example 2, etc.)
- Quote exact patent language for critical steps
- Distinguish between "Patent Data" vs "AI Optimization"

*Commercial Viability Enhancement:*
- If viability score < 75: Propose optimized route with improvements
- Address specific negative indicators identified
- Optimize for: Yield >85%, Cost-effectiveness, Safety, Scalability
- Provide alternative reagents/conditions for commercial feasibility

*Supplier Integration:*
- For each raw material, provide 8-10 global suppliers
- Prioritize {country_input or 'China and India'} suppliers
- Include estimated costs and minimum order quantities

🏗 REQUIRED OUTPUT FORMAT:

## 1️⃣ Commercial Viability Assessment Summary
**Current Route Viability: {viability_assessment['score']}/100 ({viability_assessment['level']})**
- Key Strengths: [List positive aspects]
- Major Concerns: [List issues affecting viability]
- Overall Recommendation: [PROCEED / OPTIMIZE / ALTERNATIVE NEEDED]

## 2️⃣ Patent Synthesis Route (As Disclosed)
**From Patent: [Patent Title/URL]**

*Step 1:* [Exact patent description]
- Starting Material: [Compound name, CAS if available] 
- Reagents: [From patent with quantities if given]
- Conditions: [Temperature, time, solvent - from patent]
- Yield: [X%] *(Patent Data)*
- Notes: [Any specific patent observations]

*Step 2:* [Continue format...]

## 3️⃣ Commercially Optimized Route
**{f"ALTERNATIVE ROUTE (Original shows low viability)" if viability_assessment['score'] < 75 else "OPTIMIZED VERSION OF PATENT ROUTE"}**

*Optimization Strategy:*
- Yield improvements: [Target >85%]
- Cost reductions: [Alternative reagents/conditions]
- Safety enhancements: [Safer solvents/conditions]
- Scalability improvements: [Equipment/process changes]

*Optimized Step 1:*
- Starting Material: [Name, supplier suggestions]
- Optimized Reagents: [Cost-effective alternatives]
- Improved Conditions: [Optimized T, time, solvent]
- Expected Yield: [X%] *(AI Optimized)*
- Commercial Advantages: [Why this is better]

*Optimized Step 2:* [Continue...]

## 4️⃣ Raw Material Suppliers & Economics
**For each key raw material identified in the synthesis steps, USE YOUR SEARCH TOOLS to find and provide the following information:**
| Material | CAS # | Top Suppliers (8-10) | Region | Est. Price/kg | MOQ | Quality |
|----------|-------|----------------------|---------|---------------|-----|---------|
| [Name] | [CAS] | [Supplier list] | [Region] | $[Price] | [MOQ] | [Grade] |

## 5️⃣ Process Economics Analysis
- **Raw Material Cost:** $[X]/kg of API
- **Overall Process Yield:** [X%] 
- **Estimated Production Cost:** $[X]/kg
- **Break-even Analysis:** [Volume needed for profitability]
- **ROI Timeline:** [Estimated payback period]

## 6️⃣ Risk Assessment & Mitigation
**Technical Risks:**
- Yield variability: [Risk level & mitigation]
- Impurity formation: [Potential issues & controls]
- Scale-up challenges: [Identified risks & solutions]

**Commercial Risks:**
- Supply chain: [Raw material availability]
- Regulatory: [Patent freedom to operate]
- Market: [Competition & positioning]

## 7️⃣ Implementation Roadmap
**Phase 1: Laboratory Optimization (Months 1-3)**
- Confirm patent route reproducibility
- Optimize critical parameters
- Develop analytical methods

**Phase 2: Process Development (Months 4-8)**  
- Scale-up studies (100g → 1kg → 10kg)
- Process safety evaluation
- Supply chain establishment

**Phase 3: Commercial Preparation (Months 9-12)**
- Regulatory filing preparation
- Commercial supplier agreements
- Production facility preparation

## 8️⃣ Alternative Routes Comparison
{f"**Recommended Alternative Routes:** (Since current route viability = {viability_assessment['score']}/100)" if viability_assessment['score'] < 75 else "**Alternative Routes for Consideration:**"}

*Route A:* [Brief description, advantages]
*Route B:* [Brief description, advantages] 
*Route C:* [Brief description, advantages]

**Recommended Route:** [A/B/C] - [Justification]

## 9️⃣ Quality Control Strategy  
- **Critical Process Parameters:** [Temperature, pH, reaction time]
- **In-Process Controls:** [What to monitor and when]
- **Final Product Specifications:** [Purity, impurities, physical properties]
- **Analytical Methods:** [HPLC, GC, NMR requirements]

## 🔟 Executive Recommendation
**Final Recommendation:** {f"DEVELOP ALTERNATIVE ROUTE - Patent route not commercially viable" if viability_assessment['score'] < 50 else f"OPTIMIZE PATENT ROUTE - Shows potential with improvements" if viability_assessment['score'] < 75 else "PROCEED WITH PATENT ROUTE - Commercially viable"}

**Key Success Factors:**
1. [Most critical factor for success]
2. [Second most important factor]  
3. [Third critical factor]

**Next Steps:**
1. [Immediate action item]
2. [Follow-up task]
3. [Long-term goal]

CRITICAL INSTRUCTIONS:
- Always clearly separate "Patent Data" from "AI Optimization"
- If patent route isn't viable, provide practical alternatives
- Focus on realistic, implementable solutions
- Include actual supplier names and contact information when possible
- Provide quantitative assessments wherever possible
"""
        
        return Agent(
            name="Enhanced Commercial Synthesis Expert",
            role="Generate commercially viable synthesis routes with comprehensive business analysis",
            model=Groq(id=selected_model, api_key=self.GROQ_API_KEY),
            instructions=enhanced_instructions,
            markdown=True,
        )
    
    def _create_synthesis_chatbot(self, selected_model, session_data):
        context = session_data.get('results', {})
        context_string = ""
        
        api_name_from_session = session_data.get('api_name', 'Unknown')
        analysis_date_from_session = session_data.get('timestamp', 'Unknown')
        ai_predicted_route = session_data.get('ai_predicted_route', 'Not available yet.')
        prediction_complete = session_data.get('prediction_complete', False)

        context_string = f"""
CURRENT SYNTHESIS CONTEXT:
- API Name: {api_name_from_session}
- Analysis Date: {analysis_date_from_session}
"""

        if context and context.get('analysis_complete'):
            context_string += f"""
- Commercial Viability Score: {context.get('viability_assessment', {}).get('score', 'N/A')}/100
- Patent Title: {context.get('patent_data', {}).get('title', 'Unknown')}
- Synthesis Sections Found: {context.get('patent_data', {}).get('synthesis_sections_found', 0)}

SYNTHESIS CONTENT SUMMARY:
{context.get('patent_data', {}).get('synthesis_content', '')[:2000]}

VIABILITY ASSESSMENT DETAILS:
- Positive Indicators: {', '.join(context.get('viability_assessment', {}).get('positive_indicators', []))}
- Negative Indicators: {', '.join(context.get('viability_assessment', {}).get('negative_indicators', []))}
- Maximum Yield Found: {context.get('viability_assessment', {}).get('yield_info', {}).get('max_yield', 0)}%
"""

        if ai_predicted_route != 'Not available yet.':
            context_string += f"""
AI PREDICTED ROUTE (STATUS: {'Complete' if prediction_complete else 'In Progress'}):
{ai_predicted_route[:2000]}
"""
        instructions = f"""
You are a specialized pharmaceutical synthesis expert chatbot with deep knowledge of the current analysis.

{context_string}

🎯 YOUR ROLE:
- Answer follow-up questions about the synthesis route analysis, including both the actual patent-derived route and the AI-predicted route.
- Provide additional insights and clarifications
- Help with optimization strategies and alternatives
- Explain technical details in depth
- Suggest improvements or variations

🔬 KEY CAPABILITIES:
- Detailed explanation of synthesis steps and mechanisms for both actual and AI-predicted routes
- Alternative route suggestions for better yields/economics
- Raw material sourcing and supplier recommendations
- Scale-up considerations and process optimization
- Regulatory and safety aspects
- Cost analysis and commercial viability improvements

📋 RESPONSE GUIDELINES:
- Always reference the current analysis context when relevant, explicitly distinguishing between 'Patent Data' (actual route) and 'AI Predicted Route' information.
- Be specific and technical when appropriate
- Provide actionable recommendations
- If asked about different APIs, clarify that analysis would need to be done separately
- Maintain conversation continuity with previous exchanges
- Use clear headings and bullet points for complex answers

🚨 IMPORTANT NOTES:
- If user asks about a completely different API, recommend running a new analysis
- For compound name variations (like isoflavuconazole vs ravuconazole), explain the relationship
- Always distinguish between parent compounds and their derivatives/salts
- Reference specific patent examples when discussing synthesis steps

You have access to the complete analysis context above, including both the actual synthesis analysis and the AI-predicted route. Provide helpful, accurate, and detailed responses to user questions.
"""
        return Agent(
            name="Synthesis Expert Chatbot",
            role="Provide detailed follow-up answers about synthesis routes and commercial viability",
            model=Groq(id=selected_model, api_key=self.GROQ_API_KEY),
            instructions=instructions,
            markdown=True,
        )

    def chat_response(self, message, session_data):
        # Use session_data directly for api_name and timestamp
        api_name = session_data.get('api_name', 'Unknown')
        timestamp = session_data.get('created_at', 'Unknown') # Use created_at from session data
        
        selected_model = "llama-3.3-70b-versatile"

        chatbot = self._create_synthesis_chatbot(selected_model, session_data)
        
        enhanced_question = f"""
USER QUESTION: {message}

CURRENT CONTEXT: We are discussing the synthesis of {api_name} based on the analysis initiated at {timestamp}. """
        if session_data.get('analysis_complete') and session_data.get('prediction_complete'):
            enhanced_question += "Both a patent-derived route and an AI-predicted route are available in the context. "
        elif session_data.get('analysis_complete'):
            enhanced_question += "A patent-derived route is available in the context. "
        elif session_data.get('prediction_complete'):
            enhanced_question += "An AI-predicted route is available in the context. "
        enhanced_question += "Please provide a detailed, helpful response that references the specific analysis data when relevant, explicitly distinguishing between 'Patent Data' and 'AI Predicted Route' information if both are relevant to the question."

        response = chatbot.run(enhanced_question)
        return response.content

    def run_full_analysis(self, api_name, supplier_preference="", search_depth="deep", 
                          include_alternatives=True, focus_high_yield=True, viability_threshold=75,
                          progress_callback=None, stop_event=None):
        try:
            if stop_event and stop_event.is_set(): return {'success': False, 'error': 'Analysis stopped by user.'}            
            if progress_callback: progress_callback(5, 'Generating enhanced search queries...')
            search_queries = self.generate_enhanced_search_queries(api_name)
            
            all_links = []
            num_queries = len(search_queries)
            for i, query in enumerate(search_queries):
                if stop_event and stop_event.is_set(): return {'success': False, 'error': 'Analysis stopped by user.'}
                if progress_callback: progress_callback(5 + int((i / num_queries) * 15), f'Searching for patents (Query {i+1}/{num_queries})...')
                all_links.extend(self.google_cse_search(query))
                all_links.extend(self.serpapi_search(query))
                time.sleep(0.5)
            
            unique_links = list(set(all_links))
            num_unique_links = len(unique_links)
            valid_results = []
            rejected_count = 0

            if progress_callback: progress_callback(20, f'Analyzing {num_unique_links} unique patent links...')
            for i, (url, engine) in enumerate(unique_links):
                if stop_event and stop_event.is_set(): return {'success': False, 'error': 'Analysis stopped by user.'}
                if progress_callback: progress_callback(20 + int((i / num_unique_links) * 50), f'Extracting content from patent {i+1}/{num_unique_links}...')
                
                if self.is_url_valid(url):
                    relevant, score, extra_data = self._is_source_relevant_enhanced(url, api_name)
                    if relevant and extra_data:
                        valid_results.append({
                            'url': url,
                            'source': self._classify_source(url),
                            'engine': engine,
                            'score': round(score, 3),
                            'extra_data': extra_data
                        })
                    else:
                        rejected_count += 1

            if stop_event and stop_event.is_set(): return {'success': False, 'error': 'Analysis stopped by user.'}
            if progress_callback: progress_callback(70, 'Assessing commercial viability...')
            if not valid_results:
                return {
                    'error': f'No synthesis patents found for {api_name}',
                    'suggestions': [
                        'Check the spelling of the API name',
                        'Try the generic name instead of brand name',
                        'Include common salt forms (HCl, sulfate, etc.)',
                        'For derivative compounds, try searching for the parent compound'
                    ]
                }
            
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
            best_result = valid_results[0]
            best_patent_data = best_result["extra_data"]
            
            viability_assessment = self.assess_commercial_viability(
                best_patent_data.get('synthesis_content', ''), 
                api_name
            )
            
            patent_images = best_patent_data.get('images', [])

            selected_model = "llama-3.3-70b-versatile"

            synthesis_agent = self._create_synthesis_agent(
                selected_model, api_name, supplier_preference, viability_assessment
            )

            raw_input_for_agent = f"""
VERIFIED SYNTHESIS PATENT ANALYSIS FOR: {api_name}

=== PRIMARY PATENT SOURCE ===
URL: {best_result['url']}
Relevance Score: {best_result['score']:.3f}/1.0
Title: {best_patent_data.get('title', 'Unknown')}

=== PATENT ABSTRACT ===
{best_patent_data.get('abstract', 'Not available')}

=== DETAILED SYNTHESIS CONTENT (EXTRACTED FROM PATENT) ===
{best_patent_data.get('synthesis_content', '')[:6000]}

=== DETAILED DESCRIPTION RELEVANT TO SYNTHESIS ===
{best_patent_data.get('detailed_description', '')[:2000]}

=== RELEVANT PATENT CLAIMS ===
{best_patent_data.get('claims', '')[:2000]}

=== PATENT VALIDATION ===
✅ Confirmed as SYNTHESIS patent (not formulation)
✅ Enhanced content extraction completed
✅ {best_patent_data.get('synthesis_sections_found', 0)} synthesis sections identified
✅ Patent diagrams found: {len(patent_images)}
✅ Commercial viability assessed: {viability_assessment['score']}/100

=== COMMERCIAL VIABILITY CONTEXT ===
Current Route Viability: {viability_assessment['score']}/100 ({viability_assessment['level']})
Positive Indicators Found: {', '.join(viability_assessment['positive_indicators'])}
Negative Indicators Found: {', '.join(viability_assessment['negative_indicators'])}
Maximum Yield Identified: {viability_assessment['yield_info']['max_yield']}%
Key Recommendations: {'; '.join(viability_assessment['recommendations'])}

=== ADDITIONAL SOURCES ===
{len(valid_results)-1} additional relevant patents identified for cross-reference

INSTRUCTIONS: Provide comprehensive analysis focusing on commercial viability optimization. If the current route shows low viability (< {viability_threshold}), propose practical alternatives.
"""
            ai_result = synthesis_agent.run(raw_input_for_agent)

            # The frontend expects a specific structure for results
            return {
                'success': True,
                'api_name': api_name,
                'patent_data': best_patent_data,
                'patent_url': best_result["url"],
                'viability_assessment': viability_assessment,
                'ai_analysis': ai_result.content,
                'search_stats': {
                    'total_searched': len(unique_links),
                    'relevant_found': len(valid_results),
                    'success_rate': len(valid_results)/len(unique_links)*100 if unique_links else 0
                }
            }
            
        except Exception as e:
            print(f"[DEBUG] Full analysis failed: {e}")
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }

    def predict_synthesis_route(self, api_name, country_preference="", criteria="", progress_callback=None, stop_event=None):
        try:
            if stop_event and stop_event.is_set():
                return {'success': False, 'error': 'Prediction stopped by user.'}
            if progress_callback: progress_callback(5, 'Initializing AI for route prediction...')

            selected_model = "llama-3.3-70b-versatile"
            
            default_criteria = "High yield (>95%), Cost-effectiveness, Purity (≥99%), Time efficiency, Safety, Commercial viability, Best consumption coefficients"
            effective_criteria = criteria if criteria.strip() else default_criteria

            agent_instructions = """
            You are a senior process chemist working on optimizing industrial API synthesis routes.

            Find the best synthesis route for an API, optimizing for: High yield (>95%), Cost-effectiveness, Purity (≥99%), Time efficiency, Safety, Commercial viability, Best consumption coefficients.

            Use recent patents, industrial literature, and global databases, leveraging all available search tools (Google CSE, SerpAPI) for comprehensive results.

            CRITICAL TOOL USAGE NOTE: When using search tools, ensure that `max_results` is always an INTEGER (e.g., `max_results=5`), not a string (e.g., `max_results="5"`). This is essential for the tool to function correctly.
            IMPORTANT TOOL SYNTAX: When calling a tool, ensure there are NO extra characters (especially commas) immediately after the tool's name within the `<function=...>` tag. The format should be strictly `<function=tool_name>{"param": "value"}</function>`.

            Also:
            - List all required raw materials, extracting them directly from the detailed synthesis steps.
            - For each raw material, USE YOUR SEARCH TOOLS to find and provide at least 10 commercial vendors per raw material.
            - Prioritize suppliers from the specified country if one is given.
            - Include GMP (Good Manufacturing Practice) status for each vendor, if available. Use your search tools to verify this information.

            After presenting the synthesis route and vendor table, continue with:

            3️⃣ Commercial Viability Explanation
            4️⃣ Challenges or Drawbacks
            5️⃣ Suggested Alternative Steps

            Output format:

            1️⃣ Synthesis Route

            Provide a detailed, step-by-step description of the synthesis route. For EACH step, clearly delineate: 
            - **Step N:** [A concise title for the step]
            - **Reactant(s):** [Chemical name(s)]
            - **Reagents and Conditions:** [e.g., solvent, temperature, time, specific reagents]
            - **Intermediate/Product:** [Chemical name of the main product of this step]
            - **Chemical Transformation:** [A brief description of the type of reaction, e.g., Nitration, Reduction, Acylation, Cyclization]
            - **Expected Yield:** [Percentage]
            - **Expected Purity:** [Percentage]
            Ensure these details are clearly presented for each individual numbered step.

            2️⃣ Raw Materials & Vendors Table:

            | Raw Material | Vendor Name | Country | Contact Info | GMP Status | SMILES |
            |--------------|-------------|---------|--------------|------------|--------|
            | [Name]       | [Vendor]    | [Country] | [URL]      | [Yes/No]   | [SMILES] |

            3️⃣ Commercial Viability

            4️⃣ Challenges

            5️⃣ Alternative Steps

            *6️⃣ Main Source URL*

            At the very end, after all other sections, include a line:
            *Main Source URL:* [insert the exact URL of the main source you used for the synthesis route]
            If you cannot find a single main source, list all relevant URLs.

            *7️⃣ Additional Source Links (if any)*

            IMPORTANT: Only include links that are real and accessible (do not invent or guess URLs). If you cannot find a valid link, write: "No valid source found." If you provide a patent number or article title, ensure it can be found on the official site (e.g., patents.google.com, espacenet, ACS Publications).
            """
            if progress_callback: progress_callback(20, 'Generating predicted synthesis route...')
            
            prediction_agent = Agent(
                name="AI Predicted Synthesis Route Chemist",
                role="Predict and optimize industrial API synthesis routes",
                model=Groq(id=selected_model, api_key=self.GROQ_API_KEY),
                instructions=agent_instructions,
                tools=[GoogleCSESearchTool(self), SerpAPISearchTool(self)], # Removed DuckDuckGoTools
                markdown=True,
            )

            raw_agent_input = f"Predict the best synthesis route for {api_name} with the following criteria: {effective_criteria}. Prioritize suppliers from {country_preference if country_preference else 'any country'}."
            print(f"[DEBUG] Full agent instructions for prediction model:\n{agent_instructions}")
            print(f"[DEBUG] Raw agent input to prediction model: {raw_agent_input}")
            try:
                ai_prediction_result = prediction_agent.run(raw_agent_input)
                print(f"[DEBUG] Raw AI prediction agent run result: {ai_prediction_result}")
            except Exception as e:
                import traceback
                print("[DEBUG] An error occurred during prediction_agent.run:")
                traceback.print_exc() # This will print the full traceback
                raise e # Re-raise the exception to be caught by the outer block
            
            if stop_event and stop_event.is_set():
                return {'success': False, 'error': 'Prediction stopped by user.'}
            if progress_callback: progress_callback(90, 'Finalizing predicted route details...')

            return {'success': True, 'result': ai_prediction_result.content}

        except Exception as e:
            print(f"[DEBUG] AI Predicted Route failed: {e}")
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}     