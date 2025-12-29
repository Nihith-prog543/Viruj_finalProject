# Synthesis Route Finder - API Manufacturer & Buyer Discovery

A comprehensive web application for finding API synthesis routes, manufacturers, and buyers in the pharmaceutical industry.

## Features

- **Synthesis Route Analysis**: AI-powered analysis of API synthesis routes from patents and literature
- **Manufacturer Discovery**: Automated discovery of API manufacturers from regulatory sources
- **Buyer Discovery**: Find finished dosage form (FDF) manufacturers and buyers
- **Regulatory Compliance**: Integration with FDA, EMA, PMDA, DCGI, MHRA, and other regulatory databases

## Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: Groq LLM, Agno Framework
- **Database**: PostgreSQL (Supabase)
- **Search**: DuckDuckGo, Tavily, Google CSE, SerpAPI
- **Web Scraping**: Crawl4AI
- **Deployment**: Railway

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL database (Supabase recommended)
- API Keys:
  - GROQ_API_KEY
  - DATABASE_URL (Supabase connection string)
  - TAVILY_API_KEY (optional)
  - GOOGLE_CSE_API_KEY (optional)
  - SERP_API_KEY (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nihith-prog543/Viruj_finalProject.git
cd Viruj_finalProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
# REQUIRED: GROQ_API_KEY and DATABASE_URL
# OPTIONAL: TAVILY_API_KEY, GOOGLE_CSE_API_KEY, SERP_API_KEY
```

**Important**: Never commit your `.env` file to Git! It contains sensitive API keys.

4. Run the application:
```bash
python app.py
```

## Deployment on Railway

1. Connect your GitHub repository to Railway
2. Add environment variables in Railway dashboard:
   - `GROQ_API_KEY`
   - `DATABASE_URL`
   - `TAVILY_API_KEY` (optional)
   - `GOOGLE_CSE_API_KEY` (optional)
   - `SERP_API_KEY` (optional)
3. Railway will automatically detect the `Procfile` and deploy

## Project Structure

```
synthesis_route_finder/
├── app.py                          # Main Flask application
├── synthesis_engine/
│   ├── analysis.py                # Synthesis route analysis
│   ├── api_manufacturer_discovery.py  # Manufacturer discovery
│   ├── api_buyer_finder.py         # Buyer discovery
│   ├── api_buyer_discovery.py     # Buyer discovery service
│   └── tools/                      # Custom tools (Crawl4AI, etc.)
├── templates/                      # HTML templates
├── static/                         # CSS, JS, images
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway deployment config
└── railway.json                    # Railway configuration
```

## License

Proprietary - All rights reserved

