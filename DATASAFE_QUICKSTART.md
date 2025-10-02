# DataSafe Threat Intelligence Platform - Quick Start

🚀 **Ready to use!** Your expanded DataSafe package is now integrated with your MediaMap project.

## What's Included

### 🔍 **Collectors** (Auto-fetching threat data)
- **URLhaus** - Recent malware URLs (works immediately)
- **CISA KEV** - Known Exploited Vulnerabilities (public JSON)
- **NVD CVE** - CVE database with finance-focused queries
- **PhishTank** - Phishing URLs (requires optional API key)
- **RSS Generic** - Any cyber/security RSS feed

### 🧠 **AI Processing** (Hugging Face powered)
- **Zero-shot classification** - Threat types & sectors
- **IOC extraction** - IPs, domains, hashes, CVEs
- **Executive summarization** - Key threat insights
- **Finance filtering** - Focus on financial sector threats

### 💾 **Storage**
- **SQLite database** - Local threat intelligence storage
- **Query helpers** - Filter by source, severity, sector
- **Deduplication** - Avoid storing duplicate threats

## Quick Test (30 seconds)

```bash
# 1. Install dependencies (if not already installed)
pip install transformers torch sentence-transformers feedparser

# 2. Seed your database (keeps everything during testing)
export DS_KEEP_NON_FINANCE=true
PYTHONPATH=. python scripts/ingest_finance.py

# 3. Check your results
PYTHONPATH=. python -c "
from datasafe.storage.sqlite_store import get_stats
print('Database stats:', get_stats())
"
```

## Full Production Setup

### 1. **Configure Environment**
```bash
# Copy the example environment file
cp .env.datasafe.example .env.datasafe

# Edit with your preferences
nano .env.datasafe
```

### 2. **Add Optional API Keys** (for better data)
```bash
# PhishTank (enhanced phishing data)
export PHISHTANK_CSV_URL=https://data.phishtank.com/data/YOUR_KEY/online-valid.csv

# NVD (better rate limits)
export NVD_API_KEY=your_nvd_api_key_here

# RSS feeds
export DS_RSS=https://feeds.feedburner.com/eset/blog
```

### 3. **Run Finance-Only Collection**
```bash
# Turn off test mode to focus on finance threats only
unset DS_KEEP_NON_FINANCE

# Run collection
PYTHONPATH=. python scripts/ingest_finance.py
```

### 4. **Query Your Intelligence Database**
```python
from datasafe.storage.sqlite_store import query_threats, get_stats

# Get all high-severity threats
high_threats = query_threats(severity='High', limit=10)

# Get finance-specific threats
finance_threats = query_threats(sector='Finance', limit=10)

# Database overview
print(get_stats())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DS_KEEP_NON_FINANCE` | Keep non-finance threats (testing) | `false` |
| `DATASAFE_DB_PATH` | Database file location | `datasafe.db` |
| `NVD_QUERY` | Finance-focused CVE search | `bank OR payment OR visa...` |
| `DS_RSS` | Additional RSS feed URL | None |
| `PHISHTANK_CSV_URL` | PhishTank CSV with your key | Public feed |
| `NVD_API_KEY` | NVD API key for rate limits | None |

## Integration with MediaMap

The DataSafe platform is now available in your MediaMap project:

```python
# In your Flask routes or background tasks
from datasafe.storage.sqlite_store import query_threats
from datasafe.collectors import urlhaus, cisa_kev

# Get latest threats for dashboard
recent_threats = query_threats(limit=20)

# Run collectors manually
new_urls = urlhaus.collect(limit=50)
```

## Performance Notes

- **First run**: Models download automatically (~500MB total)
- **Subsequent runs**: Much faster, models cached locally
- **Database growth**: Expect ~1MB per 1000 threat records
- **Collection time**: ~30 seconds for 200 items across all sources

## Troubleshooting

**Import errors**: Make sure to use `PYTHONPATH=.` when running scripts

**No finance data**: Set `DS_KEEP_NON_FINANCE=true` for testing

**PhishTank 404**: Normal without API key, get one at phishtank.com

**Models downloading slowly**: First run downloads ~500MB, be patient

**RSS returns 0 items**: Some feeds have different structures, this is normal

## Next Steps

1. **Schedule regular collection** - Add to cron/systemd timer
2. **Build dashboard** - Query `datasafe.db` for threat visualization
3. **Set up alerts** - Monitor high-severity threats
4. **Customize queries** - Modify `NVD_QUERY` for your sector focus
5. **Add more RSS feeds** - Expand threat intelligence sources

## File Structure

```
datasafe/
├── collectors/          # Data collection modules
│   ├── urlhaus.py      # Malware URLs
│   ├── phishtank.py    # Phishing URLs  
│   ├── cisa_kev.py     # Known exploited vulns
│   ├── nvd_cve.py      # CVE database
│   └── rss_generic.py  # RSS feed parser
├── ai/                 # ML processing
│   ├── classify.py     # Threat/sector classification
│   ├── extract.py      # IOC extraction
│   └── summarize.py    # Executive summaries
├── pipeline/           # Data processing
│   └── normalize.py    # Threat normalization
├── storage/            # Data persistence
│   └── sqlite_store.py # SQLite database
└── config.py          # Configuration
```

**Ready to protect your organization with automated threat intelligence!** 🛡️
