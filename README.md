# Detector de Mentiras — Web Scraping and Analysis Pipeline

A modular Python pipeline for extracting, processing, and visualizing articles from the **Detector de Mentiras** (Lie Detector) section of [La Silla Vacía](https://www.lasillavacia.com/detector-de-mentiras/), a Colombian investigative journalism outlet. The final output is a professional PDF report containing 5 visualizations.

---

## Project Structure

```
.
├── main.py                  # Orchestrator: runs the full pipeline
├── requirements.txt         # Project dependencies
├── .gitignore
├── data/                    # Temporary CSV (deleted at the end of the pipeline)
├── output/                  # Final PDF report (persists after execution)
└── src/
    ├── __init__.py
    ├── scraper.py           # Phase 1 — Web scraping and data collection
    ├── processor.py         # Phase 2 — Text cleaning and word frequencies
    └── visualizer.py        # Phase 3 — Visualizations and PDF export
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> NLTK resources (`stopwords`) are downloaded automatically the first time the pipeline runs.

---

## Usage

From the project root (with the virtual environment active):

```bash
python main.py
```

### Expected Output

```
=================================================================
  PIPELINE — Detector de Mentiras (La Silla Vacía)
=================================================================

📡 PHASE 1 — Scraping
  📅 Period: 2026-01-07 — 2026-04-07
  🎯 Target: 30 articles
  📄 Crawling page 1 ...
  ...

🔤 PHASE 2 — Text processing
  🔍 Top 15 most frequent words: ...

📊 PHASE 3 — Generating PDF report
  ✅ PDF report saved to: output/Misinformation_Report_LaSillaVacia.pdf

🗑️  PHASE 4 — Raw data cleanup
  ✅ Deleted: scraped_articles.csv

=================================================================
  ✅ PIPELINE COMPLETE
  📄 Report available at: output/Misinformation_Report_LaSillaVacia.pdf
=================================================================
```

---

## Generated Visualizations

| # | Visualization | Description |
|---|---|---|
| Cover | Report cover page | Title, date range, and total article count |
| VIZ 1 | Lollipop chart | Top 20 most frequent words in the corpus |
| VIZ 2 | Vertical bar chart | Verdict distribution — FALSO vs ENGAÑOSO |
| VIZ 3 | Stacked bar chart | Articles published per date |
| VIZ 4 | Word cloud | Lexical density of the full corpus |
| VIZ 5 | Horizontal bar chart | Top 10 words compared by verdict |

---

## Architecture

The project follows a **modular pipeline architecture** with strict separation of concerns:

```
main.py
  └── scraper.py    →  Raw DataFrame
  └── processor.py  →  Frequency counters + top terms per verdict
  └── visualizer.py →  PDF with 5 visualizations
```

### Design Decisions

| Decision | Rationale |
|---|---|
| **One class per module** | Facilitates unit testing and independent extension of each phase |
| **Relative paths via `os.path`** | The project is portable across machines without path modifications |
| **`relativedelta` for date ranges** | Ensures exact calendar months (not fixed 30-day periods) |
| **4-second sleep between requests** | Respects the server’s capacity and avoids rate-limiting blocks |
| **Data cleanup at the end** | Data minimization principle: no unnecessary intermediate data is retained |
| **`Agg` backend in matplotlib** | Enables headless execution on servers without a display |
| **Manually enriched stopwords** | Journalism/politics vocabulary requires terms beyond the NLTK default set |

---

## System Requirements

- Python ≥ 3.10
- Internet connection (for scraping and downloading NLTK stopwords)

---

## Ethical Behavior

- The pipeline enforces a **4-second delay** between each HTTP request.
- Upon completion, it **automatically deletes** the CSV file containing raw scraped data.
- No personal data from article authors or site users is stored.
