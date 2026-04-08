"""
main.py — Pipeline Orchestrator
================================
Executes the 4 pipeline phases in order:
  Phase 1 — Scraping:       collect article URLs + deep crawl → CSV
  Phase 2 — Processing:     text cleaning + word frequencies
  Phase 3 — Visualization:  generate PDF report with 5 charts
  Phase 4 — Ethics:         delete raw data files

USAGE:
    python main.py

PATHS:
  All paths are relative to the directory containing this file,
  making the project portable across machines without code changes.
    data/    → temporary CSV with raw scraped data (deleted at end)
    output/  → final PDF report (persists)
"""

# ─── Phase execution controls (set to True to enable each phase) ─────────────────────────
PHASES_TO_RUN = {
    "scraping": True,
    "processing": True,
    "visualization": True,
    "cleanup": True,
}

# ─── Search configuration — adjust before each run ───────────────────────────
# START_DATE   : reference point for the search (searches backwards from this date).
# MONTHS_BACK  : how many full calendar months to look back from START_DATE.
# MAX_ARTICLES : stop collecting once this many FALSO/ENGAÑOSO articles are found.
#                The scraper will stop on whichever condition is met first.
from datetime import datetime

SEARCH_CONFIG = {
    "start_date":   datetime(2026, 4, 7),  # e.g. datetime(2025, 12, 31)
    "months_back":  3,                      # e.g. 6 for a half-year window
    "max_articles": 30,                     # e.g. 50 for a larger sample
}

import os
import glob
import sys
import pickle
import pandas as pd

# ─── Paths relative to the project directory ─────────────────────────────────
# BASE_DIR always points to the folder containing main.py,
# regardless of the current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ─── Intermediate files to share data between phases ─────────────────────────
ARTICLES_PICKLE_FILE = os.path.join(DATA_DIR, "_articles_processed.pkl")
WORD_FREQ_PICKLE_FILE = os.path.join(DATA_DIR, "_word_freq.pkl")
TOP_BY_VERDICT_PICKLE_FILE = os.path.join(DATA_DIR, "_top_by_verdict.pkl")

# ─── Import pipeline modules ───────────────────────────────────────────────────
from src.scraper import DetectorMentirasScraper
from src.processor import compute_word_frequencies, compute_top_by_verdict
from src.visualizer import generate_pdf_report
from src.report_generator import build_report


# ─── Helper functions to save/load intermediate data between phases ────────────────
def save_intermediate_data(df, word_freq, top_by_verdict):
    """Persists processed data as pickle files for reuse across pipeline phases."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        if df is not None:
            with open(ARTICLES_PICKLE_FILE, 'wb') as f:
                pickle.dump(df, f)
        if word_freq is not None:
            with open(WORD_FREQ_PICKLE_FILE, 'wb') as f:
                pickle.dump(word_freq, f)
        if top_by_verdict is not None:
            with open(TOP_BY_VERDICT_PICKLE_FILE, 'wb') as f:
                pickle.dump(top_by_verdict, f)
        print("  ✅ Intermediate data saved for subsequent phases")
    except Exception as e:
        print(f"  ⚠️  Error saving intermediate data: {e}")


def load_intermediate_data():
    """Loads processed data from pickle files if they exist."""
    df = None
    word_freq = None
    top_by_verdict = None
    
    try:
        if os.path.exists(ARTICLES_PICKLE_FILE):
            with open(ARTICLES_PICKLE_FILE, 'rb') as f:
                df = pickle.load(f)
        if os.path.exists(WORD_FREQ_PICKLE_FILE):
            with open(WORD_FREQ_PICKLE_FILE, 'rb') as f:
                word_freq = pickle.load(f)
        if os.path.exists(TOP_BY_VERDICT_PICKLE_FILE):
            with open(TOP_BY_VERDICT_PICKLE_FILE, 'rb') as f:
                top_by_verdict = pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  Error loading intermediate data: {e}")
    
    return df, word_freq, top_by_verdict


def run_pipeline() -> None:
    """Orchestrates all 4 pipeline phases from start to finish."""

    print("=" * 65)
    print("  PIPELINE — Misinformation Detector (La Silla Vacía)")
    print("=" * 65)

    df_articles = None
    word_freq = None
    top_by_verdict = None
    pdf_path = None

        # ── PHASE 1: Scraping ──────────────────────────────────────────────────────────
    if PHASES_TO_RUN["scraping"]:
        print("\n📡 PHASE 1 — Scraping\n")
        scraper = DetectorMentirasScraper(
            data_dir=DATA_DIR,
            start_date=SEARCH_CONFIG["start_date"],
            months_back=SEARCH_CONFIG["months_back"],
            max_articles=SEARCH_CONFIG["max_articles"],
        )

        # Collect article URLs from the listing page
        urls = scraper.get_article_urls()
        if not urls:
            print("⚠  No articles found within the date range. Aborting.")
            sys.exit(1)

        # Extract full content from each article
        df_articles = scraper.deep_crawl(urls)
        if df_articles.empty:
            print("⚠  Deep crawl returned no results. Aborting.")
            sys.exit(1)

        # Save raw data to data/
        csv_path = scraper.save_raw(df_articles)
    else:
        print("\n⏭️  PHASE 1 — Scraping (SKIPPED)\n")
        # Attempt to load data from CSV or pickle if they exist
        if os.path.exists(ARTICLES_PICKLE_FILE):
            try:
                with open(ARTICLES_PICKLE_FILE, 'rb') as f:
                    df_articles = pickle.load(f)
                print(f"  ✅ Data loaded from: _articles_processed.pkl")
            except Exception as e:
                print(f"  ⚠️  Error loading pickle: {e}")
        
        if df_articles is None:
            csv_files = glob.glob(os.path.join(DATA_DIR, "articles_*.csv"))
            if csv_files:
                csv_path = sorted(csv_files)[-1]  # Load the most recent file
                df_articles = pd.read_csv(csv_path)
                print(f"  ✅ Data loaded from: {os.path.basename(csv_path)}")

    # ── PHASE 2: Processing ─────────────────────────────────────────────────────
    if PHASES_TO_RUN["processing"]:
        if df_articles is None:
            print("⚠  Scraping data is required. Run PHASE 1 first.")
            sys.exit(1)

        print("\n🔤 PHASE 2 — Text Processing\n")
        word_freq = compute_word_frequencies(df_articles)
        top_by_verdict = compute_top_by_verdict(df_articles, n=10)

        # Save intermediate data to allow running Phase 3 independently
        save_intermediate_data(df_articles, word_freq, top_by_verdict)

        # Console preview: top 15 words with visual bar
        print("\n🔍 Top 15 most frequent words:")
        for word, freq in word_freq.most_common(15):
            bar_visual = "█" * (freq // 5)
            print(f"  {word:<22} {freq:>4}  {bar_visual}")
    else:
        print("\n⏭️  PHASE 2 — Processing (SKIPPED)\n")
        # If Phase 2 is skipped but other phases run, attempt to load processed data
        if PHASES_TO_RUN["visualization"] or PHASES_TO_RUN["cleanup"]:
            df_loaded, word_freq_loaded, top_by_verdict_loaded = load_intermediate_data()
            if df_articles is None:
                df_articles = df_loaded
            if word_freq is None:
                word_freq = word_freq_loaded
            if top_by_verdict is None:
                top_by_verdict = top_by_verdict_loaded

    # ── PHASE 3: Visualization ─────────────────────────────────────────────────────
    if PHASES_TO_RUN["visualization"]:
        # Load intermediate data if not already available
        if word_freq is None or df_articles is None:
            df_loaded, word_freq_loaded, top_by_verdict_loaded = load_intermediate_data()
            if df_articles is None:
                df_articles = df_loaded
            if word_freq is None:
                word_freq = word_freq_loaded
            if top_by_verdict is None:
                top_by_verdict = top_by_verdict_loaded
        
        # Validate that all required data is available
        if word_freq is None or df_articles is None:
            print("❌ PHASE 3 — Visualization (ABORTED)")
            print("   Processed data is required. Expected files:")
            print(f"   - Articles: {ARTICLES_PICKLE_FILE}")
            print(f"   - Word Freq: {WORD_FREQ_PICKLE_FILE}")
            print("   Run first: scraping=True, processing=True")
            sys.exit(1)

        print("\n📊 PHASE 3 — Generating PDF report\n")
        pdf_path = build_report(
            word_freq=word_freq,
            df=df_articles,
            top_by_verdict=top_by_verdict,
            output_dir=OUTPUT_DIR,
        )
    else:
        print("\n⏭️  PHASE 3 — Visualization (SKIPPED)\n")

    # ── PHASE 4: Ethics — delete raw data files ────────────────────────────────────
    if PHASES_TO_RUN["cleanup"]:
        print("\n🗑️  PHASE 4 — Deleting raw data files\n")
        patterns = ["*.csv", "*.xlsx"]
        deleted_files = 0
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(DATA_DIR, pattern)):
                try:
                    os.remove(file_path)
                    print(f"  ✅ Deleted: {os.path.basename(file_path)}")
                    deleted_files += 1
                except OSError as exc:
                    print(f"  ❌ Could not delete {os.path.basename(file_path)}: {exc}")

        if deleted_files == 0:
            print("  (No raw data files found to delete.)")
    else:
        print("\n⏭️  PHASE 4 — Cleanup (SKIPPED)\n")

    # ── Final summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅ PIPELINE COMPLETE")
    if pdf_path:
        print(f"  📄 Report available at: {pdf_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
