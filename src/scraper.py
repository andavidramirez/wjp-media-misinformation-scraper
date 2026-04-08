"""
scraper.py — Phase 1: Data Collection
==========================================
Responsibilities:
  - Paginate through the Misinformation Detector section of La Silla Vacía.
  - Filter articles with verdict FALSO or ENGÁÑOSO published within the last
    3 calendar months (using relativedelta for exact month boundaries).
  - Extract the full content (title, lead, body) of each article.
  - Respect the server with a 6-second sleep between requests.
  - Save the raw output to data/scraped_articles.csv.
"""

import time
import random
import re
import os

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DetectorMentirasScraper:
    """Encapsulates the scraping logic for the Misinformation Detector."""

    # WordPress category URL: supports standard pagination up to page 613.
    # The custom URL /detector-de-mentiras/ only handles 3 pages with
    # standard pagination; the rest requires AJAX (incompatible with requests).
    BASE_URL = "https://www.lasillavacia.com/category/detector-de-mentiras/"

    # User-Agent and additional headers to mimic a real web browser
    # and avoid being blocked by anti-bot protections
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9",  # Keep Spanish locale to match target site
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # Maximum number of articles to collect per run
    MAX_ARTICLES = 30

    # Delay between requests (seconds) to avoid overloading the server.
    # Stochastic range [3, 5] prevents deterministic request fingerprinting
    # while maintaining a courteous crawling cadence.
    SLEEP_RANGE = (3.0, 5.0)
    
    # Retry parameters for temporary errors (429, 503, etc.)
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier

    # Month name (Spanish) → number mapping (for parsing dates from the target site)
    MONTHS = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    }

    def __init__(self, data_dir: str):
        """
        Parameters
        ----------
        data_dir : str
            Path to the folder where the raw CSV will be saved.
            Passed from main.py to preserve relative paths.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create a cloudscraper client to bypass anti-bot protections
        self.scraper = cloudscraper.create_scraper()

        # Date range: from today back exactly 3 calendar months
        self.current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.limit_date = self.current_date - relativedelta(months=3)

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos privados (helpers)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_with_retries(self, url: str, timeout: int = 10):
        """Performs a GET request with intelligent retries for temporary errors.
        
        Uses cloudscraper to bypass anti-bot protections (Cloudflare, etc.).
        Handles 429 (Too Many Requests) and 503 (Service Unavailable) with exponential backoff.
        
        Parameters
        ----------
        url : str
            URL to request
        timeout : int
            Request timeout in seconds
            
        Returns
        -------
        Response | None
            Successful response or None if all retries fail
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.scraper.get(url, timeout=timeout)
                
                # Temporary errors: retry with exponential backoff
                if response.status_code in (429, 503):
                    if attempt < self.MAX_RETRIES:
                        wait_time = self.RETRY_BACKOFF_FACTOR ** (attempt - 1)
                        print(f"  ⚠️  HTTP {response.status_code}. Retrying in {wait_time:.1f}s (attempt {attempt}/{self.MAX_RETRIES})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ❌ HTTP {response.status_code} after {self.MAX_RETRIES} retries.")
                        return None
                
                # Other HTTP errors: fail immediately
                response.raise_for_status()
                return response
                
            except Exception as exc:
                if attempt < self.MAX_RETRIES:
                    wait_time = self.RETRY_BACKOFF_FACTOR ** (attempt - 1)
                    print(f"  ⚠️  Error ({type(exc).__name__}): {str(exc)[:60]}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Failed after {self.MAX_RETRIES} attempts: {exc}")
                    return None
        
        return None
    
    # ──────────────────────────────────────────────────────────────────────────
    # Parsing and classification
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_date(self, time_tag) -> datetime | None:
        """Extracts the date from an HTML <time> tag.

        Tries the 'datetime' attribute in ISO format first (reliable when present).
        Falls back to parsing the visible text (e.g., 'Abril 1, 2026').
        Returns a datetime object or None if parsing fails.
        """
        if time_tag is None:
            return None

        # Option 1: ISO datetime attribute (e.g., "2026-04-01T14:30:00")
        dt_attr = time_tag.get("datetime", "")
        if dt_attr:
            try:
                return datetime.fromisoformat(dt_attr[:10])
            except ValueError:
                pass

        # Option 2: visible Spanish text (e.g., "Abril 1, 2026")
        try:
            raw = time_tag.get_text(strip=True).lower().replace(",", "").strip()
            partes = raw.split()
            if len(partes) >= 3:
                mes = self.MONTHS.get(partes[0])
                dia = int(partes[1])
                anio = int(partes[2])
                if mes:
                    return datetime(anio, mes, dia)
        except Exception as exc:
            print(f"  ⚠  Error parsing date '{time_tag.get_text(strip=True)}': {exc}")

        return None

    def _get_label(self, card, article_url: str) -> str | None:
        """Determines whether an article is FALSO (false) or ENGÁÑOSO (misleading).

        Strategy 1: reads the div.cat-links from the card HTML (most reliable).
        Strategy 2: searches for /falso/ or /enganoso/ patterns in the URL.
        Returns 'FALSO', 'ENGÁÑOSO', or None if the article belongs to neither.
        """
        cat_links = card.find("div", class_="cat-links")
        if cat_links:
            label = cat_links.get_text(strip=True).upper()
            if label in ("FALSO", "ENGAÑOSO"):
                return label

        url_lower = article_url.lower()
        if "/falso/" in url_lower:
            return "FALSO"
        if "/enganoso/" in url_lower or "/engañoso/" in url_lower:
            return "ENGAÑOSO"

        return None

    def _clean_boilerplate(self, text: str) -> str:
        """Removes boilerplate text that La Silla Vacía appends to every article
        (calls to action, IFCN disclaimers, etc.) that would skew the analysis.
        """
        boilerplate_patterns = [
            r"Escr[ií]banos al\s*DetectBot.*?para usted\.?",
            r"La Silla Vac[ií]a es parte del International Fact\-Checking Network.*?conocer ac[aá]\.?",
            r"International Fact\-Checking Network.*?c[oó]digo de principios.*?\.?",
            r"Nuestro equipo de periodistas la verificar[aá] para usted\.?",
            r"firmamos y acatamos un c[oó]digo de principios.*?\.?",
        ]
        for pat in boilerplate_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s{2,}", " ", text).strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Collect article URLs from the general listing
    # ──────────────────────────────────────────────────────────────────────────

    def get_article_urls(self) -> list[dict]:
        """Paginates through the Misinformation Detector listing from newest to oldest,
        collecting up to MAX_ARTICLES articles labeled FALSO or ENGÁÑOSO
        published within the 3-month date range.

        Returns
        -------
        list[dict]
            List of dicts with keys: url, verdict_list, date.
        """
        articles_data = []
        seen_urls: set[str] = set()
        page = 1
        keep_scraping = True

        print(f"📅 Date range: {self.limit_date.strftime('%Y-%m-%d')} → {self.current_date.strftime('%Y-%m-%d')}")
        print(f"🎯 Target: {self.MAX_ARTICLES} articles\n")

        while keep_scraping and len(articles_data) < self.MAX_ARTICLES:
            url = self.BASE_URL if page == 1 else f"{self.BASE_URL}page/{page}/"
            print(f"📄 Crawling page {page}: {url}")

            try:
                response = self._get_with_retries(url, timeout=10)
                if response is None:
                    print(f"  🛑 Could not retrieve page {page}. Stopping.")
                    break
                    
                soup = BeautifulSoup(response.text, "html.parser")

                cards = soup.find_all("article")
                if not cards:
                    print(f"  No articles found on page {page}. End of listing.")
                    break

                print(f"  → {len(cards)} article cards found.")

                for card in cards:
                    if len(articles_data) >= self.MAX_ARTICLES:
                        keep_scraping = False
                        break

                    title_tag = card.find("h2", class_="entry-title") or card.find("h2")
                    if not title_tag:
                        continue
                    a_tag = title_tag.find("a")
                    if not a_tag:
                        continue
                    article_url = a_tag["href"]

                    pub_date = self._parse_date(card.find("time"))

                    if pub_date and pub_date < self.limit_date:
                        print(f"  🛑 Article outside date range ({pub_date.strftime('%Y-%m-%d')}). Stopping.")
                        keep_scraping = False
                        break

                    label = self._get_label(card, article_url)
                    if label is None:
                        continue

                    if article_url not in seen_urls:
                        seen_urls.add(article_url)
                        articles_data.append({
                            "url": article_url,
                            "verdict_list": label,
                            "date": pub_date.strftime("%Y-%m-%d") if pub_date else "Unknown",
                        })
                        print(f"  ✅ ({len(articles_data)}/{self.MAX_ARTICLES}) [{label}] {article_url}")

            except Exception as exc:
                print(f"  ❌ Unexpected error on page {page}: {exc}")
                break

            page += 1
            time.sleep(random.uniform(*self.SLEEP_RANGE))

        print(f"\n🏁 Collection complete: {len(articles_data)} articles recorded.")
        return articles_data

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Deep crawl — extract full content from each article
    # ──────────────────────────────────────────────────────────────────────────

    def deep_crawl(self, articles: list[dict]) -> pd.DataFrame:
        """Visits each collected URL and extracts the title, lead, and full body text.

        Parameters
        ----------
        articles : list[dict]
            List of dicts produced by get_article_urls().

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: url, date, title, verdict, lead, full_text.
        """
        scraped_data = []

        for idx, item in enumerate(articles, start=1):
            print(f"[{idx}/{len(articles)}] Extracting: {item['url']}")
            try:
                response = self._get_with_retries(item["url"], timeout=12)
                if response is None:
                    print(f"  ❌ Could not download article (permanent error).")
                    continue
                    
                soup = BeautifulSoup(response.text, "html.parser")

                # Title
                h1 = soup.find("h1", class_="entry-title") or soup.find("h1")
                title = h1.get_text(strip=True) if h1 else "Title not found"

                # Lead text (meta description)
                lead_tag = (
                    soup.find("meta", attrs={"name": "description"})
                    or soup.find("meta", attrs={"property": "og:description"})
                )
                lead = lead_tag.get("content", "").strip() if lead_tag else ""

                # Verdict (prioritize the value from the article page itself)
                verdict = item.get("verdict_list", "Unknown")
                cat_div = soup.find("div", class_="cat-links")
                if cat_div:
                    cats = cat_div.get_text(" ", strip=True).upper()
                    if "FALSO" in cats:
                        verdict = "FALSO"
                    elif "ENGAÑOSO" in cats:
                        verdict = "ENGAÑOSO"

                # Article body
                content_div = soup.find("div", class_="entry-content")
                if content_div:
                    paragraphs = content_div.find_all("p")
                    full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
                    full_text = self._clean_boilerplate(full_text)
                else:
                    full_text = ""

                scraped_data.append({
                    "url": item["url"],
                    "date": item.get("date", "Unknown"),
                    "title": title,
                    "verdict": verdict,
                    "lead": lead,
                    "full_text": full_text,
                })
                print(f"  ✅ {title[:80]}...")

            except Exception as exc:
                print(f"  ❌ Error processing {item['url']}: {exc}")

            time.sleep(random.uniform(*self.SLEEP_RANGE))

        print(f"\n✅ Deep crawl complete: {len(scraped_data)} articles extracted.")
        return pd.DataFrame(scraped_data)

    # ──────────────────────────────────────────────────────────────────────────
    # Save raw data
    # ──────────────────────────────────────────────────────────────────────────

    def save_raw(self, df: pd.DataFrame) -> str:
        """Persists the DataFrame to data/scraped_articles.csv.

        Returns the absolute path of the saved file.
        """
        csv_path = os.path.join(self.data_dir, "scraped_articles.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 CSV saved to: {csv_path}")
        return csv_path
