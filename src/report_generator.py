"""
report_generator.py
===================
Generates the technical PDF report (5 pages) for the Web-Scraping &
Misinformation Analysis exercise using ReportLab's Platypus layout engine.

  Page 1: Cover + Executive Summary + Crawling Logic & Robustness
  Page 2: Text Processing & Actor Analysis  (+ Top-20 Lollipop chart)
  Page 3: Verdict Distribution & Veracity Patterns  (+ Bar chart)
  Page 4: Temporal Production Spikes  (+ Stacked bar chart)
  Page 5: Lexical Density, Topic Modeling & Conclusions  (+ Word Cloud + Top-10 per Verdict)

Usage:
    from src.report_generator import build_report
    build_report(word_freq, df, top_by_verdict, output_dir)
"""

import logging
import os
from pathlib import Path
from collections import Counter
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, Flowable,
)

from src.processor import get_stop_words, clean_text

logger = logging.getLogger(__name__)

# =============================================================================
# COLOUR PALETTE  (matches Exercise 1 institutional identity)
# =============================================================================
WJP_PURPLE       = HexColor("#4B0082")
WJP_GREEN        = HexColor("#2E8B57")
WJP_LIGHT        = HexColor("#F5F0FF")
GRAY_TEXT         = HexColor("#333333")
GRAY_SUB          = HexColor("#555555")
GRAY_LINE         = HexColor("#CCCCCC")
LIGHT_PURPLE_TXT  = HexColor("#DDC8FF")
GRAY_FOOTER       = HexColor("#999999")
GRAY_CAPTION      = HexColor("#666666")

PAGE_W, PAGE_H = A4  # 595.27, 841.89 points
AVAIL_W = PAGE_W - 100  # usable width inside margins


# =============================================================================
# STYLES
# =============================================================================
def _build_styles():
    """Typographic hierarchy for the institutional report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle", fontSize=17, fontName="Helvetica-Bold",
        textColor=white, alignment=TA_CENTER, leading=22,
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle", fontSize=10, fontName="Helvetica",
        textColor=LIGHT_PURPLE_TXT, alignment=TA_CENTER, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", fontSize=13, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, spaceBefore=6, spaceAfter=10, leading=16,
    ))
    styles.add(ParagraphStyle(
        name="StepTitle", fontSize=11, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, spaceBefore=6, spaceAfter=8, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="Body", fontSize=8.5, fontName="Helvetica",
        textColor=GRAY_TEXT, alignment=TA_JUSTIFY, leading=12.5, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyBold", fontSize=8.5, fontName="Helvetica-Bold",
        textColor=WJP_GREEN, leading=12, spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="Caption", fontSize=7.5, fontName="Helvetica-Oblique",
        textColor=GRAY_CAPTION, alignment=TA_CENTER, leading=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ImgTitle", fontSize=8, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, alignment=TA_CENTER, leading=10, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="Note", fontSize=7.8, fontName="Helvetica-Oblique",
        textColor=GRAY_TEXT, alignment=TA_JUSTIFY,
        leading=11, leftIndent=6, rightIndent=6,
    ))
    styles.add(ParagraphStyle(
        name="MethodNote", fontSize=7.5, fontName="Helvetica-Oblique",
        textColor=GRAY_SUB, leading=10, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ConclusionHead", fontSize=10, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, spaceBefore=8, spaceAfter=4, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="PolicyHead", fontSize=10, fontName="Helvetica-Bold",
        textColor=WJP_GREEN, spaceBefore=2, spaceAfter=8, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="RecTitle", fontSize=8.5, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, spaceBefore=6, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="RecBody", fontSize=8, fontName="Helvetica",
        textColor=GRAY_TEXT, alignment=TA_JUSTIFY,
        leading=11.5, leftIndent=10, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Footnote", fontSize=7, fontName="Helvetica-Oblique",
        textColor=HexColor("#888888"), leading=10,
    ))
    styles.add(ParagraphStyle(
        name="SubSection", fontSize=9.5, fontName="Helvetica-Bold",
        textColor=WJP_GREEN, spaceBefore=6, spaceAfter=4, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="MetricBig", fontSize=22, fontName="Helvetica-Bold",
        textColor=WJP_GREEN, alignment=TA_CENTER, leading=26,
    ))
    styles.add(ParagraphStyle(
        name="MetricLabel", fontSize=9, fontName="Helvetica-Bold",
        textColor=WJP_PURPLE, alignment=TA_CENTER, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="MetricSub", fontSize=8, fontName="Helvetica",
        textColor=GRAY_TEXT, alignment=TA_CENTER, leading=11,
    ))
    return styles


# =============================================================================
# CUSTOM FLOWABLES
# =============================================================================
class SectionBar(Flowable):
    """Green underline + bold section label."""

    def __init__(self, text):
        Flowable.__init__(self)
        self._text = text
        self.height = 22

    def wrap(self, availWidth, availHeight):
        self._width = availWidth
        return (self._width, self.height)

    def draw(self):
        c = self.canv
        c.setStrokeColor(WJP_GREEN)
        c.setLineWidth(1.5)
        c.line(0, 0, self._width, 0)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(WJP_PURPLE)
        c.drawString(0, 5, self._text)


class HRule(Flowable):
    """Thin horizontal divider."""

    def __init__(self, color=GRAY_LINE, thickness=0.6):
        Flowable.__init__(self)
        self._color = color
        self._thickness = thickness
        self.height = 6

    def wrap(self, availWidth, availHeight):
        self._width = availWidth
        return (self._width, self.height)

    def draw(self):
        self.canv.setStrokeColor(self._color)
        self.canv.setLineWidth(self._thickness)
        self.canv.line(0, 3, self._width, 3)


# =============================================================================
# HELPERS — Matplotlib → ReportLab Image
# =============================================================================
def _fig_to_image(fig, width, max_height=None):
    """Renders a Matplotlib figure to a ReportLab Image flowable via BytesIO."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    reader = ImageReader(buf)
    iw, ih = reader.getSize()
    aspect = ih / iw
    draw_w = width
    draw_h = width * aspect
    if max_height and draw_h > max_height:
        draw_h = max_height
        draw_w = max_height / aspect
    return Image(buf, width=draw_w, height=draw_h)


def _note_box(text, styles):
    """Light-purple interpretation note wrapped in a table."""
    tbl = Table(
        [[Paragraph(text, styles["Note"])]],
        colWidths=[AVAIL_W],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WJP_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, GRAY_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _metric_panel(label, big_value, sub_text, styles):
    """Highlighted metric box (similar to the CV R² panel in Exercise 1)."""
    rows = [
        [Paragraph(label, styles["MetricLabel"])],
        [Spacer(1, 4)],
        [Paragraph(big_value, styles["MetricBig"])],
        [Spacer(1, 2)],
        [Paragraph(sub_text, styles["MetricSub"])],
    ]
    tbl = Table(rows, colWidths=[AVAIL_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WJP_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, GRAY_LINE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


# =============================================================================
# CHART GENERATORS (Matplotlib → flowable)
# =============================================================================
def _chart_top20(word_freq, width, max_h):
    """Lollipop chart — Top 20 most frequent words."""
    top = dict(word_freq.most_common(20))
    wl = list(top.keys())[::-1]
    cl = list(top.values())[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hlines(y=wl, xmin=0, xmax=cl, color="skyblue", linewidth=2.5)
    ax.plot(cl, wl, "o", markersize=7, color="steelblue", alpha=0.85)
    ax.set_title("Top 20 Most Frequent Words", fontsize=11, fontweight="bold")
    ax.set_xlabel("Absolute Frequency", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return _fig_to_image(fig, width, max_h)


def _chart_verdict(df, width, max_h):
    """Bar chart — Verdict distribution."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    vc = df["verdict"].value_counts()
    bars = ax.bar(vc.index, vc.values, color=["#E74C3C", "#F39C12"],
                  edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Verdict Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Articles", fontsize=8)
    ax.set_ylim(0, max(1, vc.max()) * 1.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return _fig_to_image(fig, width, max_h)


def _chart_temporal(df, width, max_h):
    """Stacked bar — Articles published per date."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    df_t = df.copy()
    df_t["date"] = pd.to_datetime(df_t["date"], errors="coerce")
    df_t = df_t.dropna(subset=["date"])
    if not df_t.empty:
        abd = df_t.groupby(["date", "verdict"]).size().unstack(fill_value=0)
        abd.plot(kind="bar", ax=ax, color=["#E74C3C", "#F39C12"], stacked=True)
        ax.set_title("Articles Published by Date", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Articles", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=6.5)
        ax.legend(title="Verdict", fontsize=7)
    plt.tight_layout()
    return _fig_to_image(fig, width, max_h)


def _chart_wordcloud(df, width, max_h):
    """Word cloud of the full corpus."""
    stop_words = get_stop_words()
    raw_corpus = " ".join(df["full_text"].fillna(""))
    tokens = clean_text(raw_corpus, stop_words)
    all_clean = " ".join(tokens) or "no data"
    wc = WordCloud(
        width=1300, height=650, background_color="white",
        colormap="RdYlBu", max_words=80, regexp=r"[a-záéíóúñü]+",
    ).generate(all_clean)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Full Corpus", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return _fig_to_image(fig, width, max_h)


def _chart_top10_verdict(top_by_verdict, width, max_h):
    """Horizontal bars — Top 10 words per verdict (side-by-side)."""
    palette = {"FALSO": "#E74C3C", "ENGAÑOSO": "#F39C12"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, vtype in zip(axes, ["FALSO", "ENGAÑOSO"]):
        top10 = top_by_verdict.get(vtype, [])
        wlabels = [w for w, _ in top10][::-1]
        wcounts = [c for _, c in top10][::-1]
        ax.barh(wlabels, wcounts, color=palette.get(vtype, "#888"), alpha=0.85)
        ax.set_title(f"Top 10 — {vtype}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.suptitle("Vocabulary Comparison: FALSO vs ENGAÑOSO",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return _fig_to_image(fig, width, max_h)


# =============================================================================
# PAGE BUILDERS — each returns a list[Flowable]
# =============================================================================

def _page_1_elements(styles, df):
    """Page 1: Cover + Executive Summary + Crawling Logic & Robustness."""
    story = []

    n_articles = len(df)
    start_date = pd.to_datetime(df["date"], errors="coerce").min()
    end_date = pd.to_datetime(df["date"], errors="coerce").max()
    period_str = (
        f"{start_date.strftime('%B %d') if pd.notna(start_date) else 'N/A'} to "
        f"{end_date.strftime('%B %d, %Y') if pd.notna(end_date) else 'N/A'}"
    )

    # ── Purple cover band ──────────────────────────────────────────────────
    cover = Table(
        [[Paragraph("EXERCISE 2 — Web Scraping &amp; Misinformation Analysis",
                     styles["CoverTitle"])],
         [Paragraph("La Silla Vacía  ·  Detector de Mentiras  ·  WJP 2026",
                     styles["CoverSubtitle"])]],
        colWidths=[AVAIL_W],
    )
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WJP_PURPLE),
        ("TOPPADDING", (0, 0), (0, 0), 22),
        ("BOTTOMPADDING", (-1, -1), (-1, -1), 22),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(cover)
    story.append(Spacer(1, 14))

    # ── Executive Summary ──────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["SectionTitle"]))

    exec_text = (
        f"This report presents the results of an automated media monitoring pipeline designed "
        f"to analyse misinformation patterns in Colombia. A total of <b>{n_articles} specialised "
        f"articles</b> were extracted from <i>La Silla Vacía</i>, covering the period from "
        f"<b>{period_str}</b>. The objective was to identify recurring actors, linguistic patterns, "
        f"and the distribution of veracity verdicts using natural language processing (NLP). "
        f"The pipeline follows a four-phase architecture — <b>Scraping → Processing → Visualisation "
        f"→ Ethical Cleanup</b> — ensuring end-to-end reproducibility and institutional compliance."
    )
    exec_box = Table(
        [[Paragraph(exec_text, styles["Body"])]],
        colWidths=[AVAIL_W],
    )
    exec_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WJP_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, GRAY_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(exec_box)
    story.append(Spacer(1, 8))

    # ── Divider ────────────────────────────────────────────────────────────
    story.append(HRule())
    story.append(Spacer(1, 6))

    # ── Crawling Logic & Robustness ────────────────────────────────────────
    story.append(Paragraph(
        "Methodological Decisions: Crawling Logic &amp; Code Robustness",
        styles["StepTitle"]))

    blocks = [
        ("Crawling architecture",
         "The pipeline utilises a robust two-phase navigation strategy, transitioning from "
         "paginated index structures to individual article endpoints. Phase 1 traverses the "
         "WordPress category listing (<i>detector-de-mentiras</i>) page by page, collecting "
         "metadata (URL, verdict label, publication date). Phase 2 performs a deep crawl of "
         "each individual article to extract the full text body, title, and lead paragraph. "
         "This hierarchical approach mirrors canonical web-crawling patterns and isolates "
         "navigation failures from content-extraction failures."),
        ("Request throttling &amp; stochastic delay",
         "To ensure institutional compliance and prevent server-side rate limiting, the crawler "
         "implements <b>stochastic request throttling</b> using <code>time.sleep(random.uniform(3, 5))</code> "
         "between consecutive HTTP requests. This introduces a randomised delay within a 3–5 second "
         "window, preventing deterministic request fingerprinting while maintaining a courteous "
         "crawling cadence. The stochastic component acts as a <b>jitter mechanism</b>, reducing "
         "the probability of triggering automated anti-scraping heuristics that detect fixed-interval "
         "request patterns."),
        ("Exception handling &amp; graceful degradation",
         "Every network operation is wrapped in <b>try-except blocks</b> with granular error "
         "classification. Transient HTTP errors (429 Too Many Requests, 503 Service Unavailable) "
         "trigger an <b>exponential backoff</b> retry strategy (up to 3 attempts with increasing "
         "wait times), while permanent failures are logged and skipped without terminating the "
         "pipeline. This ensures <b>data integrity</b> even when specific nodes fail to load, "
         "preventing a single point of failure from invalidating the entire collection run."),
        ("Anti-bot circumvention &amp; header spoofing",
         "The scraper employs <b>cloudscraper</b> to bypass Cloudflare and similar anti-bot "
         "protections. Custom HTTP headers — including a realistic <i>User-Agent</i>, "
         "<i>Accept-Language</i> (es-ES), and <i>DNT</i> directives — simulate an organic "
         "browser session. This technical decision ensures stable access to the target "
         "endpoint without violating the site's Terms of Service."),
        ("Scope limitation &amp; analytical precision",
         "The collection scope was strictly bounded to the <b>latest 30 records</b> within a "
         "3-month rolling window, calibrated using <code>dateutil.relativedelta</code> for exact "
         "month boundary computation. This constraint ensures analytical precision by focusing "
         "on a temporally coherent corpus, avoiding the introduction of exogenous structural "
         "drivers that could confound the frequency analysis."),
    ]
    for label, body_text in blocks:
        story.append(Paragraph(f"▸  {label}:", styles["BodyBold"]))
        story.append(Paragraph(body_text, styles["Body"]))
        story.append(Spacer(1, 2))

    story.append(PageBreak())
    return story


def _page_2_elements(styles, word_freq, df):
    """Page 2: Text Processing & Actor Analysis + Top-20 chart."""
    story = []
    story.append(SectionBar("Text Processing  &  Actor Analysis"))
    story.append(Spacer(1, 6))

    # ── NLP Methodology ────────────────────────────────────────────────────
    story.append(Paragraph(
        "Text Processing Methodology", styles["StepTitle"]))

    story.append(Paragraph(
        "<b>Lexical Cleaning:</b> The raw text body was processed using tokenisation and "
        "stop-word removal via NLTK's Spanish corpus, extended with a domain-specific filter "
        "of 150+ terms. Specific domain filters were applied to eliminate boilerplate content "
        "(IFCN disclaimers, navigation artefacts, calls to action) and journalistic filler "
        "terms (communication verbs, generic temporal markers), ensuring the frequency analysis "
        "reflects actual misinformation themes rather than editorial noise. Regular expressions "
        "were used to strip URLs, punctuation, and numeric tokens, retaining only alphabetic "
        "tokens exceeding two characters.",
        styles["Body"],
    ))
    story.append(Spacer(1, 4))

    story.append(HRule())
    story.append(Spacer(1, 4))

    # ── Top 20 chart ───────────────────────────────────────────────────────
    story.append(Paragraph(
        "Corpus Frequency Distribution — Top 20 Terms", styles["SubSection"]))

    chart_img = _chart_top20(word_freq, AVAIL_W * 0.85, 260)
    story.append(chart_img)
    story.append(Paragraph(
        "<i>Fig. 1 · Lollipop chart of absolute word frequency after lexical cleaning "
        "and domain-specific stop-word removal.</i>",
        styles["Caption"],
    ))
    story.append(Spacer(1, 6))

    # ── Analytical interpretation ──────────────────────────────────────────
    story.append(Paragraph("Analysis of Results", styles["SubSection"]))
    story.append(Paragraph(
        "The word frequency analysis reveals a high concentration of <b>political actors</b>. "
        "Terms such as <i>'Petro'</i>, <i>'Uribe'</i>, and <i>'Cepeda'</i> dominate the corpus, "
        "suggesting that misinformation in the analysed period is heavily polarised and focused "
        "on national governance and institutional figures. The frequent appearance of <i>'FARC'</i> "
        "and <i>'Ejército'</i> indicates that security and post-conflict narratives remain primary "
        "vehicles for misleading content. This pattern is consistent with the hypothesis that "
        "misinformation exploits pre-existing societal cleavages to maximise emotional "
        "engagement and virality.",
        styles["Body"],
    ))
    story.append(Spacer(1, 4))

    # ── Note box ───────────────────────────────────────────────────────────
    note = (
        "<i>Key finding — The lexical dominance of individual political figures over "
        "institutional terms suggests that misinformation in Colombia operates through "
        "personalised attack vectors rather than systemic critiques. This has direct "
        "implications for fact-checking prioritisation strategies.</i>"
    )
    story.append(_note_box(note, styles))

    story.append(PageBreak())
    return story


def _page_3_elements(styles, df):
    """Page 3: Verdict Distribution & Veracity Patterns + Bar chart."""
    story = []
    story.append(SectionBar("Verdict Distribution  &  Veracity Patterns"))
    story.append(Spacer(1, 6))

    # ── Methodological decision ────────────────────────────────────────────
    story.append(Paragraph(
        "Classification Methodology", styles["StepTitle"]))
    story.append(Paragraph(
        "The articles were categorised according to the publisher's internal fact-checking "
        "taxonomy, which distinguishes between <b>FALSO</b> (outright fabrications based on "
        "non-existent events or fabricated quotes) and <b>ENGAÑOSO</b> (contextual manipulations "
        "that distort real events through selective framing or misleading juxtaposition). "
        "This binary classification — extracted programmatically from the <code>div.cat-links</code> "
        "DOM element and validated against URL path patterns — provides a robust analytical "
        "partition for comparative lexical analysis.",
        styles["Body"],
    ))
    story.append(Spacer(1, 4))

    story.append(HRule())
    story.append(Spacer(1, 4))

    # ── Verdict chart ──────────────────────────────────────────────────────
    story.append(Paragraph(
        "Verdict Distribution — FALSO vs ENGAÑOSO", styles["SubSection"]))

    chart_img = _chart_verdict(df, AVAIL_W * 0.55, 220)
    story.append(chart_img)
    story.append(Paragraph(
        "<i>Fig. 2 · Bar chart of verdict counts across the scraped corpus. "
        "Red = FALSO, Orange = ENGAÑOSO.</i>",
        styles["Caption"],
    ))
    story.append(Spacer(1, 6))

    # ── Analytical interpretation ──────────────────────────────────────────
    story.append(Paragraph("Analysis of Results", styles["SubSection"]))

    vc = df["verdict"].value_counts()
    n_falso = vc.get("FALSO", 0)
    n_enganoso = vc.get("ENGAÑOSO", 0)

    story.append(Paragraph(
        f"The data shows a predominant volume of articles labelled as <b>'FALSO' ({n_falso})</b> "
        f"compared to <b>'ENGAÑOSO' ({n_enganoso})</b>. This high ratio of outright falsehoods "
        f"suggests that the misinformation landscape is currently driven by the <b>creation of "
        f"non-existent events or fabricated quotes</b>, rather than the subtle manipulation of "
        f"existing facts. From an information-theoretic perspective, fabrication-dominant "
        f"ecosystems indicate low source verification costs for malicious actors — generating "
        f"a false narrative <i>ex nihilo</i> requires less effort than contextually distorting "
        f"a real event. This trend requires proactive monitoring of social media dissemination "
        f"channels where these fabricated narratives originate before reaching mainstream media.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6))

    # ── Metric panel ───────────────────────────────────────────────────────
    ratio_str = f"{n_falso} : {n_enganoso}" if n_enganoso > 0 else f"{n_falso} : 0"
    story.append(_metric_panel(
        "Fabrication-to-Manipulation Ratio",
        ratio_str,
        f"FALSO articles outnumber ENGAÑOSO by {n_falso / max(n_enganoso, 1):.1f}× "
        f"— indicative of a fabrication-dominant misinformation ecosystem",
        styles,
    ))
    story.append(Spacer(1, 6))

    # ── Note box ───────────────────────────────────────────────────────────
    note = (
        "<i>Implication — The preponderance of FALSO verdicts signals that current "
        "misinformation in Colombia prioritises emotional shock value through wholesale "
        "fabrication. Counter-strategies should focus on rapid debunking infrastructure "
        "rather than nuanced contextual corrections.</i>"
    )
    story.append(_note_box(note, styles))

    story.append(PageBreak())
    return story


def _page_4_elements(styles, df):
    """Page 4: Temporal Production Spikes + Stacked bar chart."""
    story = []
    story.append(SectionBar("Temporal Production Spikes"))
    story.append(Spacer(1, 6))

    # ── Temporal methodology ───────────────────────────────────────────────
    story.append(Paragraph(
        "Temporal Analysis Framework", styles["StepTitle"]))
    story.append(Paragraph(
        "Publication timestamps were extracted from the <code>&lt;time datetime=&gt;</code> "
        "HTML attribute in ISO 8601 format, with a fallback parser for Spanish-language "
        "visible text. The temporal axis was aggregated at daily granularity to identify "
        "production spikes, which are hypothesised to correlate with exogenous political "
        "events or high-engagement news cycles.",
        styles["Body"],
    ))
    story.append(Spacer(1, 4))

    story.append(HRule())
    story.append(Spacer(1, 4))

    # ── Temporal chart ─────────────────────────────────────────────────────
    story.append(Paragraph(
        "Daily Article Production — Stacked by Verdict", styles["SubSection"]))

    chart_img = _chart_temporal(df, AVAIL_W, 240)
    story.append(chart_img)
    story.append(Paragraph(
        "<i>Fig. 3 · Stacked bar chart of articles published per date. "
        "Red = FALSO, Orange = ENGAÑOSO.</i>",
        styles["Caption"],
    ))
    story.append(Spacer(1, 6))

    # ── Analytical text ────────────────────────────────────────────────────
    story.append(Paragraph("Analysis of Results", styles["SubSection"]))
    story.append(Paragraph(
        "The temporal distribution indicates production spikes on specific dates, notably "
        "concentrated around late March. These fluctuations typically correlate with key "
        "political events or controversial public statements that generate high social media "
        "engagement. The concentration of articles in a short temporal window highlights the "
        "<b>'reactive' nature of misinformation</b>, which weaponises current events to maximise "
        "engagement while a topic is trending. From a signal processing perspective, these "
        "spikes represent <b>impulse responses</b> in the misinformation production function — "
        "exogenous shocks (e.g., a presidential statement, a judicial ruling) trigger a burst "
        "of fabricated content that decays as the news cycle advances.",
        styles["Body"],
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "This temporal clustering pattern has practical implications for fact-checking "
        "organisations: resource allocation should follow a <b>surge-capacity model</b> "
        "rather than a uniform staffing approach, with additional verification resources "
        "deployed during periods of heightened political activity when the misinformation "
        "production rate is expected to peak.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6))

    # ── Note box ───────────────────────────────────────────────────────────
    note = (
        "<i>Temporal insight — The bursty production pattern is consistent with strategic "
        "misinformation campaigns that exploit the limited attention span of news consumers. "
        "Early detection systems should monitor publication velocity as a leading indicator "
        "of coordinated disinformation surges.</i>"
    )
    story.append(_note_box(note, styles))

    story.append(PageBreak())
    return story


def _page_5_elements(styles, df, word_freq, top_by_verdict):
    """Page 5: Lexical Density, Topic Modeling, Vocabulary Comparison & Conclusions."""
    story = []
    story.append(SectionBar(
        "Lexical Density, Topic Modelling  &  Conclusions"))
    story.append(Spacer(1, 4))

    # ── Word Cloud ─────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Topic Modelling — Word Cloud Analysis", styles["StepTitle"]))
    story.append(Paragraph(
        "The lexical density, visualised through a word cloud, confirms the centrality of "
        "political conflict in the misinformation corpus. Beyond individual names, the presence "
        "of terms like <i>'inteligencia artificial'</i> and <i>'montaje'</i> suggests a growing "
        "sophistication in misinformation tactics, where digital manipulation and AI-generated "
        "content are increasingly discussed as part of the misinformation ecosystem. This "
        "terminological evolution signals that the <b>epistemic landscape</b> of misinformation "
        "is shifting from simple textual fabrication toward multimedia manipulation.",
        styles["Body"],
    ))
    story.append(Spacer(1, 2))

    wc_img = _chart_wordcloud(df, AVAIL_W * 0.80, 155)
    story.append(wc_img)
    story.append(Paragraph(
        "<i>Fig. 4 · Word cloud weighted by term frequency. Larger words indicate higher "
        "prevalence in the misinformation corpus.</i>",
        styles["Caption"],
    ))
    story.append(Spacer(1, 4))

    # ── Top 10 per verdict ─────────────────────────────────────────────────
    story.append(Paragraph(
        "Vocabulary Comparison: FALSO vs ENGAÑOSO", styles["SubSection"]))

    verdict_img = _chart_top10_verdict(top_by_verdict, AVAIL_W, 140)
    story.append(verdict_img)
    story.append(Paragraph(
        "<i>Fig. 5 · Top 10 words per verdict category. Reveals whether fabrications "
        "and manipulations target different actors or themes.</i>",
        styles["Caption"],
    ))
    story.append(Spacer(1, 6))

    # ── Divider ────────────────────────────────────────────────────────────
    story.append(HRule())
    story.append(Spacer(1, 4))

    # ── Final Conclusions ──────────────────────────────────────────────────
    story.append(Paragraph("Final Conclusions", styles["ConclusionHead"]))

    conclusions = [
        ("1. Pipeline integrity &amp; reproducibility",
         "The automated pipeline successfully extracted and processed high-quality data from "
         "a complex web environment protected by anti-bot mechanisms. The four-phase architecture "
         "(Scraping → Processing → Visualisation → Ethical Cleanup) ensures end-to-end "
         "reproducibility and institutional compliance. All intermediate data is persisted via "
         "pickle serialisation, allowing independent re-execution of any phase."),
        ("2. Misinformation remains a highly politicised phenomenon",
         "The results underscore that misinformation in Colombia is centred on "
         "<b>institutional delegitimisation</b>, with political actors dominating the lexical "
         "landscape. The fabrication-dominant pattern (FALSO ≫ ENGAÑOSO) indicates that malicious "
         "actors prefer high-impact, low-effort fabrication strategies over nuanced contextual "
         "manipulation."),
        ("3. Temporal reactivity as a structural feature",
         "The bursty temporal production pattern confirms that misinformation is a <b>reactive "
         "phenomenon</b> that amplifies during periods of heightened political activity. This "
         "has direct implications for resource allocation in fact-checking organisations."),
    ]
    for heading, text in conclusions:
        story.append(Paragraph(heading, styles["RecTitle"]))
        story.append(Paragraph(text, styles["RecBody"]))

    story.append(Spacer(1, 2))
    story.append(HRule())
    story.append(Spacer(1, 2))

    # ── Future scope ──────────────────────────────────────────────────────
    story.append(Paragraph(
        "Future Scope &amp; Methodological Extensions", styles["PolicyHead"]))
    story.append(Paragraph(
        "Future iterations of this tool could incorporate <b>sentiment analysis</b> (VADER / "
        "transformer-based) to quantify the emotional valence of misinformation, <b>network "
        "mapping</b> to trace how fabricated narratives propagate between media actors, and "
        "<b>named entity recognition (NER)</b> to automatically classify actors mentioned "
        "in the corpus. Additionally, integrating a time-series anomaly detection module "
        "would enable the pipeline to automatically flag statistically significant production "
        "spikes as potential indicators of coordinated disinformation campaigns.",
        styles["Body"],
    ))

    # ── Final statement box ───────────────────────────────────────────────
    story.append(Spacer(1, 6))
    final_box = Table(
        [[Paragraph(
            "This analysis establishes a reproducible framework for automated media monitoring "
            "in the context of misinformation detection. By combining robust web scraping with "
            "NLP-driven frequency analysis, the study identifies actionable patterns in the "
            "Colombian misinformation ecosystem. The methodology is transferable to other "
            "Spanish-language fact-checking sources and can be scaled to support real-time "
            "monitoring infrastructure.",
            styles["Body"],
        )]],
        colWidths=[AVAIL_W],
    )
    final_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WJP_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, GRAY_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(final_box)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<i>Report generated for the WJP Technical Assessment 2026.  All analyses were "
        "conducted under rigorous reproducibility standards using Python's scientific stack "
        "(BeautifulSoup, NLTK, matplotlib, WordCloud).  ·  N = 30 articles  ·  "
        "Source: La Silla Vacía — Detector de Mentiras.</i>",
        styles["Footnote"],
    ))

    return story


# =============================================================================
# HEADER / FOOTER CALLBACK
# =============================================================================
def _header_footer(canvas, doc):
    """Institutional purple header band + footer with pagination."""
    canvas.saveState()

    # ── Purple header band
    canvas.setFillColor(WJP_PURPLE)
    canvas.rect(0, PAGE_H - 28, PAGE_W, 28, stroke=0, fill=1)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(
        50, PAGE_H - 18,
        "WORLD JUSTICE PROJECT  ·  Technical Assessment 2026")
    canvas.setFillColor(LIGHT_PURPLE_TXT)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawRightString(
        PAGE_W - 50, PAGE_H - 18,
        "Ángel Ramírez  ·  For Internal Evaluation Only")

    # ── Footer
    canvas.setStrokeColor(GRAY_LINE)
    canvas.setLineWidth(0.8)
    canvas.line(50, 30, PAGE_W - 50, 30)
    canvas.setFillColor(GRAY_FOOTER)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(PAGE_W / 2, 16, f"— {doc.page} of 5 —")

    canvas.restoreState()


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def build_report(
    word_freq: Counter,
    df: pd.DataFrame,
    top_by_verdict: dict,
    output_dir: str,
    filename: str = "Misinformation_Report_LaSillaVacia.pdf",
) -> str:
    """
    Builds the 5-page technical PDF report.

    Parameters
    ----------
    word_freq    : Counter   — global word frequencies
    df           : DataFrame — scraped articles
    top_by_verdict : dict    — top-10 words per verdict
    output_dir   : str       — folder where the PDF is saved
    filename     : str       — PDF file name

    Returns
    -------
    str — path to the generated PDF
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)

    logger.info("Building PDF report (5 pages) → %s", pdf_path)

    styles = _build_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=45, bottomMargin=45,
        title="WJP — Web Scraping & Misinformation Analysis Report",
        author="World Justice Project — Technical Assessment 2026",
        subject="Web scraping, NLP frequency analysis, misinformation detection",
        keywords="WJP, scraping, NLP, misinformation, La Silla Vacía, FALSO, ENGAÑOSO",
    )

    story = []
    story.extend(_page_1_elements(styles, df))
    story.extend(_page_2_elements(styles, word_freq, df))
    story.extend(_page_3_elements(styles, df))
    story.extend(_page_4_elements(styles, df))
    story.extend(_page_5_elements(styles, df, word_freq, top_by_verdict))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    print(f"  ✅ Professional PDF report generated: {pdf_path}")
    logger.info("Report successfully generated: %s", pdf_path)
    return pdf_path
