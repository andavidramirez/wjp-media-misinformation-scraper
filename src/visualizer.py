"""
visualizer.py — Phase 3b: Visualizations and PDF Export
=============================================================
Responsibilities:
  - Receive word frequencies and the articles DataFrame.
  - Generate 5 exploratory visualizations:
      VIZ 1 — Lollipop chart: Top 20 most frequent words.
      VIZ 2 — Bar chart: Verdict distribution — FALSO vs ENGAÑOSO.
      VIZ 3 — Stacked bar: Articles published per date.
      VIZ 4 — Word cloud of the full corpus.
      VIZ 5 — Horizontal bars: Top 10 words per verdict.
  - Compile all visualizations into a single professional PDF.
  - The output folder is received as a parameter (no hardcoded paths).
"""

import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")          # Non-GUI backend (required outside notebooks)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from wordcloud import WordCloud

from src.processor import get_stop_words, clean_text


def generate_pdf_report(
    word_freq: Counter,
    df: pd.DataFrame,
    top_by_verdict: dict,
    output_dir: str,
) -> str:
    """Generates the PDF report with all 5 visualizations.

    Parameters
    ----------
    word_freq : Counter
        Global word frequencies (produced by processor.py).
    df : pd.DataFrame
        Scraped articles DataFrame (columns: date, verdict, full_text, …).
    top_by_verdict : dict
        Top-10 words per verdict (produced by processor.py).
    output_dir : str
        Folder where the PDF will be saved.

    Returns
    -------
    str
        Absolute path of the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "Misinformation_Report_LaSillaVacia.pdf")

    top_words = dict(word_freq.most_common(20))
    stop_words = get_stop_words()

    with PdfPages(pdf_path) as pdf:

        # ── PORTADA ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#2C3E50")
        fig.text(
            0.5, 0.60,
            "Misinformation Detector",
            ha="center", va="center", fontsize=32, fontweight="bold",
            color="white",
        )
        fig.text(
            0.5, 0.50,
            "La Silla Vacía — Misinformation Analysis",
            ha="center", va="center", fontsize=18, color="#ECF0F1",
        )
        # Analysis period
        start_date = pd.to_datetime(df["date"], errors="coerce").min()
        end_date = pd.to_datetime(df["date"], errors="coerce").max()
        fig.text(
            0.5, 0.40,
            f"Period: {start_date.strftime('%d/%m/%Y') if pd.notna(start_date) else 'N/A'}"
            f" — {end_date.strftime('%d/%m/%Y') if pd.notna(end_date) else 'N/A'}",
            ha="center", va="center", fontsize=13, color="#BDC3C7",
        )
        fig.text(
            0.5, 0.30,
            f"Total articles analyzed: {len(df)}",
            ha="center", va="center", fontsize=13, color="#BDC3C7",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── VIZ 1: Piruleta — Top 20 palabras más frecuentes ─────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        wl = list(top_words.keys())[::-1]
        cl = list(top_words.values())[::-1]
        ax.hlines(y=wl, xmin=0, xmax=cl, color="skyblue", linewidth=3)
        ax.plot(cl, wl, "o", markersize=10, color="steelblue", alpha=0.85)
        ax.set_title(
            "Top 20 Most Frequent Words in Misinformation Content",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Absolute Frequency")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.figtext(
            0.5, -0.02,
            "The most repeated words reveal the key actors and topics "
            "of misinformation during the analyzed period.",
            ha="center", fontsize=9, style="italic",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── VIZ 2: Bar chart — Verdict distribution ─────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        vc = df["verdict"].value_counts()
        bars = ax.bar(
            vc.index, vc.values,
            color=["#E74C3C", "#F39C12"],
            edgecolor="white", linewidth=1.5,
        )
        ax.bar_label(bars, padding=4, fontsize=13, fontweight="bold")
        ax.set_title("Verdict Distribution (FALSO vs. ENGAÑOSO)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Articles")
        ax.set_ylim(0, max(1, vc.max()) * 1.25)
        plt.tight_layout()
        plt.figtext(
            0.5, -0.02,
            "Shows whether outright fabrications (FALSO) "
            "or contextual manipulations (ENGAÑOSO) predominate.",
            ha="center", fontsize=9, style="italic",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── VIZ 3: Stacked bars — Articles by date ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 5))
        df_time = df.copy()
        df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
        df_time = df_time.dropna(subset=["date"])
        if not df_time.empty:
            abd = df_time.groupby(["date", "verdict"]).size().unstack(fill_value=0)
            abd.plot(kind="bar", ax=ax, color=["#E74C3C", "#F39C12"], stacked=True)
            ax.set_title("Articles Published by Date", fontsize=13, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Articles")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Verdict")
        plt.tight_layout()
        plt.figtext(
            0.5, -0.03,
            "Production spikes may correlate with key political events "
            "during the analyzed period.",
            ha="center", fontsize=9, style="italic",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── VIZ 4: Word cloud ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 6))
        raw_corpus = " ".join(df["full_text"].fillna(""))
        tokens_corpus = clean_text(raw_corpus, stop_words)
        all_clean = " ".join(tokens_corpus) or "no data"
        wc = WordCloud(
            width=1300, height=650, background_color="white",
            colormap="RdYlBu", max_words=80, regexp=r"[a-záéíóúñü]+",
        ).generate(all_clean)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud of the Full Corpus", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.figtext(
            0.5, -0.01,
            "Larger words appear more frequently. "
            "A quick view of the lexical density of the corpus.",
            ha="center", fontsize=9, style="italic",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── VIZ 5: Barras horizontales por veredicto ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        palette = {"FALSO": "#E74C3C", "ENGAÑOSO": "#F39C12"}
        for ax, vtype in zip(axes, ["FALSO", "ENGAÑOSO"]):
            top10 = top_by_verdict.get(vtype, [])
            wlabels = [w for w, _ in top10][::-1]
            wcounts = [c for _, c in top10][::-1]
            ax.barh(wlabels, wcounts, color=palette[vtype], alpha=0.85)
            ax.set_title(f"Top 10 Words — {vtype}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Frequency")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.suptitle(
            "Vocabulary Comparison: FALSO vs ENGAÑOSO",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.figtext(
            0.5, -0.03,
            "Reveals whether pure fabrications (FALSO) differ "
            "in vocabulary from contextual manipulations (ENGAÑOSO).",
            ha="center", fontsize=9, style="italic",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Metadatos del PDF
        d = pdf.infodict()
        d["Title"] = "Misinformation Report — La Silla Vacía"
        d["Author"] = "Misinformation Detector Pipeline"
        d["Subject"] = "Corpus frequency analysis and visualizations"

    print(f"✅ PDF report generated at: {pdf_path}")
    return pdf_path
