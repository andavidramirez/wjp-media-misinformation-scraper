"""
processor.py — Phase 2 & 3a: Text Cleaning and Frequency Analysis
========================================================================
Responsibilities:
  - Automatically download NLTK resources if not already present.
  - Build an enriched Spanish stopword set, extended with generic terms
    from the journalism/politics domain that carry no analytical value.
  - Clean each article: lowercase, remove URLs, punctuation, numbers,
    and stopwords; retain only tokens longer than 2 characters.
  - Compute absolute word frequencies over the full corpus.
  - Compute the top-10 terms separately for each verdict (FALSO / ENGÁÑOSO).
"""

import re
from collections import Counter

import nltk
import pandas as pd


# Automatically download required NLTK resources (no user interaction needed)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords


def _build_stopwords() -> set[str]:
    """Builds the enriched Spanish stopword set, extended with generic terms
    from the journalism domain and La Silla Vacía specifically that
    carry no semantic value for the analysis.
    """
    sw = set(nltk_stopwords.words("spanish"))
    sw.update([
        # Articles (determiners)
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            # Prepositions
            "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde",
            "durante", "en", "entre", "hacia", "hasta", "mediante", "para",
            "por", "según", "sin", "sobre", "tras",
            # Conjunctions
            "e", "ni", "o", "u", "y", "pero", "sino", "aunque", "porque",
            "pues", "que", "si", "como", "cuando", "donde", "mientras",
            # Pronouns
            "él", "ella", "ellos", "ellas", "ello", "yo", "tú", "usted",
            "nosotros", "nosotras", "vosotros", "vosotras", "ustedes",
            "me", "te", "se", "nos", "os", "le", "les", "lo", "las",
            "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
            "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
            "este", "esta", "esto", "estos", "estas",
            "ese", "esa", "eso", "esos", "esas",
            "aquel", "aquella", "aquello", "aquellos", "aquellas",
            "quien", "quienes", "cual", "cuales", "cuyo", "cuya",
            # Common auxiliary and copulative verbs
            "ser", "estar", "haber", "tener", "hacer", "poder", "deber",
            "es", "son", "era", "eran", "fue", "fueron", "será", "serán",
            "está", "están", "estuvo", "estuvieron", "estará", "estarán",
            "ha", "han", "había", "habían", "habrá", "habrán", "hubo", "hubieron",
            "tiene", "tienen", "tuvo", "tuvieron", "tendrá", "tendrán",
            "hace", "hacen", "hizo", "hicieron", "hará", "harán",
            "puede", "pueden", "pudo", "pudieron", "podrá", "podrán",
            "hay", "hubo", "habrá",
            # Common adverbs
            "no", "sí", "más", "menos", "muy", "bien", "mal", "ya", "aún",
            "también", "tampoco", "así", "aquí", "allí", "ahí", "ahora",
            "antes", "después", "entonces", "siempre", "nunca", "jamás",
            "además", "sin", "embargo", "solo", "sólo", "incluso", "además",
            "quizás", "quizá", "tal", "vez", "acaso", "casi", "bastante",
            "demasiado", "tanto", "tan", "cuanto", "cómo", "cuándo", "dónde",
            # Demonstratives and quantifiers
            "todo", "toda", "todos", "todas", "cada", "otro", "otra", "otros",
            "otras", "mismo", "misma", "mismos", "mismas", "algún", "alguna",
            "algunos", "algunas", "ningún", "ninguna", "ningunos", "ningunas",
            "mucho", "mucha", "muchos", "muchas", "poco", "poca", "pocos", "pocas",
            # Interrogatives / relatives
            "qué", "quién", "cuál", "cuánto", "cuánta", "cómo", "cuándo", "dónde",
            # Generic time terms
            "año", "años", "mes", "meses", "día", "días", "vez", "veces",
            "semana", "semanas", "hora", "horas", "momento", "tiempo",
            # Communication verbs (very frequent in news; low analytical value)
            "dijo", "dice", "dicen", "dijeron", "señaló", "señalaron", "indicó",
            "indicaron", "afirmó", "afirmaron", "agregó", "agregaron", "explicó",
            "explicaron", "aseguró", "aseguraron", "manifestó", "manifestaron",
            "comentó", "comentaron", "anunció", "anunciaron", "publicó", "publicaron",
            "encontró", "encontraron", "reveló", "revelaron", "confirmó", "confirmaron",
            # Generic journalistic article terms
            "según", "través", "parte", "caso", "tipo", "forma", "manera",
            "hecho", "hechos", "dicho", "dicha", "dichos", "dichas",
            "embargo", "sino", "donde", "cuando", "aunque", "porque",
            "imagen", "video", "foto", "gráfico", "infografía", "cita", "entrevista",
            "medios", "prensa", "periodistas", "reporteros", "fuentes",
            # Domain-specific terms (La Silla Vacía / Colombia)
            "colombia", "colombiano", "colombiana", "colombianos", "colombianas",
            "silla", "vacía", "lasillavacia", "publicación", "artículo",
            "detector", "mentiras", "verificación", "verificado",
            "falso", "engañoso", "engañosos", "desinformación", "desinformar",
            # Other common terms that carry no analytical value
            "marzo","sido","circula","supuesta","frase","frase","frases",
            "publicó","publican","publican","publicó","dos","tres","cuatro","cinco",
            "seis","siete","ocho","nueve","diez",
    ])
    return sw


# Singleton to avoid rebuilding the set on every call
_STOP_WORDS: set[str] | None = None


def get_stop_words() -> set[str]:
    """Returns the stopword set (built only once)."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = _build_stopwords()
    return _STOP_WORDS


def clean_text(text: str, stop_words: set[str]) -> list[str]:
    """Cleans a block of text and returns the list of valid tokens.

    Steps:
      1. Lowercase the text.
      2. Remove URLs (http/www).
      3. Keep only Spanish letters (a-z, accented vowels, ñ).
      4. Tokenize by whitespace.
      5. Remove tokens of 2 or fewer characters and stopwords.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-záéíóúñü]", " ", text)
    tokens = [w for w in text.split() if len(w) > 2 and w not in stop_words]
    return tokens


def compute_word_frequencies(df: pd.DataFrame) -> Counter:
    """Computes word frequencies across all articles in the corpus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'full_text' column.

    Returns
    -------
    Counter
        Frequencies sorted from most to least common.
    """
    sw = get_stop_words()
    all_tokens: list[str] = []
    for text in df["full_text"].fillna(""):
        all_tokens.extend(clean_text(text, sw))
    return Counter(all_tokens)


def compute_top_by_verdict(df: pd.DataFrame, n: int = 10) -> dict[str, list[tuple[str, int]]]:
    """Computes the top-N terms separately for FALSO and ENGÁÑOSO verdicts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'verdict' and 'full_text' columns.
    n : int
        Number of terms to return per verdict.

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        {'FALSO': [(word, frequency), ...], 'ENGÁÑOSO': [...]}
    """
    sw = get_stop_words()
    result: dict[str, list[tuple[str, int]]] = {}
    for verdict in ("FALSO", "ENGAÑOSO"):
        subset_texts = df[df["verdict"] == verdict]["full_text"].fillna("").str.cat(sep=" ")
        tokens = clean_text(subset_texts, sw)
        result[verdict] = Counter(tokens).most_common(n)
    return result
