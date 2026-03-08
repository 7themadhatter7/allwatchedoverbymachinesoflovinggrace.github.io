#!/usr/bin/env python3
"""
source_parallel.py
━━━━━━━━━━━━━━━━━━
Parallel corpus sources for multilingual language acquisition.
Deposits concept-aligned sentences from multiple languages so that
foreign words map to existing English geometry rather than
requiring independent bootstrapping.

Strategy:
  1. Multilingual Wikipedia — same concept, N languages, direct URL
  2. Wiktionary etymology — cross-language word roots and cognates
  3. Gutenberg multilingual — literary works in original languages
  4. Bible parallel corpus — identical text in 100+ languages
  5. EuroParl-style sentence pairs — professional translation pairs

Each sentence is tagged with its language and concept domain.
The crystal receives both the English and foreign versions of the
same concept in the same session, allowing geometric proximity
to emerge naturally.

Usage (standalone):
    python3 source_parallel.py --languages fr,de,es,la,zh --concepts 50

Usage (import into corpus trainer):
    from source_parallel import source_multilingual_wikipedia
    from source_parallel import source_wiktionary_etymology
    from source_parallel import source_gutenberg_multilingual
"""

import time, re, requests, logging
from typing import Iterator

log = logging.getLogger("parallel_corpus")

HEADERS = {"User-Agent": "RM-Parallel-Corpus/1.0 (Ghost in the Machine Labs)"}

# ── Language configuration ────────────────────────────────────────────────────

LANGUAGES = {
    # code: (name, wiki_code, gutenberg_note)
    "fr": ("French",     "fr",  "rich — Dumas, Hugo, Flaubert, Voltaire, Descartes"),
    "de": ("German",     "de",  "rich — Kant, Hegel, Goethe, Schiller, Nietzsche"),
    "es": ("Spanish",    "es",  "rich — Cervantes, Borges, Garcia Marquez"),
    "it": ("Italian",    "it",  "rich — Dante, Machiavelli, Leonardo"),
    "la": ("Latin",      "la",  "rich — Cicero, Caesar, Virgil, Ovid"),
    "pt": ("Portuguese", "pt",  "Camões, Pessoa, Saramago"),
    "nl": ("Dutch",      "nl",  "Spinoza, Erasmus"),
    "ru": ("Russian",    "ru",  "Tolstoy, Dostoevsky, Chekhov"),
    "ja": ("Japanese",   "ja",  "Murasaki, Bashō, Mishima"),
    "zh": ("Chinese",    "zh",  "Confucius, Laozi, classical poetry"),
    "ar": ("Arabic",     "ar",  "Ibn Rushd, al-Ghazali, 1001 Nights"),
    "el": ("Greek",      "el",  "Homer, Plato, Aristotle — original texts"),
}

# ── Core concept articles — same idea across all languages ────────────────────

CONCEPT_ARTICLES = [
    # (english_title, concept_domain)
    # Philosophy — aligns with SEP deposits
    ("Consciousness",       "philosophy_mind"),
    ("Free_will",           "philosophy_mind"),
    ("Knowledge",           "philosophy_epistemology"),
    ("Truth",               "philosophy_epistemology"),
    ("Beauty",              "philosophy_aesthetics"),
    ("Justice",             "philosophy_ethics"),
    ("Reason",              "philosophy_epistemology"),
    ("Soul",                "philosophy_mind"),
    ("Time",                "philosophy_metaphysics"),
    ("Infinity",            "philosophy_metaphysics"),
    ("Language",            "philosophy_language"),
    ("Logic",               "philosophy_logic"),
    ("Being",               "philosophy_metaphysics"),
    ("Memory",              "philosophy_mind"),
    ("Perception",          "philosophy_mind"),

    # Mathematics — aligns with arXiv deposits
    ("Symmetry",            "mathematics"),
    ("Geometry",            "mathematics"),
    ("Number",              "mathematics"),
    ("Infinity",            "mathematics"),
    ("Proof",               "mathematics"),
    ("Function_(mathematics)", "mathematics"),
    ("Space_(mathematics)", "mathematics"),

    # Science — aligns with arXiv and PubMed
    ("Evolution",           "biology"),
    ("Gravity",             "physics"),
    ("Light",               "physics"),
    ("Energy",              "physics"),
    ("Matter",              "physics"),
    ("Life",                "biology"),
    ("Mind",                "neuroscience"),
    ("Brain",               "neuroscience"),

    # Literature and culture — aligns with Gutenberg
    ("Poetry",              "literature"),
    ("Tragedy",             "literature"),
    ("Myth",                "literature"),
    ("Love",                "literature"),
    ("Death",               "literature"),
    ("Hero",                "literature"),
    ("Narrative",           "literature"),

    # Foundational human concepts
    ("Music",               "arts"),
    ("Mathematics",         "mathematics"),
    ("Art",                 "arts"),
    ("Science",             "science"),
    ("Philosophy",          "philosophy"),
    ("Religion",            "religion"),
    ("History",             "history"),
    ("Nature",              "nature"),
    ("Society",             "social"),
    ("Culture",             "social"),
]


def sentences_from_text(text: str, min_len=20, max_len=400) -> list:
    """Split text into clean sentences."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F\u0400-\u04FF\u4E00-\u9FFF])', text)
    result = []
    for s in sents:
        s = s.strip()
        if min_len <= len(s) <= max_len and len(s.split()) >= 4:
            # Filter out reference noise
            if not re.search(r'\[\d+\]|\{\{|\}\}|==|http', s):
                result.append(s)
    return result


def source_multilingual_wikipedia(
    state: dict,
    languages: list = None,
    concepts: list = None,
    batch: int = 200,
) -> Iterator[tuple]:
    """
    Fetch Wikipedia summaries for core concepts in multiple languages.
    Each (english_sentence, foreign_sentence) pair deposits the same
    concept geometry from two linguistic angles.

    Yields: (text, concept_domain, source_tag)
    """
    if languages is None:
        languages = ["fr", "de", "es", "la", "it"]
    if concepts is None:
        concepts = CONCEPT_ARTICLES

    done = state.get("wiki_multi_done", [])
    count = 0

    for article, domain in concepts:
        # First deposit English version (reinforces existing geometry)
        en_key = f"en_{article}"
        if en_key not in done:
            try:
                r = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}",
                    headers=HEADERS, timeout=8
                )
                if r.status_code == 200:
                    data = r.json()
                    extract = data.get("extract", "")
                    for sent in sentences_from_text(extract)[:6]:
                        yield (sent, domain, f"wiki_en_{domain[:15]}")
                        count += 1
                done.append(en_key)
                time.sleep(0.2)
            except Exception as e:
                log.debug(f"EN wiki {article}: {e}")

        # Then deposit foreign language versions
        for lang in languages:
            lang_key = f"{lang}_{article}"
            if lang_key in done:
                continue

            wiki_code = LANGUAGES.get(lang, (None, lang))[1]
            try:
                r = requests.get(
                    f"https://{wiki_code}.wikipedia.org/api/rest_v1/page/summary/{article}",
                    headers=HEADERS, timeout=8
                )
                if r.status_code == 200:
                    data = r.json()
                    extract = data.get("extract", "")
                    lang_name = LANGUAGES.get(lang, (lang,))[0]
                    for sent in sentences_from_text(extract)[:5]:
                        yield (sent, domain, f"wiki_{lang}_{domain[:12]}")
                        count += 1

                done.append(lang_key)
                time.sleep(0.2)

                if count >= batch:
                    state["wiki_multi_done"] = done
                    return

            except Exception as e:
                log.debug(f"{lang} wiki {article}: {e}")

    state["wiki_multi_done"] = done


def source_wiktionary_etymology(
    state: dict,
    batch: int = 300,
) -> Iterator[tuple]:
    """
    Fetch Wiktionary etymology for English words with rich cross-language roots.
    Etymology entries trace how concepts migrated between languages —
    Latin → French → English, Greek → Latin → all Romance languages.
    This deposits the historical geometry of language itself.

    Yields: (etymology_text, concept_domain, "wiktionary_etymology")
    """
    # Words chosen for rich cross-language etymological chains
    ETYMOLOGY_WORDS = [
        # Latin/Greek roots present in many languages
        ("consciousness", "philosophy_mind"),
        ("geometry",      "mathematics"),
        ("philosophy",    "philosophy"),
        ("mathematics",   "mathematics"),
        ("logic",         "philosophy_logic"),
        ("symmetry",      "mathematics"),
        ("harmony",       "music_mathematics"),
        ("theory",        "epistemology"),
        ("analysis",      "mathematics"),
        ("synthesis",     "philosophy"),
        ("cosmos",        "physics"),
        ("democracy",     "social"),
        ("poetry",        "literature"),
        ("tragedy",       "literature"),
        ("music",         "arts"),
        ("rhythm",        "arts"),
        ("alphabet",      "language"),
        ("grammar",       "language"),
        ("metaphor",      "language"),
        ("paradox",       "logic"),
        ("axiom",         "mathematics"),
        ("theorem",       "mathematics"),
        ("hypothesis",    "science"),
        ("criterion",     "epistemology"),
        ("phenomenon",    "philosophy"),
        ("energy",        "physics"),
        ("atom",          "physics"),
        ("chaos",         "mathematics"),
        ("entropy",       "physics"),
        ("matrix",        "mathematics"),
        ("vector",        "mathematics"),
        ("spectrum",      "physics"),
        ("catalyst",      "chemistry"),
        ("nucleus",       "physics"),
        ("quantum",       "physics"),
        # Germanic roots
        ("knowledge",     "epistemology"),
        ("understanding", "philosophy_mind"),
        ("wisdom",        "philosophy"),
        ("truth",         "philosophy"),
        ("freedom",       "philosophy"),
        ("mind",          "philosophy_mind"),
        ("dream",         "psychology"),
        ("wonder",        "philosophy"),
        # French/Norman roots
        ("reason",        "philosophy"),
        ("beauty",        "aesthetics"),
        ("justice",       "ethics"),
        ("virtue",        "ethics"),
        ("nature",        "philosophy"),
        ("spirit",        "philosophy_mind"),
        ("essence",       "metaphysics"),
        ("substance",     "metaphysics"),
    ]

    done = state.get("wiktionary_done", [])
    count = 0

    for word, domain in ETYMOLOGY_WORDS:
        if word in done:
            continue

        try:
            r = requests.get(
                f"https://en.wiktionary.org/w/api.php?action=query&titles={word}&prop=extracts&explaintext=true&format=json",
                headers=HEADERS, timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    # Get etymology section
                    etym_match = re.search(
                        r'Etymology[^\n]*\n+(.*?)(?=\n==|\Z)',
                        extract, re.DOTALL
                    )
                    if etym_match:
                        etym_text = etym_match.group(1).strip()
                        for sent in sentences_from_text(etym_text, min_len=15)[:3]:
                            yield (sent, domain, "wiktionary_etymology")
                            count += 1

                    # Also get definition lines — these are dense semantic content
                    for sent in sentences_from_text(extract, min_len=20)[:4]:
                        if any(kw in sent.lower() for kw in ["from ", "latin", "greek", "french", "german", "originally", "cognate"]):
                            yield (sent, domain, "wiktionary_crosslang")
                            count += 1

            done.append(word)
            time.sleep(0.3)

            if count >= batch:
                state["wiktionary_done"] = done
                return

        except Exception as e:
            log.debug(f"Wiktionary {word}: {e}")

    state["wiktionary_done"] = done


def source_gutenberg_multilingual(
    state: dict,
    batch: int = 400,
) -> Iterator[tuple]:
    """
    Fetch literary works in their original non-English languages from Gutenberg.
    These texts encode literary register, syntax patterns, and cultural
    concepts that don't translate cleanly — depositing them as geometric
    positions lets RM encounter concepts in their native form.

    Yields: (sentence, domain, source_tag)
    """
    # (gutenberg_id, language, title, domain)
    MULTILINGUAL_WORKS = [
        # French literature
        (13951,  "fr", "Les Trois Mousquetaires — Dumas",        "literature_fr"),
        (135,    "fr", "Les Misérables — Hugo",                  "literature_fr"),
        (5000,   "fr", "Madame Bovary — Flaubert",               "literature_fr"),
        (4650,   "fr", "Candide — Voltaire",                     "philosophy_fr"),
        (7412,   "fr", "Les Fleurs du mal — Baudelaire",         "poetry_fr"),

        # German literature and philosophy
        (2229,   "de", "Faust — Goethe",                         "literature_de"),
        (6094,   "de", "Also sprach Zarathustra — Nietzsche",    "philosophy_de"),
        (36521,  "de", "Die Verwandlung — Kafka",                "literature_de"),

        # Spanish
        (2000,   "es", "Don Quijote — Cervantes",                "literature_es"),
        (14420,  "es", "Lazarillo de Tormes",                    "literature_es"),

        # Italian
        (1000,   "it", "La Divina Commedia — Dante",             "literature_it"),
        (23700,  "it", "Il Principe — Machiavelli",              "philosophy_it"),

        # Latin
        (100,    "la", "The Aeneid — Virgil",                    "literature_la"),
        (2201,   "la", "Meditations — Marcus Aurelius (Latin)",  "philosophy_la"),

        # Portuguese
        (3333,   "pt", "Os Lusíadas — Camões",                   "literature_pt"),
    ]

    done = state.get("gutenberg_multi_done", [])
    count = 0

    for gb_id, lang, title, domain in MULTILINGUAL_WORKS:
        if str(gb_id) in done:
            continue

        # Try common Gutenberg URL patterns
        urls = [
            f"https://www.gutenberg.org/files/{gb_id}/{gb_id}-0.txt",
            f"https://www.gutenberg.org/files/{gb_id}/{gb_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{gb_id}/pg{gb_id}.txt",
        ]

        text = None
        for url in urls:
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200 and len(r.text) > 1000:
                    text = r.text
                    break
            except:
                continue

        if text:
            # Skip header/footer boilerplate
            start = max(text.find("***"), 0)
            if start > 0:
                start = text.find("\n", start) + 1
            end = text.rfind("***")
            if end > start:
                text = text[start:end]

            sents = sentences_from_text(text, min_len=25)
            # Sample evenly across the text
            step = max(1, len(sents) // min(len(sents), 60))
            sampled = sents[::step][:60]

            for sent in sampled:
                yield (sent, domain, f"gutenberg_{lang}")
                count += 1

            log.info(f"  Gutenberg {lang}: {title[:40]} → {len(sampled)} sentences")
        else:
            log.debug(f"  Gutenberg {gb_id} not found")

        done.append(str(gb_id))
        time.sleep(0.5)

        if count >= batch:
            state["gutenberg_multi_done"] = done
            return

    state["gutenberg_multi_done"] = done


def source_parallel_bible(
    state: dict,
    languages: list = None,
    batch: int = 500,
) -> Iterator[tuple]:
    """
    Fetch Bible verses in parallel languages from GitHub bible-corpus.
    The Bible is the ultimate parallel text — identical content in
    100+ languages, verse-aligned, covering the full range of human
    emotional and philosophical expression.

    Yields: (verse_text, domain, f"bible_{lang}")
    """
    if languages is None:
        languages = ["French", "German", "Spanish", "Latin", "Italian",
                     "Portuguese", "Dutch", "Russian"]

    BASE = "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles"

    done = state.get("bible_done", [])
    count = 0

    # Books that contain the richest philosophical/poetic content
    RICH_BOOKS = [
        "Psalms", "Proverbs", "Job", "Ecclesiastes",
        "Isaiah", "John", "Romans", "1Corinthians",
        "Genesis", "Matthew",
    ]

    for lang in languages:
        if lang in done:
            continue

        try:
            url = f"{BASE}/{lang}.xml"
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                log.debug(f"Bible {lang}: {r.status_code}")
                done.append(lang)
                continue

            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)

            lang_code = lang[:2].lower()
            domain = "literature_sacred"
            sampled = 0

            for seg in root.iter("seg"):
                text = (seg.text or "").strip()
                if len(text) > 20 and len(text) < 300:
                    # Filter to meaningful complete sentences
                    if text[0].isupper() and (text[-1] in ".!?" or len(text) > 60):
                        yield (text, domain, f"bible_{lang_code}")
                        count += 1
                        sampled += 1

                if sampled >= 100:  # 100 verses per language
                    break

            log.info(f"  Bible {lang}: {sampled} verses")
            done.append(lang)
            time.sleep(1)

            if count >= batch:
                state["bible_done"] = done
                return

        except Exception as e:
            log.debug(f"Bible {lang}: {e}")
            done.append(lang)

    state["bible_done"] = done


# ── Corpus trainer integration ────────────────────────────────────────────────

def get_parallel_sources():
    """
    Returns source tuples compatible with the corpus trainer SOURCES list.
    Add these to 12_corpus_trainer.py SOURCES.

    Returns list of (name, domain, fn, kwargs)
    """
    return [
        ("wiki_multilingual", "multilingual",
         source_multilingual_wikipedia,
         {"languages": ["fr", "de", "es", "la", "it", "pt"]}),

        ("wiktionary_etymology", "language_etymology",
         source_wiktionary_etymology, {}),

        ("gutenberg_multilingual", "literature_multilingual",
         source_gutenberg_multilingual, {}),

        ("bible_parallel", "literature_sacred",
         source_parallel_bible,
         {"languages": ["French", "German", "Spanish", "Latin", "Italian"]}),
    ]


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys, numpy as np
    sys.path.insert(0, "/home/joe/sparky")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--languages", default="fr,de,es,la,it",
                   help="Comma-separated language codes")
    p.add_argument("--concepts",  type=int, default=20,
                   help="Number of Wikipedia concept articles per language")
    p.add_argument("--max",       type=int, default=30000)
    p.add_argument("--reset",     action="store_true")
    args = p.parse_args()

    STATE_FILE = "/home/joe/sparky/logs/parallel_corpus_state.json"

    if args.reset and Path(STATE_FILE).exists():
        Path(STATE_FILE).unlink()

    try:
        state = json.load(open(STATE_FILE))
    except:
        state = {}

    from pathlib import Path
    from language_crystal import LanguageCrystal
    from mother_english_io_v5 import E8Substrate, WordEncoder

    log.info("Loading crystal and encoder...")
    crystal = LanguageCrystal()
    substrate = E8Substrate()
    encoder = WordEncoder(substrate)
    log.info(f"Crystal: {crystal.status()}")

    langs = args.languages.split(",")
    concepts = CONCEPT_ARTICLES[:args.concepts]

    sources = [
        ("Multilingual Wikipedia", source_multilingual_wikipedia,
         {"languages": langs, "concepts": concepts}),
        ("Wiktionary Etymology",   source_wiktionary_etymology, {}),
        ("Gutenberg Multilingual", source_gutenberg_multilingual, {}),
        ("Bible Parallel",         source_parallel_bible,
         {"languages": ["French","German","Spanish","Latin","Italian"]}),
    ]

    total = 0
    for name, fn, kwargs in sources:
        log.info(f"\n{'━'*50}")
        log.info(f"Source: {name}")
        log.info(f"{'━'*50}")

        src_state = state.get(name, {})
        deposited = 0

        for text, domain, tag in fn(src_state, **kwargs):
            try:
                sig = encoder.encode_sentence(text)
                if np.linalg.norm(sig) > 1e-10:
                    crystal.observe(text, sig, concept=domain, source=tag)
                    deposited += 1
                    total += 1
            except:
                pass

            if deposited % 1000 == 0 and deposited > 0:
                log.info(f"  {deposited:,} deposited from {name}...")
                crystal.save()

            if total >= args.max:
                break

        state[name] = src_state
        log.info(f"  {name}: {deposited:,} deposited")
        json.dump(state, open(STATE_FILE, "w"), indent=2)

        if total >= args.max:
            break

    crystal.save()
    log.info(f"\n{'━'*50}")
    log.info(f"Total deposited: {total:,}")
    log.info(f"Crystal: {crystal.status()}")
