"""
12_corpus_trainer.py — RM Continuous Knowledge Acquisition
Ghost in the Machine Labs

Downloads and deposits scientific/educational corpora into the
Language Crystal. Runs whenever SPARKY has been idle 15 minutes.
Incremental — only fetches what's new since last run.

Sources (all open access):
    Wikipedia   — English abstracts, monthly dump
    arXiv       — math, physics, cs, engineering, bio, chem (daily)
    PubMed      — biomedical abstracts (daily)
    Gutenberg   — full literary catalog (70K texts)
    OpenStax    — undergraduate textbooks (all subjects)
    OEIS        — integer sequences with descriptions
    ProofWiki   — mathematical proofs
    NASA TRS    — technical reports
    NIST        — reference data
    SEP         — Stanford Encyclopedia of Philosophy
    RFC         — internet protocol specifications

Interval: 900 seconds (15 minutes idle trigger)
Each run fetches one batch per source, deposits, saves cursor.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations, product
from typing import List, Iterator

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/joe/sparky")

from _lib import MaintenanceScript, log, load_state, save_state, SPARKY

# Parallel multilingual corpus sources
import sys as _sys
_sys.path.insert(0, "/home/joe/sparky")
try:
    from source_programming import (
        source_programming_seeds,
        source_programming_arxiv,
        source_programming_wikipedia,
        source_rosetta_code,
    )
    _PROGRAMMING_AVAILABLE = True
except ImportError as _e:
    log.warning(f"Programming sources unavailable: {_e}")
    _PROGRAMMING_AVAILABLE = False

try:
    from source_parallel import (
        source_multilingual_wikipedia,
        source_wiktionary_etymology,
        source_gutenberg_multilingual,
        source_parallel_bible,
    )
    _PARALLEL_AVAILABLE = True
except ImportError as _e:
    log.warning(f"Parallel sources unavailable: {_e}")
    _PARALLEL_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────
OBSERVE_N       = 7       # Observations per sentence
BATCH_SIZE      = 200     # Sentences per source per run
ENCODE_WORKERS  = max(1, __import__('os').cpu_count() - 2)  # cores for parallel encode
FETCH_WORKERS   = 13      # one thread per source (I/O bound)
MIN_WORDS       = 6
MAX_WORDS       = 80
TIMEOUT         = 20      # HTTP timeout seconds
BANDWIDTH_CAP   = 9_800_000  # 50% of WAN = ~9.8 MB/s
REST_SECONDS    = 5          # pause between continuous passes
REGISTRY_PATH   = SPARKY / "corpus_registry.json"
CRYSTAL_LOG     = SPARKY / "logs" / "corpus_trainer.log"

import numpy as np
import hashlib


# ═══════════════════════════════════════════════════════════════════
# INLINE E8 ENCODER  (no external deps beyond numpy)
# ═══════════════════════════════════════════════════════════════════

class _E8:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @staticmethod
    def _build():
        verts = []
        for pos in combinations(range(8), 2):
            for signs in product([-1,1], repeat=2):
                v = [0.0]*8
                v[pos[0]], v[pos[1]] = float(signs[0]), float(signs[1])
                verts.append(v)
        for signs in product([-0.5,0.5], repeat=8):
            if signs.count(-0.5) % 2 == 0:
                verts.append(list(signs))
        verts = np.array(verts, dtype=np.float32)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        adj = np.zeros((240,240), dtype=np.float32)
        for i in range(240):
            d = np.linalg.norm(verts - verts[i], axis=1)
            mask = (d > 0.01) & (d < d[d>0.01].min() + 0.01)
            adj[i,mask] = 1.0
        L = np.diag(adj.sum(1)) - adj
        _, em = np.linalg.eigh(L)
        return em.astype(np.float32)

def encode_text(text: str, eigenmodes=None) -> np.ndarray:
    if eigenmodes is None:
        eigenmodes = _E8.get()
    words = text.lower().split()
    combined = np.zeros(240, dtype=np.float32)
    for i, w in enumerate(words):
        c = ''.join(x for x in w if x.isalnum())
        if not c: continue
        h = int.from_bytes(hashlib.sha256(c.encode()).digest()[:8], 'big')
        inj = np.zeros(240, dtype=np.float32)
        for j in range(4):
            inj[(h>>(j*8))%240] += ((h>>(32+j*4))%16+1)/16.0
        m = eigenmodes.T @ inj
        n = np.linalg.norm(m)
        if n > 0: m = m/n
        combined += m / (1.0 + 0.1*i)
    n = np.linalg.norm(combined)
    return combined/n if n > 0 else combined


def is_good_sentence(text: str) -> bool:
    words = text.split()
    return (MIN_WORDS <= len(words) <= MAX_WORDS
            and any(c.isalpha() for c in text))


def fetch_url(url: str, timeout: int = TIMEOUT) -> str:
    """Fetch URL, return text or empty string."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'GhostInTheMachineLabs/1.0 RM-CorpusTrainer'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            chunks = []
            chunk_size = 65536
            while True:
                t0 = time.time()
                chunk = r.read(chunk_size)
                if not chunk: break
                chunks.append(chunk)
                elapsed = time.time() - t0
                expected = len(chunk) / BANDWIDTH_CAP
                if expected > elapsed:
                    time.sleep(expected - elapsed)
            raw = b''.join(chunks)
            try: return raw.decode('utf-8')
            except: return raw.decode('latin-1', errors='ignore')
    except Exception as e:
        return ""


# ═══════════════════════════════════════════════════════════════════
# SOURCE ADAPTERS
# ═══════════════════════════════════════════════════════════════════

def sentences_from_text(text: str) -> List[str]:
    """Split raw text into clean sentences."""
    sents = re.split(r'[.!?]+', text)
    out = []
    for s in sents:
        s = re.sub(r'\s+', ' ', s).strip()
        if is_good_sentence(s):
            out.append(s)
    return out


def source_arxiv(state: dict, domain: str, category: str,
                 batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """
    Fetch arXiv abstracts for a category.
    state key: f"arxiv_{category}_start"
    Yields (text, concept, source_tag)
    """
    start = state.get(f"arxiv_{category}_start", 0)
    url = (f"http://export.arxiv.org/api/query?"
           f"search_query=cat:{category}&start={start}"
           f"&max_results=50&sortBy=submittedDate&sortOrder=descending")
    xml = fetch_url(url, timeout=30)
    if not xml: return
    try:
        root = ET.fromstring(xml)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        count = 0
        for entry in entries:
            title_el   = entry.find('atom:title', ns)
            summary_el = entry.find('atom:summary', ns)
            if title_el is None or summary_el is None: continue
            title   = re.sub(r'\s+', ' ', title_el.text or '').strip()
            summary = re.sub(r'\s+', ' ', summary_el.text or '').strip()
            concept = domain
            if title and is_good_sentence(title):
                yield (title, concept, f"arxiv_{category}")
                count += 1
            for sent in sentences_from_text(summary)[:6]:
                yield (sent, concept, f"arxiv_{category}")
                count += 1
            if count >= batch: break
        state[f"arxiv_{category}_start"] = start + len(entries)
    except Exception:
        pass


def source_pubmed(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch PubMed abstracts via E-utils."""
    # Search for recent articles
    offset = state.get("pubmed_offset", 0)
    search_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                  f"db=pubmed&term=science[title]&retstart={offset}"
                  f"&retmax=20&usehistory=y&retmode=json")
    data = fetch_url(search_url, timeout=30)
    if not data: return
    try:
        j = json.loads(data)
        ids = j.get("esearchresult", {}).get("idlist", [])
        if not ids: return
        fetch_url2 = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                      f"db=pubmed&id={','.join(ids)}&retmode=xml&rettype=abstract")
        xml = fetch_url(fetch_url2, timeout=30)
        if not xml: return
        root = ET.fromstring(xml)
        count = 0
        for article in root.findall('.//PubmedArticle'):
            title_el = article.find('.//ArticleTitle')
            abs_el   = article.find('.//AbstractText')
            title   = (title_el.text or '') if title_el is not None else ''
            abstract = (abs_el.text or '')  if abs_el  is not None else ''
            if title and is_good_sentence(title):
                yield (title, "biomedical", "pubmed")
                count += 1
            for sent in sentences_from_text(abstract)[:5]:
                yield (sent, "biomedical", "pubmed")
                count += 1
            if count >= batch: break
        state["pubmed_offset"] = offset + len(ids)
    except Exception:
        pass


def source_wikipedia(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch Wikipedia article summaries via REST API."""
    # Use random featured articles + category walk
    fetched = 0
    for _ in range(batch // 5):
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        data = fetch_url(url, timeout=15)
        if not data: continue
        try:
            j = json.loads(data)
            title   = j.get("title", "")
            extract = j.get("extract", "")
            concept = j.get("description", "wikipedia")[:50]
            if title and is_good_sentence(title):
                yield (title, concept, "wikipedia")
                fetched += 1
            for sent in sentences_from_text(extract)[:8]:
                yield (sent, concept, "wikipedia")
                fetched += 1
            if fetched >= batch: break
        except Exception:
            continue
    state["wikipedia_fetched"] = state.get("wikipedia_fetched", 0) + fetched


def source_gutenberg(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch Project Gutenberg texts via NLTK (already downloaded)."""
    try:
        from nltk.corpus import gutenberg
        fileids = gutenberg.fileids()
        idx = state.get("gutenberg_idx", 0) % len(fileids)
        fileid = fileids[idx]
        concept = fileid.replace(".txt","").replace("-","_")
        raw = gutenberg.raw(fileid)
        sents = [s.strip() for s in re.split(r'[.!?]+', raw)
                 if is_good_sentence(s.strip())]
        offset = state.get(f"gutenberg_{idx}_offset", 0)
        count = 0
        for sent in sents[offset:offset+batch]:
            yield (sent, concept, "gutenberg")
            count += 1
        state[f"gutenberg_{idx}_offset"] = offset + count
        if offset + count >= len(sents):
            state["gutenberg_idx"] = idx + 1
    except Exception:
        pass


def source_oeis(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch OEIS sequence descriptions."""
    start_id = state.get("oeis_id", 1)
    count = 0
    for seq_id in range(start_id, start_id + batch):
        url = f"https://oeis.org/A{seq_id:06d}/internal"
        data = fetch_url(url, timeout=10)
        if not data:
            state["oeis_id"] = seq_id + 1
            continue
        # Extract name line
        for line in data.split('\n'):
            if line.startswith('%N'):
                desc = line[3:].strip()
                if is_good_sentence(desc):
                    yield (f"A{seq_id:06d}: {desc}", "mathematics", "oeis")
                    count += 1
                break
        state["oeis_id"] = seq_id + 1
        if count >= batch: break


def source_sep(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch Stanford Encyclopedia of Philosophy entries."""
    # SEP table of contents
    toc_url = "https://plato.stanford.edu/contents.html"
    if "sep_entries" not in state:
        html = fetch_url(toc_url, timeout=20)
        entries = re.findall(r'href="entries/([^"]+)"', html)
        state["sep_entries"] = list(set(entries))
        state["sep_idx"] = 0

    entries = state.get("sep_entries", [])
    idx = state.get("sep_idx", 0)
    count = 0

    for entry in entries[idx:idx+10]:
        url = f"https://plato.stanford.edu/entries/{entry}/"
        html = fetch_url(url, timeout=20)
        if not html: continue
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        for sent in sentences_from_text(text)[:20]:
            yield (sent, "philosophy", "sep")
            count += 1
        if count >= batch: break

    state["sep_idx"] = idx + 10


def source_rfc(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch IETF RFC documents."""
    rfc_num = state.get("rfc_num", 1000)
    count = 0
    attempts = 0
    while count < batch and attempts < 50:
        url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"
        text = fetch_url(url, timeout=15)
        if text:
            for sent in sentences_from_text(text)[:20]:
                yield (sent, "networking_protocols", "rfc")
                count += 1
        rfc_num += 1
        attempts += 1
    state["rfc_num"] = rfc_num


def source_nist(state: dict, batch: int = BATCH_SIZE) -> Iterator[tuple]:
    """Fetch NIST DLMF mathematical function descriptions."""
    chapters = state.get("nist_chapters_done", [])
    # DLMF chapter URLs
    dlmf_chapters = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"
    ]
    count = 0
    for ch in dlmf_chapters:
        if ch in chapters: continue
        url = f"https://dlmf.nist.gov/{ch}"
        html = fetch_url(url, timeout=20)
        if not html: continue
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        for sent in sentences_from_text(text)[:30]:
            yield (sent, "mathematics", "nist_dlmf")
            count += 1
        chapters.append(ch)
        if count >= batch: break
    state["nist_chapters_done"] = chapters


# ═══════════════════════════════════════════════════════════════════
# SOURCE REGISTRY
# ═══════════════════════════════════════════════════════════════════

SOURCES = [
    # (name, domain, generator_fn, kwargs)
    ("wikipedia",       "general",              source_wikipedia,   {}),
    ("arxiv_math",      "mathematics",          source_arxiv,
     {"domain": "mathematics",   "category": "math"}),
    ("arxiv_physics",   "physics",              source_arxiv,
     {"domain": "physics",       "category": "physics"}),
    ("arxiv_cs",        "computer_science",     source_arxiv,
     {"domain": "computer_science", "category": "cs"}),
    ("arxiv_eng",       "engineering",          source_arxiv,
     {"domain": "engineering",   "category": "eess"}),
    ("arxiv_bio",       "biology",              source_arxiv,
     {"domain": "biology",       "category": "q-bio"}),
    ("arxiv_chem",      "chemistry",            source_arxiv,
     {"domain": "chemistry",     "category": "physics.chem-ph"}),
    ("pubmed",          "biomedical",           source_pubmed,      {}),
    ("gutenberg",       "literature",           source_gutenberg,   {}),
    ("oeis",            "mathematics",          source_oeis,        {}),
    ("sep",             "philosophy",           source_sep,         {}),
    ("rfc",             "networking",           source_rfc,         {}),
    ("nist",            "mathematics",          source_nist,        {}),
    # ── Parallel multilingual sources ──────────────────────────────
    *([ 
        ("wiki_multilingual",    "multilingual",      source_multilingual_wikipedia,
         {"languages": ["fr", "de", "es", "la", "it", "pt", "ru"]}),
        ("wiktionary_etymology", "language_etymology", source_wiktionary_etymology, {}),
        ("gutenberg_multilingual","literature_multilingual", source_gutenberg_multilingual, {}),
        ("bible_parallel",       "literature_sacred",  source_parallel_bible,
         {"languages": ["French","German","Spanish","Latin","Italian"]}),
    ] if _PARALLEL_AVAILABLE else []),
    # ── Programming language sources ────────────────────────────────
    *([
        ("prog_seeds",     "computer_science", source_programming_seeds,    {}),
        ("prog_arxiv",     "computer_science", source_programming_arxiv,    {}),
        ("prog_wikipedia", "computer_science", source_programming_wikipedia,{}),
        ("prog_rosetta",   "computer_science", source_rosetta_code,         {}),
    ] if _PROGRAMMING_AVAILABLE else []),
]



# ═══════════════════════════════════════════════════════════════════
# PARALLEL ENCODE WORKER  (module-level — multiprocessing picklable)
# ═══════════════════════════════════════════════════════════════════

def _encode_worker(args):
    text, concept, tag = args
    try:
        eigenmodes = _E8.get()
        sig = encode_text(text, eigenmodes)
        if np.linalg.norm(sig) < 1e-10:
            return None
        return (text, sig.tobytes(), concept, tag)
    except Exception:
        return None


def _fetch_source(args):
    """
    Fetch one source in a thread. Returns list of (text, concept, tag).
    Runs in ThreadPoolExecutor — I/O bound, GIL release is fine.
    """
    src_name, domain, fn, kwargs, src_st = args
    results = []
    try:
        for text, concept, tag in fn(src_st, **kwargs):
            if is_good_sentence(text):
                results.append((text, concept, tag))
    except Exception:
        pass
    return src_name, src_st, results


# ═══════════════════════════════════════════════════════════════════
# MAINTENANCE SCRIPT
# ═══════════════════════════════════════════════════════════════════

class CorpusTrainer(MaintenanceScript):
    NAME        = "corpus_trainer"
    DESCRIPTION = "Continuous knowledge acquisition from science/education corpora"
    INTERVAL    = 0     # continuous

    MAX_RUN_SECONDS = 240  # hard cap so master stays responsive

    def run(self) -> dict:
        name = self.NAME
        st   = load_state(name)

        # Load crystal
        try:
            from language_crystal import LanguageCrystal
            crystal = LanguageCrystal()
        except Exception as e:
            log(name, f"Crystal load failed: {e}", "ERROR")
            return {"status": "error", "reason": str(e)}

        eigenmodes  = _E8.get()
        total_deposited = 0
        source_results  = {}
        run_start       = time.time()

        # ── Stage 1: Parallel fetch (all sources simultaneously) ────
        source_states = st.get("source_states", {})
        fetch_args = [
            (src_name, domain, fn, kwargs,
             source_states.get(src_name, {}))
            for src_name, domain, fn, kwargs in SOURCES
        ]
        all_items   = []   # (text, concept, tag) across all sources
        src_buffers = {}   # src_name -> list of items (for per-source counts)

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futures = {pool.submit(_fetch_source, a): a[0]
                       for a in fetch_args}
            for future in as_completed(futures):
                try:
                    src_name, src_st, items = future.result()
                    source_states[src_name] = src_st
                    src_buffers[src_name]   = items
                    all_items.extend(items)
                except Exception as e:
                    src_name = futures[future]
                    log(name, f"Fetch {src_name} error: {e}", "WARN")

        # ── Stage 2: Parallel encode ─────────────────────────────
        encoded = []   # (text, sig_bytes, concept, tag)
        if all_items:
            with ThreadPoolExecutor(max_workers=ENCODE_WORKERS) as pool:
                futures = {pool.submit(_encode_worker, item): item
                           for item in all_items}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        encoded.append(result)

        # ── Stage 3: Serial deposit into crystal ─────────────────
        # Track per-source counts for logging
        src_deposited = {src_name: 0 for src_name, *_ in SOURCES}

        for text, sig_bytes, concept, tag in encoded:
            sig = np.frombuffer(sig_bytes, dtype=np.float32).copy()
            for _ in range(OBSERVE_N):
                crystal.observe(text, sig, concept=concept, source=tag)
            total_deposited += 1
            # Attribute to source by tag match
            for src_name, domain, fn, kwargs in SOURCES:
                if src_name in tag or tag in src_name:
                    src_deposited[src_name] = src_deposited.get(src_name, 0) + 1
                    break

        # Log per-source results
        for src_name, domain, fn, kwargs in SOURCES:
            fetched = len(src_buffers.get(src_name, []))
            source_results[src_name] = fetched
            log(name, f"  {src_name}: {fetched} sentences deposited")

        # Save crystal (atomic)
        try:
            crystal.save()
        except Exception as e:
            log(name, f"Crystal save error: {e}", "WARN")

        # Save state
        st["source_states"]    = source_states
        st["total_deposited"]  = st.get("total_deposited", 0) + total_deposited
        st["last_run"]         = datetime.now().isoformat()
        st["runs_completed"]   = st.get("runs_completed", 0) + 1
        save_state(name, st)

        cs      = crystal.status()
        elapsed = time.time() - run_start
        summary = (f"{total_deposited} sentences deposited in {elapsed:.0f}s — "
                   f"crystal: {cs['total_vertices']:,} vertices, "
                   f"{cs['locked_vertices']:,} locked")
        log(name, summary)

        # Append to crystal log
        try:
            with open(CRYSTAL_LOG, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "deposited": total_deposited,
                    "by_source": source_results,
                    "total_vertices": cs["total_vertices"],
                    "locked": cs["locked_vertices"],
                }) + "\n")
        except Exception:
            pass

        time.sleep(REST_SECONDS)
        return {
            "status":    "ok",
            "deposited": total_deposited,
            "by_source": source_results,
            "crystal_vertices": cs["total_vertices"],
            "crystal_locked":   cs["locked_vertices"],
            "elapsed_s": round(elapsed, 1),
        }


# Allow direct execution for testing
if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--run-once", action="store_true",
                    help="Run one training pass and exit (called by master)")
    _args = _p.parse_args()
    if _args.run_once:
        script = CorpusTrainer()
        result = script.run()
        sys.exit(0 if result.get("status") == "ok" else 1)
    import importlib.util, pathlib
    # Bootstrap _lib if running standalone
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    trainer = CorpusTrainer()
    result  = trainer.run()
    print(json.dumps(result, indent=2))
