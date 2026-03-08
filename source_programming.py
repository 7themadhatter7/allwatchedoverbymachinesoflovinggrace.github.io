#!/usr/bin/env python3
"""
source_programming.py
━━━━━━━━━━━━━━━━━━━━━
Programming language corpus for RM's crystal substrate.

Strategy: deposit computational CONCEPTS alongside their expressions
in multiple languages. A for-loop, une boucle, eine Schleife, and
Python/Haskell/C/Lisp syntax are all the same geometric position.

The crystal should hold:
  - Deep CS theory: type theory, lambda calculus, computability,
    complexity, formal semantics, category theory in CS
  - Language paradigms: imperative, functional, OO, logic, concurrent
  - The same algorithm expressed in N languages (Rosetta Code pattern)
  - Official language documentation prose (dense conceptual content)
  - arXiv CS.PL papers on language design and semantics
  - Curated seed sentences: the conceptual skeleton of computation

Sources:
  1. Curated seeds — core definitions every CS student learns
  2. arXiv CS.PL, CS.LO, CS.DS — research on programming languages
  3. Official documentation prose — Python, MDN, Rust, Haskell
  4. Rosetta Code — same task, many languages, natural parallel corpus
  5. GitHub stdlib source comments — real-world code documentation
  6. Wikipedia CS articles — algorithms, data structures, paradigms
"""

import re, time, requests, logging
from typing import Iterator

log = logging.getLogger("programming_corpus")
HEADERS = {"User-Agent": "RM-Programming-Corpus/1.0 (Ghost in the Machine Labs)"}


# ── Seed corpus: computational concepts ──────────────────────────────────────
# Three layers per concept:
#   1. The abstract definition (what it IS)
#   2. The computational expression (what it DOES)
#   3. Cross-language equivalence (how it APPEARS)

SEED_CORPUS = [

    # ── Fundamental computation ───────────────────────────────────────────────
    ("A function maps inputs to outputs, producing the same output for the same input every time.", "functions", "seed"),
    ("Pure functions have no side effects — they neither read nor modify state outside themselves.", "functions", "seed"),
    ("A recursive function calls itself, reducing the problem until it reaches a base case.", "recursion", "seed"),
    ("Recursion and iteration are equivalent in expressive power but differ in clarity and stack usage.", "recursion", "seed"),
    ("The call stack records where to return after each function call completes.", "recursion", "seed"),
    ("Tail recursion can be optimized into iteration, eliminating stack growth entirely.", "recursion", "seed"),
    ("An algorithm is a finite sequence of well-defined steps that solves a problem.", "algorithms", "seed"),
    ("Time complexity measures how an algorithm's runtime scales with input size.", "complexity", "seed"),
    ("Space complexity measures how an algorithm's memory usage scales with input size.", "complexity", "seed"),
    ("O(n log n) is optimal for comparison-based sorting — no comparison sort can do better.", "complexity", "seed"),
    ("NP-complete problems have no known polynomial-time solution, but solutions can be verified quickly.", "complexity", "seed"),
    ("The halting problem is undecidable — no program can determine whether an arbitrary program halts.", "computability", "seed"),
    ("Turing completeness means a system can simulate any Turing machine given sufficient resources.", "computability", "seed"),
    ("Church's thesis equates effective computability with Turing computability.", "computability", "seed"),

    # ── Lambda calculus and functional programming ────────────────────────────
    ("Lambda calculus is a formal system for expressing computation through function abstraction and application.", "lambda_calculus", "seed"),
    ("In lambda calculus, everything is a function — numbers, booleans, and data structures are all encodable.", "lambda_calculus", "seed"),
    ("Beta reduction substitutes an argument into a function body, computing one step of evaluation.", "lambda_calculus", "seed"),
    ("A closure captures its enclosing scope, carrying free variables with it wherever it goes.", "closures", "seed"),
    ("Higher-order functions take functions as arguments or return functions as values.", "higher_order", "seed"),
    ("Map applies a function to every element of a collection, returning a new collection.", "higher_order", "seed"),
    ("Filter selects elements satisfying a predicate, returning a subset of the original collection.", "higher_order", "seed"),
    ("Reduce folds a collection into a single value by repeatedly applying a combining function.", "higher_order", "seed"),
    ("Currying transforms a function of multiple arguments into a chain of single-argument functions.", "currying", "seed"),
    ("Partial application fixes some arguments of a function, returning a function of the remaining ones.", "currying", "seed"),
    ("Immutability means values cannot be changed after creation — transformation creates new values.", "immutability", "seed"),
    ("Referential transparency means an expression can be replaced by its value without changing behavior.", "immutability", "seed"),
    ("Monads are a design pattern for sequencing computations with effects in a principled way.", "monads", "seed"),
    ("The Maybe monad handles computations that might fail without explicit null checking.", "monads", "seed"),
    ("Functors are types that can be mapped over, preserving structure while transforming content.", "category_theory_cs", "seed"),

    # ── Type theory ───────────────────────────────────────────────────────────
    ("A type system classifies values and enforces constraints that prevent certain classes of errors.", "type_theory", "seed"),
    ("Static typing catches type errors at compile time; dynamic typing catches them at runtime.", "type_theory", "seed"),
    ("Type inference deduces types automatically from context, without requiring explicit annotations.", "type_theory", "seed"),
    ("Parametric polymorphism allows functions to operate on values of any type uniformly.", "type_theory", "seed"),
    ("Algebraic data types combine product types and sum types to express complex data structures.", "type_theory", "seed"),
    ("A sum type represents a value that is exactly one of several alternatives.", "type_theory", "seed"),
    ("A product type combines multiple values into a single compound value.", "type_theory", "seed"),
    ("The Curry-Howard correspondence equates types with propositions and programs with proofs.", "type_theory", "seed"),
    ("Dependent types allow types to depend on values, enabling very precise specifications.", "type_theory", "seed"),
    ("Linear types ensure resources are used exactly once, preventing use-after-free and double-free.", "type_theory", "seed"),
    ("Ownership in Rust is a type-level guarantee that memory is managed correctly without a garbage collector.", "type_theory", "seed"),

    # ── Data structures ───────────────────────────────────────────────────────
    ("A linked list stores elements in nodes, each pointing to the next, enabling O(1) insertion.", "data_structures", "seed"),
    ("A hash table maps keys to values using a hash function, achieving O(1) average lookup.", "data_structures", "seed"),
    ("A binary search tree maintains sorted order, enabling O(log n) search, insertion, and deletion.", "data_structures", "seed"),
    ("A heap is a tree satisfying the heap property, efficiently supporting priority queue operations.", "data_structures", "seed"),
    ("A graph represents relationships between entities as vertices connected by edges.", "data_structures", "seed"),
    ("A stack is LIFO — last in, first out — supporting push and pop in O(1).", "data_structures", "seed"),
    ("A queue is FIFO — first in, first out — supporting enqueue and dequeue in O(1).", "data_structures", "seed"),
    ("Persistent data structures preserve previous versions after modification.", "data_structures", "seed"),
    ("Tries store strings as paths from root to leaf, enabling O(k) lookup where k is key length.", "data_structures", "seed"),

    # ── Paradigms ─────────────────────────────────────────────────────────────
    ("Imperative programming describes computation as sequences of statements that change program state.", "paradigms", "seed"),
    ("Functional programming treats computation as the evaluation of mathematical functions.", "paradigms", "seed"),
    ("Object-oriented programming organizes code around objects that combine data and behavior.", "paradigms", "seed"),
    ("Logic programming expresses computation as relations and derives answers through inference.", "paradigms", "seed"),
    ("Concurrent programming manages multiple computations executing simultaneously.", "paradigms", "seed"),
    ("Reactive programming models systems as data flows that propagate changes automatically.", "paradigms", "seed"),
    ("Declarative programming specifies what to compute rather than how to compute it.", "paradigms", "seed"),

    # ── Language families and their concepts ──────────────────────────────────
    # Python
    ("Python's duck typing means an object's suitability is determined by its methods, not its type.", "python", "seed"),
    ("Python generators yield values lazily, computing each element only when requested.", "python", "seed"),
    ("Python decorators wrap functions to modify their behavior without changing their source code.", "python", "seed"),
    ("Python's GIL prevents true thread parallelism but allows cooperative multitasking via asyncio.", "python", "seed"),
    ("List comprehensions express filtered and transformed collections in a single readable expression.", "python", "seed"),
    # Haskell / ML family
    ("Haskell is purely functional — all functions are pure and effects are managed through the type system.", "haskell", "seed"),
    ("Pattern matching in Haskell deconstructs data structures and dispatches on their shape.", "haskell", "seed"),
    ("Lazy evaluation in Haskell defers computation until the result is needed.", "haskell", "seed"),
    ("ML's Hindley-Milner type inference reconstructs types without any annotations.", "haskell", "seed"),
    # Lisp family
    ("Lisp represents code as data — programs are lists, enabling macros that transform programs.", "lisp", "seed"),
    ("Homoiconicity means the language's code has the same structure as its primary data structure.", "lisp", "seed"),
    ("Scheme's continuations reify the rest of the computation as a first-class value.", "lisp", "seed"),
    ("Common Lisp's CLOS is one of the most expressive object systems ever designed.", "lisp", "seed"),
    # C family
    ("C gives direct access to memory through pointers, enabling precise control at the cost of safety.", "c_family", "seed"),
    ("C++'s RAII ties resource lifetime to object scope, ensuring cleanup on exit.", "c_family", "seed"),
    ("Rust's borrow checker enforces memory safety at compile time without a garbage collector.", "rust", "seed"),
    ("Rust's ownership model eliminates data races by preventing shared mutable state.", "rust", "seed"),
    # JavaScript
    ("JavaScript's prototype chain implements inheritance through object delegation.", "javascript", "seed"),
    ("Promises represent future values, enabling non-blocking asynchronous computation.", "javascript", "seed"),
    ("JavaScript's event loop processes callbacks from a queue after the call stack empties.", "javascript", "seed"),
    # SQL and logic
    ("SQL expresses queries declaratively — you state what data you want, not how to retrieve it.", "sql", "seed"),
    ("Relational algebra provides the mathematical foundation for SQL and relational databases.", "sql", "seed"),
    ("Prolog's unification matches terms by finding variable bindings that make them identical.", "prolog", "seed"),
    # Systems concepts
    ("A garbage collector automatically reclaims memory that is no longer reachable.", "memory_management", "seed"),
    ("Reference counting tracks how many references point to an object, freeing it when count reaches zero.", "memory_management", "seed"),
    ("A virtual machine executes bytecode, abstracting over the underlying hardware.", "virtual_machines", "seed"),
    ("Just-in-time compilation compiles hot code paths to native instructions at runtime.", "virtual_machines", "seed"),
    ("Continuations passing style makes control flow explicit as a chain of function calls.", "continuation", "seed"),

    # ── Formal semantics ──────────────────────────────────────────────────────
    ("Operational semantics defines meaning by describing how programs execute step by step.", "formal_semantics", "seed"),
    ("Denotational semantics defines meaning by mapping programs to mathematical objects.", "formal_semantics", "seed"),
    ("Axiomatic semantics defines meaning through pre and post conditions on program states.", "formal_semantics", "seed"),
    ("The fixed point of a function f is a value x such that f(x) equals x.", "formal_semantics", "seed"),
    ("Least fixed points define the semantics of recursive definitions in domain theory.", "formal_semantics", "seed"),

    # ── Concurrency ───────────────────────────────────────────────────────────
    ("A mutex ensures only one thread executes a critical section at a time.", "concurrency", "seed"),
    ("Deadlock occurs when two processes each wait for a resource held by the other.", "concurrency", "seed"),
    ("The actor model treats concurrent entities as actors that communicate only through messages.", "concurrency", "seed"),
    ("Software transactional memory treats memory operations like database transactions.", "concurrency", "seed"),
    ("Coroutines are functions that can suspend and resume, enabling cooperative multitasking.", "concurrency", "seed"),

    # ── Cross-language concept equivalences ───────────────────────────────────
    ("A for loop in Python, a map in Haskell, and a forEach in JavaScript express the same iteration concept.", "cross_language", "seed"),
    ("Null in Java, None in Python, nil in Ruby, and Nothing in Haskell all encode the absence of a value.", "cross_language", "seed"),
    ("Pattern matching in Haskell, match in Rust, and switch in C all dispatch on the shape of data.", "cross_language", "seed"),
    ("Python's list comprehension, Haskell's list monad, and SQL's SELECT WHERE express the same filter-map.", "cross_language", "seed"),
    ("Interfaces in Java, traits in Rust, typeclasses in Haskell, and protocols in Swift encode the same polymorphism.", "cross_language", "seed"),
    ("A promise in JavaScript, a Future in Rust, and an IO action in Haskell encode deferred computation.", "cross_language", "seed"),
    ("Python decorators, Java annotations, and Lisp macros all operate on code as data.", "cross_language", "seed"),
    ("Memory management: C uses manual malloc/free, Python uses reference counting, Haskell uses lazy GC.", "cross_language", "seed"),
]

# ── arXiv CS queries ──────────────────────────────────────────────────────────

ARXIV_CS_QUERIES = [
    ("cs.PL", "type+theory+programming+language",          "type_theory"),
    ("cs.PL", "lambda+calculus+semantics",                 "lambda_calculus"),
    ("cs.PL", "functional+programming+monads",             "functional_programming"),
    ("cs.PL", "dependent+types+proof+assistant",           "type_theory"),
    ("cs.PL", "ownership+memory+safety+rust",              "memory_safety"),
    ("cs.LO", "curry+howard+propositions+proofs",          "type_theory"),
    ("cs.LO", "linear+logic+programming",                  "linear_types"),
    ("cs.DS", "algorithm+complexity+data+structure",       "algorithms"),
    ("cs.PL", "concurrency+actor+model+message+passing",   "concurrency"),
    ("cs.PL", "compiler+optimization+intermediate+representation", "compilers"),
    ("cs.PL", "gradual+typing+dynamic+static",             "type_theory"),
    ("cs.PL", "effect+system+algebraic+effects",           "effects"),
    ("cs.PL", "program+synthesis+specification",           "program_synthesis"),
    ("cs.LO", "modal+logic+temporal+verification",         "formal_verification"),
    ("cs.PL", "garbage+collection+memory+management",      "memory_management"),
]

# ── Official documentation sources ───────────────────────────────────────────

DOCUMENTATION_SOURCES = [
    # (url, domain, tag, selector_hint)
    ("https://docs.python.org/3/glossary.html",
     "python", "python_glossary"),
    ("https://docs.python.org/3/library/functions.html",
     "python", "python_builtins"),
    ("https://doc.rust-lang.org/book/ch01-00-getting-started.html",
     "rust", "rust_book"),
    ("https://www.haskell.org/tutorial/functions.html",
     "haskell", "haskell_tutorial"),
]

# ── Wikipedia CS articles ─────────────────────────────────────────────────────

WIKIPEDIA_CS_ARTICLES = [
    ("Lambda_calculus",              "lambda_calculus"),
    ("Type_theory",                  "type_theory"),
    ("Curry%E2%80%93Howard_correspondence", "type_theory"),
    ("Monad_(functional_programming)", "monads"),
    ("Closure_(computer_programming)", "closures"),
    ("Recursive_function",           "recursion"),
    ("Big_O_notation",               "complexity"),
    ("NP-completeness",              "complexity"),
    ("Halting_problem",              "computability"),
    ("Turing_completeness",          "computability"),
    ("Garbage_collection_(computer_science)", "memory_management"),
    ("Functional_programming",       "paradigms"),
    ("Object-oriented_programming",  "paradigms"),
    ("Logic_programming",            "paradigms"),
    ("Concurrent_computing",         "concurrency"),
    ("Actor_model",                  "concurrency"),
    ("Algebraic_data_type",          "type_theory"),
    ("Pattern_matching",             "type_theory"),
    ("Continuation",                 "continuation"),
    ("Coroutine",                    "concurrency"),
    ("Hash_table",                   "data_structures"),
    ("Binary_search_tree",           "data_structures"),
    ("Graph_(abstract_data_type)",   "data_structures"),
    ("Dynamic_programming",          "algorithms"),
    ("Divide_and_conquer_algorithm", "algorithms"),
    ("Formal_semantics_of_programming_languages", "formal_semantics"),
    ("Denotational_semantics",       "formal_semantics"),
    ("Operational_semantics",        "formal_semantics"),
    ("Dependent_type",               "type_theory"),
    ("Linear_type_system",           "type_theory"),
    ("Ownership_(object-oriented_programming)", "memory_safety"),
    ("Hindley%E2%80%93Milner_type_system", "type_theory"),
    ("Category_theory",              "category_theory_cs"),
    ("Functor",                      "category_theory_cs"),
    ("Lisp_(programming_language)",  "lisp"),
    ("Haskell_(programming_language)", "haskell"),
    ("Python_(programming_language)", "python"),
    ("Rust_(programming_language)",  "rust"),
    ("Prolog",                       "prolog"),
    ("SQL",                          "sql"),
]

# ── Rosetta Code tasks — same problem in many languages ───────────────────────
# These are golden: pure parallel programming corpus

ROSETTA_CODE_TASKS = [
    ("Fibonacci_sequence",           "recursion"),
    ("Factorial",                    "recursion"),
    ("Quicksort",                    "algorithms"),
    ("Merge_sort",                   "algorithms"),
    ("Binary_search",                "algorithms"),
    ("Tree_traversal",               "data_structures"),
    ("Towers_of_Hanoi",              "recursion"),
    ("Sieve_of_Eratosthenes",        "algorithms"),
    ("Greatest_common_divisor",      "algorithms"),
    ("Y_combinator",                 "lambda_calculus"),
    ("Higher-order_functions",       "higher_order"),
    ("Map",                          "higher_order"),
    ("Filter",                       "higher_order"),
    ("Fold",                         "higher_order"),
    ("Currying",                     "currying"),
    ("Closures",                     "closures"),
    ("Memoization",                  "algorithms"),
    ("Mutual_recursion",             "recursion"),
    ("Tail_recursion",               "recursion"),
    ("Pattern_matching",             "type_theory"),
    ("Algebraic_data_types",         "type_theory"),
    ("Monads",                       "monads"),
    ("Continuations",                "continuation"),
    ("Coroutines",                   "concurrency"),
    ("Producer-consumer",            "concurrency"),
    ("Dining_philosophers",          "concurrency"),
]


def sentences_from_text(text, min_len=20, max_len=500):
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    result = []
    for s in sents:
        s = re.sub(r'\s+', ' ', s).strip()
        if min_len <= len(s) <= max_len and len(s.split()) >= 5:
            if not re.search(r'\[\d+\]|\{\{|\}\}|http|<[^>]+>', s):
                result.append(s)
    return result


def fetch_arxiv(category, query, domain, max_results=25):
    url = f"https://export.arxiv.org/api/query?search_query=cat:{category}+AND+all:{query}&start=0&max_results={max_results}"
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        if r.status_code != 200: return []
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            if title is not None:
                t = title.text.strip().replace("\n", " ")
                if len(t) > 20:
                    results.append((t, domain, f"arxiv_{category}"))
            if summary is not None:
                for sent in sentences_from_text(summary.text or "")[:5]:
                    results.append((sent, domain, f"arxiv_{category}"))
        return results
    except Exception as e:
        log.debug(f"arXiv {category}/{query}: {e}")
        return []


def fetch_wikipedia(article, domain):
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}",
            headers=HEADERS, timeout=8)
        if r.status_code != 200: return []
        data = r.json()
        return [(s, domain, f"wiki_cs_{domain[:15]}")
                for s in sentences_from_text(data.get("extract", ""))[:7]]
    except Exception as e:
        log.debug(f"Wikipedia {article}: {e}")
        return []


def fetch_rosetta_code(task, domain):
    """
    Fetch Rosetta Code task page. Extract the prose description
    (not the code itself) and any cross-language notes.
    Code comments and docstrings are also valuable.
    """
    try:
        r = requests.get(
            f"https://rosettacode.org/wiki/{task}",
            headers=HEADERS, timeout=12)
        if r.status_code != 200: return []

        # Extract text, skip code blocks
        text = r.text
        # Remove script/style
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        # Remove code blocks (preserve comments within them below)
        code_comments = re.findall(r'(?:#|//|--|;)\s*([A-Z][^<\n]{20,200})', text)
        # Strip HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        results = []
        # Prose description sentences
        for s in sentences_from_text(text)[:6]:
            results.append((s, domain, f"rosetta_{task[:20]}"))
        # Code comments that are sentences
        for c in code_comments[:4]:
            c = c.strip()
            if len(c) > 20:
                results.append((c, domain, f"rosetta_comment_{task[:15]}"))
        return results
    except Exception as e:
        log.debug(f"Rosetta {task}: {e}")
        return []


def source_programming_seeds(state, batch=10000):
    """Deposit the full seed corpus — highest priority, done once."""
    if state.get("seeds_done"):
        return
    # Triple reinforce all seeds
    for repeat in range(3):
        for text, concept, tag in SEED_CORPUS:
            yield (text, concept, f"{tag}_r{repeat}")
    state["seeds_done"] = True


def source_programming_arxiv(state, batch=2000):
    """Fetch arXiv CS papers on PL theory and algorithms."""
    done = state.get("arxiv_done", [])
    count = 0
    for category, query, domain in ARXIV_CS_QUERIES:
        key = f"{category}_{query[:20]}"
        if key in done: continue
        items = fetch_arxiv(category, query, domain)
        for item in items:
            yield item
            count += 1
        done.append(key)
        state["arxiv_done"] = done
        time.sleep(1)
        if count >= batch: return


def source_programming_wikipedia(state, batch=2000):
    """Fetch Wikipedia articles on CS concepts."""
    done = state.get("wiki_done", [])
    count = 0
    for article, domain in WIKIPEDIA_CS_ARTICLES:
        if article in done: continue
        for item in fetch_wikipedia(article, domain):
            yield item
            count += 1
        done.append(article)
        state["wiki_done"] = done
        time.sleep(0.3)
        if count >= batch: return


def source_rosetta_code(state, batch=1500):
    """Fetch Rosetta Code task descriptions — algorithmic concepts across languages."""
    done = state.get("rosetta_done", [])
    count = 0
    for task, domain in ROSETTA_CODE_TASKS:
        if task in done: continue
        for item in fetch_rosetta_code(task, domain):
            yield item
            count += 1
        done.append(task)
        state["rosetta_done"] = done
        time.sleep(0.5)
        if count >= batch: return


def get_programming_sources():
    """
    Returns source tuples for corpus trainer SOURCES list.
    Add these to 12_corpus_trainer.py SOURCES.
    """
    return [
        ("prog_seeds",     "computer_science",  source_programming_seeds,    {}),
        ("prog_arxiv",     "computer_science",  source_programming_arxiv,    {}),
        ("prog_wikipedia", "computer_science",  source_programming_wikipedia,{}),
        ("prog_rosetta",   "computer_science",  source_rosetta_code,         {}),
    ]


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys, numpy as np
    from pathlib import Path
    sys.path.insert(0, "/home/joe/sparky")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--max",   type=int, default=40000)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    STATE_FILE = "/home/joe/sparky/logs/programming_corpus_state.json"
    LOG_FILE   = "/home/joe/sparky/logs/programming_corpus.log"

    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    if args.reset and Path(STATE_FILE).exists():
        Path(STATE_FILE).unlink()

    try:
        state = json.load(open(STATE_FILE))
    except:
        state = {}

    from language_crystal import LanguageCrystal
    from mother_english_io_v5 import E8Substrate, WordEncoder

    log.info("Loading crystal and encoder...")
    crystal  = LanguageCrystal()
    substrate = E8Substrate()
    encoder  = WordEncoder(substrate)
    log.info(f"Crystal: {crystal.status()}")

    sources = [
        ("Seeds",     source_programming_seeds,    {}),
        ("arXiv CS",  source_programming_arxiv,    {}),
        ("Wikipedia", source_programming_wikipedia,{}),
        ("Rosetta",   source_rosetta_code,         {}),
    ]

    total = 0
    for name, fn, kwargs in sources:
        log.info(f"\n{'━'*55}")
        log.info(f"Source: {name}")
        log.info(f"{'━'*55}")

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
    cs = crystal.status()
    log.info(f"\n{'━'*55}")
    log.info(f"Total deposited: {total:,}")
    log.info(f"Crystal: {cs}")
