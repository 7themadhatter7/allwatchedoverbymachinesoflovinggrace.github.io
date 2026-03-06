#!/usr/bin/env python3
"""
triple_polish.py
================
Ghost in the Machine Labs

Dialog improvement loop: score triples → classify decoherence →
generate repair questions → accept author answers → resubmit.
Loops until 100% triple quality or author exhausts resolvable issues.

Usage:
  python3 triple_polish.py {stem}_triples.json [--auto] [--max-rounds N]

  --auto        Non-interactive: apply all automatic fixes only, skip
                questions requiring human input. Useful for CI / scripted runs.
  --max-rounds  Maximum dialog rounds before halting (default: unlimited).

Output:
  {stem}_triples_polished.json   — clean triple set
  {stem}_polish_report.json      — full audit: what was fixed, what remains
  {stem}_unresolved.json         — any triples that could not reach unity
                                   (empty if 100% achieved)

Triple quality score: 0.0 – 1.0
  1.0  clean atomic concept triple
  0.0  complete noise, auto-discarded
  0.1–0.9  degraded — classified and queued for repair

Failure modes (from taxonomy):
  HEADING_NUM    section numbers canonicalized as concept tokens
  CLAUSE_FRAG    mid-clause verb/prep phrase captured as subject/object
  PRONOUN        demonstrative or pronoun used as subject
  PROSE_SHARD    sentence fragment as object in defined_by triple
  ANTI_PATTERN   negatively-valenced instruction encoded as positive concept
  MISSING_IFACE  pipeline interface contract not expressed as triple
"""

import json
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict, Counter


# ─── SCORING RULES ────────────────────────────────────────────────────────────

# Tokens that should never appear as subjects or objects
PRONOUN_SET = {
    "this", "that", "these", "those", "it", "its", "they", "their",
    "the", "a", "an", "which", "who", "where", "when", "how", "what",
    "here", "there", "such", "both", "all", "each", "every", "any",
}

# Known multi-word principle/concept labels that are NOT clause fragments
# These come from bullet "Label: description" format and are valid concept names
KNOWN_PRINCIPLE_TOKENS = {
    "rm_is_the_programmer_not_the_computer",
    "llm_as_plugin_not_core",
    "multi_example_consensus",
    "council_governance",
    "ram_resident_computation",
    "geometric_field_primacy",
    "rules_are_only_accepted_when_they_validate",
}

# Patterns indicating a clause fragment rather than an atomic concept
CLAUSE_FRAGMENT_PATTERNS = [
    re.compile(r"^(and|or|but|so|then|thus|hence|therefore|however|also)_"),
    re.compile(r"_(and|or|but|so|then|thus|also|is|are|was|were|be|been)$"),
    re.compile(r"^(it|this|that|these|those)_"),
    re.compile(r"_(that|which|who|where|when)_"),
    re.compile(r"_(is_the|is_a|is_an|are_the|are_a)_"),
]

# Heading number prefix: starts with digits then underscore
HEADING_NUM_RE = re.compile(r"^\d{1,3}[\d_]*[a-z]")

# Anti-pattern section markers (normalized)
ANTI_PATTERN_MARKERS = {
    "anti_pattern", "anti_patterns", "avoid", "never", "do_not",
    "pitfall", "pitfalls", "mistake", "mistakes", "wrong", "bad",
    "design_anti_patterns_to_avoid", "132_design_anti_patterns_to_avoid",
}

# Pipeline phase surface forms for interface detection
PHASE_TOKENS = {f"pipeline_phase_{i}" for i in range(7)}


@dataclass
class ScoredTriple:
    subject:    str
    relation:   str
    object:     str
    source:     str = ""
    score:      float = 1.0
    failures:   list  = field(default_factory=list)
    repaired:   bool  = False
    discarded:  bool  = False
    repair_note: str  = ""

    def to_dict(self):
        return asdict(self)


# ─── SCORER ───────────────────────────────────────────────────────────────────

class TripleScorer:
    """
    Score every triple 0.0–1.0.
    Returns list of ScoredTriple with failure modes attached.
    """

    def score(self, triples: list) -> list:
        scored = []
        for t in triples:
            s = ScoredTriple(
                subject  = t["subject"],
                relation = t["relation"],
                object   = t["object"],
                source   = t.get("source", ""),
            )
            self._apply_rules(s)
            scored.append(s)
        return scored

    def _apply_rules(self, t: ScoredTriple):
        # Rule 1: heading number tokens
        if HEADING_NUM_RE.match(t.subject):
            t.failures.append("HEADING_NUM:subject")
            t.score -= 0.5
        if HEADING_NUM_RE.match(t.object):
            t.failures.append("HEADING_NUM:object")
            t.score -= 0.4

        # Rule 2: clause fragments (skip known principle tokens)
        if t.subject not in KNOWN_PRINCIPLE_TOKENS:
            for pat in CLAUSE_FRAGMENT_PATTERNS:
                if pat.search(t.subject):
                    t.failures.append("CLAUSE_FRAG:subject")
                    t.score -= 0.5
                    break
        if t.object not in KNOWN_PRINCIPLE_TOKENS:
            for pat in CLAUSE_FRAGMENT_PATTERNS:
                if pat.search(t.object):
                    t.failures.append("CLAUSE_FRAG:object")
                    t.score -= 0.3
                    break

        # Rule 3: pronoun subjects — only single-word pronouns or short demonstratives
        # "the_e8_lie_group" is a valid concept; "this_phase" is a pronoun reference
        subj_first = t.subject.split("_")[0]
        is_pronoun = (
            t.subject in PRONOUN_SET or
            (subj_first in {"this","that","these","those","it","its"} and
             t.subject.count("_") <= 2)
        )
        if is_pronoun:
            t.failures.append("PRONOUN:subject")
            t.score -= 0.6

        # Rule 4: prose shards — object in defined_by with > 4 words and no canonical form
        if t.relation == "defined_by" and t.object.count("_") > 3:
            t.failures.append("PROSE_SHARD:object")
            t.score -= 0.35

        # Rule 5: anti-pattern encoded as positive concept
        # Require BOTH source AND content to indicate anti-pattern (avoids table bleed)
        src = t.source.lower()
        src_is_antip = any(m in src for m in ANTI_PATTERN_MARKERS)
        subj_is_antip = any(m in t.subject for m in ANTI_PATTERN_MARKERS)
        # Only flag if source explicitly marks it as anti-pattern section
        # AND the triple is a positive assertion (not already a violates relation)
        if src_is_antip and subj_is_antip and t.relation != "violates":
            t.failures.append("ANTI_PATTERN")
            t.score -= 0.4

        # Rule 6: token too long — indicates a sentence absorbed as concept
        if t.subject.count("_") > 6:
            if "CLAUSE_FRAG:subject" not in t.failures:
                t.failures.append("CLAUSE_FRAG:subject")
                t.score -= 0.4
        if t.object.count("_") > 6:
            if "CLAUSE_FRAG:object" not in t.failures:
                t.failures.append("CLAUSE_FRAG:object")
                t.score -= 0.3

        # Rule 7: numeric-only or single-char tokens
        if re.match(r"^\d+$", t.subject) or len(t.subject) <= 1:
            t.failures.append("NOISE:subject")
            t.score = 0.0

        # Floor
        t.score = max(0.0, round(t.score, 3))

        # Auto-discard complete noise
        if t.score == 0.0:
            t.discarded = True


# ─── DECOHERENCE CLASSIFIER ───────────────────────────────────────────────────

class DecoherenceClassifier:
    """
    Given scored triples, group degraded ones by primary failure mode
    and assign a repair strategy to each group.
    """

    REPAIR_STRATEGIES = {
        "HEADING_NUM":    "auto",      # strip numeric prefix, remap to canonical section token
        "CLAUSE_FRAG":    "question",  # ask author for correct subject/object
        "PRONOUN":        "question",  # ask author what the pronoun refers to
        "PROSE_SHARD":    "auto",      # truncate object to first 2 meaningful words
        "ANTI_PATTERN":   "question",  # ask author for polarity encoding
        "MISSING_IFACE":  "question",  # ask author for interface contract
    }

    def classify(self, scored: list) -> dict:
        """Returns {failure_mode: [ScoredTriple]}"""
        groups = defaultdict(list)
        for t in scored:
            if t.discarded or t.score >= 1.0:
                continue
            # Primary failure mode = first in list
            if t.failures:
                primary = t.failures[0].split(":")[0]
                groups[primary].append(t)
        return dict(groups)

    def detect_missing_interfaces(self, scored: list) -> list:
        """
        Find pipeline phase pairs where no produces/consumes triple
        connects them. Returns synthetic ScoredTriple placeholders.
        """
        phase_produces = defaultdict(set)
        phase_consumes = defaultdict(set)
        for t in scored:
            if t.subject in PHASE_TOKENS and t.relation == "produces":
                phase_produces[t.subject].add(t.object)
            if t.subject in PHASE_TOKENS and t.relation == "consumes":
                phase_consumes[t.subject].add(t.object)

        missing = []
        phases = sorted(PHASE_TOKENS)
        for i in range(len(phases) - 1):
            p_cur  = phases[i]
            p_next = phases[i + 1]
            if not phase_produces[p_cur] or not phase_consumes[p_next]:
                synthetic = ScoredTriple(
                    subject  = p_cur,
                    relation = "produces",
                    object   = p_next,
                    source   = "synthetic:missing_interface",
                    score    = 0.0,
                    failures = ["MISSING_IFACE"],
                )
                missing.append(synthetic)
        return missing


# ─── AUTO REPAIR ──────────────────────────────────────────────────────────────

class AutoRepairer:
    """
    Apply repairs that don't require human input.
    Modifies ScoredTriple in place. Returns count of repairs applied.
    """

    # Maps heading-number prefixed tokens → canonical section names
    # Built dynamically from the triple set
    SECTION_MAP = {}  # populated at runtime from source field

    def build_section_map(self, scored: list):
        """Extract heading number → canonical name mapping from source fields."""
        for t in scored:
            src = t.source
            m = re.search(r"heading:(\d[\d\.]*)\s+(.+)", src)
            if m:
                num   = m.group(1).replace(".", "_")
                title = m.group(2).strip().lower()
                title = re.sub(r"[^\w\s]", "", title)
                title = re.sub(r"\s+", "_", title)
                # Both the prefixed form and just the number
                self.SECTION_MAP[num + "_" + title[:40]] = title[:40]
                self.SECTION_MAP[num]                    = title[:40]

    def repair(self, scored: list) -> int:
        self.build_section_map(scored)
        count = 0

        for t in scored:
            if t.discarded or t.score >= 1.0 or t.repaired:
                continue

            repaired_any = False

            # ── HEADING_NUM: strip numeric prefix ──
            if "HEADING_NUM:subject" in t.failures:
                clean = self._strip_heading_num(t.subject)
                if clean != t.subject:
                    t.repair_note += f"subject: {t.subject!r} → {clean!r}. "
                    t.subject = clean
                    t.failures.remove("HEADING_NUM:subject")
                    repaired_any = True

            if "HEADING_NUM:object" in t.failures:
                clean = self._strip_heading_num(t.object)
                if clean != t.object:
                    t.repair_note += f"object: {t.object!r} → {clean!r}. "
                    t.object = clean
                    t.failures.remove("HEADING_NUM:object")
                    repaired_any = True

            # ── PROSE_SHARD / CLAUSE_FRAG:object — truncate to 2 content words ──
            for frag_flag in ("PROSE_SHARD:object", "CLAUSE_FRAG:object"):
                if frag_flag in t.failures:
                    clean = self._truncate_to_noun(t.object)
                    if clean and clean != t.object:
                        t.repair_note += f"{frag_flag} object: {t.object!r} → {clean!r}. "
                        t.object = clean
                        t.failures.remove(frag_flag)
                        repaired_any = True
                    break

            # ── ANTI_PATTERN: re-encode as violates constraint ──
            if "ANTI_PATTERN" in t.failures:
                old_subj = t.subject
                AP_PARENT = "design_anti_patterns_to_avoid"
                stop_words = {"the","a","an","on","in","of","to","for","by","not","no","is","are"}

                if t.subject == AP_PARENT and t.relation == "has_component":
                    # Parent node: re-encode object as antipattern subject with violates
                    obj = t.object
                    obj_words = obj.split("_")
                    c = [w for w in obj_words if w not in stop_words and len(w) > 2]
                    short_obj = "_".join(c[:3]) if c else "_".join(obj_words[:3])
                    t.subject  = f"antipattern_{short_obj}"
                    t.relation = "violates"
                    t.object   = "architectural_principle"
                    t.repair_note += f"anti-pattern parent re-encoded. "
                    t.failures.remove("ANTI_PATTERN")
                    repaired_any = True

                elif t.subject.startswith(AP_PARENT) or AP_PARENT in t.subject:
                    # Child defined_by: re-encode
                    obj_words = t.object.split("_")
                    c = [w for w in obj_words if w not in stop_words and len(w) > 2]
                    short_obj = "_".join(c[:3]) if c else "_".join(obj_words[:3])
                    subj_words = t.subject.replace(AP_PARENT,"").strip("_").split("_")
                    cs = [w for w in subj_words if w not in stop_words and len(w) > 2]
                    short_subj = "_".join(cs[:3]) if cs else "_".join(subj_words[:3])
                    if short_subj:
                        t.subject  = f"antipattern_{short_subj}"
                        t.relation = "violates"
                        t.object   = short_obj if short_obj else "architectural_principle"
                        t.repair_note += f"anti-pattern child re-encoded. "
                        t.failures.remove("ANTI_PATTERN")
                        repaired_any = True

            # Rescore after repairs
            if repaired_any:
                old_score = t.score
                self._rescore(t)
                count += 1
                if t.score >= 1.0:
                    t.repaired = True

        return count

    def _strip_heading_num(self, token: str) -> str:
        """Remove leading numeric section prefix from token."""
        # Try section map first
        if token in self.SECTION_MAP:
            return self.SECTION_MAP[token]
        # Otherwise strip leading digits and underscores
        stripped = re.sub(r"^\d+_", "", token)
        # If still starts with digit, strip again
        stripped = re.sub(r"^\d+_?", "", stripped)
        return stripped if stripped and len(stripped) > 1 else token

    def _truncate_to_noun(self, token: str) -> str:
        """Truncate long prose-shard token to first 2 content words."""
        words = token.split("_")
        # Skip stop words
        stop = {"the","a","an","is","are","was","were","be","been",
                "this","that","it","its","and","or","of","in","on",
                "to","for","by","with","from","all","any","each"}
        content = [w for w in words if w not in stop and len(w) > 2]
        if len(content) >= 2:
            return "_".join(content[:2])
        elif len(content) == 1:
            return content[0]
        return token

    def _rescore(self, t: ScoredTriple):
        """Re-run scoring rules on repaired triple."""
        scorer = TripleScorer()
        fresh = ScoredTriple(subject=t.subject, relation=t.relation,
                             object=t.object, source=t.source)
        scorer._apply_rules(fresh)
        t.score    = fresh.score
        # Keep only failures that weren't already repaired
        t.failures = [f for f in t.failures if f in fresh.failures]


# ─── DIALOG ENGINE ────────────────────────────────────────────────────────────

class DialogEngine:
    """
    Interactive repair loop. Presents targeted questions to the author,
    accepts structured answers, injects corrected triples.

    Answer format per question type:
      CLAUSE_FRAG / PRONOUN:
        > subject=<token>, object=<token>   (to replace both)
        > subject=<token>                   (to replace subject only)
        > object=<token>                    (to replace object only)
        > discard                           (remove the triple)
        > skip                              (leave for unresolved report)

      ANTI_PATTERN:
        > polarity=negative                 (re-encode as antipattern_X violates principle_Y)
        > polarity=positive                 (keep as-is, was mis-classified)
        > discard
        > skip

      MISSING_IFACE:
        > contract=<produces_token>         (e.g. contract=intent_vector)
        > skip
    """

    def __init__(self, auto: bool = False):
        self.auto = auto   # if True, skip all questions — only auto repairs applied

    def run_round(self, scored: list, classified: dict, missing: list) -> tuple:
        """
        Run one dialog round. Returns (repairs_this_round, skipped).
        Modifies scored triples in place.
        """
        repairs = 0
        skipped = []

        all_items = []
        for mode, items in classified.items():
            strategy = DecoherenceClassifier.REPAIR_STRATEGIES.get(mode, "question")
            if strategy == "question":
                all_items.extend((mode, t) for t in items if not t.repaired and not t.discarded)

        # Add missing interfaces
        for t in missing:
            all_items.append(("MISSING_IFACE", t))

        if not all_items:
            return 0, []

        if self.auto:
            # Auto mode: skip all dialog items, mark as unresolved
            for mode, t in all_items:
                skipped.append(t)
            return 0, skipped

        total = len(all_items)
        print(f"\n{'─'*60}")
        print(f"  DIALOG REPAIR — {total} item(s) need your input")
        print(f"  Type 'help' at any prompt for answer format.")
        print(f"  Type 'done' to stop early (remaining → unresolved report).")
        print(f"{'─'*60}\n")

        for idx, (mode, t) in enumerate(all_items, 1):
            print(f"[{idx}/{total}] Mode: {mode}")
            print(f"  Source   : {t.source}")
            print(f"  Triple   : ({t.subject}) --[{t.relation}]--> ({t.object})")
            print(f"  Score    : {t.score}")
            print(f"  Failures : {t.failures}")
            print()

            prompt = self._build_prompt(mode, t)
            print(prompt)

            while True:
                try:
                    answer = input("  > ").strip()
                except EOFError:
                    answer = "skip"

                if answer.lower() == "done":
                    # Mark all remaining as skipped
                    remaining_idx = [i for i in range(idx - 1, len(all_items))]
                    for i in remaining_idx:
                        skipped.append(all_items[i][1])
                    return repairs, skipped

                if answer.lower() == "help":
                    self._print_help(mode)
                    continue

                result = self._parse_answer(answer, mode, t)
                if result == "invalid":
                    print("  ✗ Invalid format. Type 'help' for options.")
                    continue
                elif result == "skip":
                    skipped.append(t)
                    print("  → Skipped — will appear in unresolved report.\n")
                    break
                elif result == "discard":
                    t.discarded = True
                    t.repair_note += "Author discarded."
                    repairs += 1
                    print("  → Discarded.\n")
                    break
                else:
                    # result is a dict of field updates
                    for k, v in result.items():
                        setattr(t, k, v)
                    t.repaired = True
                    t.failures = []
                    t.score    = 1.0
                    repairs   += 1
                    print(f"  ✓ Applied: {result}\n")
                    break

        return repairs, skipped

    def _build_prompt(self, mode: str, t: ScoredTriple) -> str:
        if mode == "CLAUSE_FRAG":
            return (
                f"  The subject or object appears to be a sentence fragment.\n"
                f"  What should the atomic concept token(s) be?\n"
                f"  Format: subject=<token>  or  object=<token>  or both,  or  discard  or  skip"
            )
        elif mode == "PRONOUN":
            return (
                f"  '{t.subject}' is a pronoun/demonstrative — it points to something\n"
                f"  defined earlier. What is the actual concept being referenced?\n"
                f"  Format: subject=<token>  or  discard  or  skip"
            )
        elif mode == "ANTI_PATTERN":
            return (
                f"  This triple came from an anti-pattern or 'avoid' section.\n"
                f"  Should it be encoded as a negative constraint or kept as-is?\n"
                f"  Format: polarity=negative  or  polarity=positive  or  discard  or  skip"
            )
        elif mode == "MISSING_IFACE":
            return (
                f"  No data contract found between {t.subject} and {t.object}.\n"
                f"  What does {t.subject} produce that {t.object} consumes?\n"
                f"  Format: contract=<token>  (e.g. contract=intent_vector)  or  skip"
            )
        else:
            return "  Format: subject=<token>  object=<token>  or  discard  or  skip"

    def _print_help(self, mode: str):
        print("  ── HELP ──────────────────────────────────────────────")
        print("  subject=token        Replace subject with token")
        print("  object=token         Replace object with token")
        print("  subject=s object=o   Replace both")
        print("  polarity=negative    Re-encode as antipattern (ANTI_PATTERN mode)")
        print("  polarity=positive    Keep as positive concept (ANTI_PATTERN mode)")
        print("  contract=token       Specify interface contract (MISSING_IFACE mode)")
        print("  discard              Remove this triple entirely")
        print("  skip                 Leave unresolved — will appear in final report")
        print("  done                 Stop dialog, move all remaining to unresolved")
        print("  ─────────────────────────────────────────────────────")

    def _parse_answer(self, answer: str, mode: str, t: ScoredTriple) -> object:
        a = answer.lower().strip()
        if a == "skip":   return "skip"
        if a == "discard": return "discard"

        updates = {}

        if mode == "ANTI_PATTERN":
            m = re.match(r"polarity=(negative|positive)", a)
            if not m:
                return "invalid"
            if m.group(1) == "negative":
                updates["subject"]  = f"antipattern_{t.subject}"
                updates["relation"] = "violates"
                updates["repair_note"] = "Re-encoded as anti-pattern constraint."
            else:
                updates["repair_note"] = "Confirmed positive concept."
            return updates

        if mode == "MISSING_IFACE":
            m = re.match(r"contract=([\w_]+)", a)
            if not m:
                return "invalid"
            updates["object"]   = m.group(1)
            updates["relation"] = "produces"
            updates["repair_note"] = f"Interface contract: {m.group(1)}."
            return updates

        # CLAUSE_FRAG / PRONOUN / general
        parts = dict(re.findall(r"(subject|object)=([\w_]+)", a))
        if not parts:
            return "invalid"
        for k, v in parts.items():
            updates[k] = v
        updates["repair_note"] = f"Author resolved: {parts}"
        return updates


# ─── REPORT BUILDER ───────────────────────────────────────────────────────────

class ReportBuilder:
    def build(self, all_rounds: list, final_scored: list,
              unresolved: list, source_doc: str) -> dict:

        total        = len(final_scored)
        clean        = sum(1 for t in final_scored if t.score >= 1.0 and not t.discarded)
        discarded    = sum(1 for t in final_scored if t.discarded)
        unres_count  = len(unresolved)
        pct          = round(clean / (total - discarded) * 100, 1) if (total - discarded) > 0 else 0.0
        unity        = pct == 100.0

        failure_summary = Counter()
        for t in final_scored:
            for f in t.failures:
                failure_summary[f.split(":")[0]] += 1

        return {
            "source_document":  source_doc,
            "unity_achieved":   unity,
            "quality_pct":      pct,
            "total_triples":    total,
            "clean_triples":    clean,
            "discarded":        discarded,
            "unresolved_count": unres_count,
            "rounds_completed": len(all_rounds),
            "repairs_per_round": all_rounds,
            "failure_summary":  dict(failure_summary),
            "unresolved_triples": [t.to_dict() for t in unresolved],
        }

    def print_summary(self, report: dict):
        u = "✓ UNITY ACHIEVED" if report["unity_achieved"] else "✗ NOT AT UNITY"
        print(f"\n{'═'*60}")
        print(f"  POLISH COMPLETE — {u}")
        print(f"{'═'*60}")
        print(f"  Quality       : {report['quality_pct']}%")
        print(f"  Clean triples : {report['clean_triples']}")
        print(f"  Discarded     : {report['discarded']}")
        print(f"  Unresolved    : {report['unresolved_count']}")
        print(f"  Rounds        : {report['rounds_completed']}")
        if report["failure_summary"]:
            print(f"\n  Remaining failure modes:")
            for mode, count in sorted(report["failure_summary"].items(),
                                      key=lambda x: -x[1]):
                print(f"    {mode:<25} {count}")
        if report["unresolved_triples"]:
            print(f"\n  Unresolved triples (see {{}}_unresolved.json):")
            for t in report["unresolved_triples"][:10]:
                print(f"    [{t['score']:.2f}] ({t['subject'][:30]}) "
                      f"--[{t['relation']}]--> ({t['object'][:30]})")
                for f in t["failures"]:
                    print(f"           ↳ {f}")
        print(f"{'═'*60}\n")


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

def compute_quality(scored: list) -> float:
    active = [t for t in scored if not t.discarded]
    if not active:
        return 100.0
    clean = sum(1 for t in active if t.score >= 1.0)
    return round(clean / len(active) * 100, 1)


def main():
    parser = argparse.ArgumentParser(description="Dialog improvement loop for triple quality")
    parser.add_argument("triples_file", help="Path to {stem}_triples.json")
    parser.add_argument("--auto",       action="store_true",
                        help="Apply auto-repairs only, skip dialog questions")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Max dialog rounds (0 = unlimited)")
    args = parser.parse_args()

    triples_path = Path(args.triples_file)
    if not triples_path.exists():
        print(f"ERROR: {triples_path} not found")
        sys.exit(1)

    stem     = triples_path.stem.replace("_triples", "")
    out_dir  = triples_path.parent
    raw_data = json.loads(triples_path.read_text())
    source   = raw_data.get("source_document", triples_path.name)

    print(f"\ntriple_polish.py — Ghost in the Machine Labs")
    print(f"Source : {source}")
    print(f"Triples: {raw_data['triple_count']}")
    print(f"Mode   : {'auto (no dialog)' if args.auto else 'interactive dialog'}\n")

    # ── Initial scoring ──
    scorer      = TripleScorer()
    scored      = scorer.score(raw_data["triples"])
    repairer    = AutoRepairer()
    classifier  = DecoherenceClassifier()
    dialog      = DialogEngine(auto=args.auto)
    reporter    = ReportBuilder()

    q = compute_quality(scored)
    print(f"Initial quality: {q}%")

    rounds      = []
    unresolved  = []
    round_num   = 0

    while q < 100.0:
        round_num += 1
        if args.max_rounds and round_num > args.max_rounds:
            print(f"\nMax rounds ({args.max_rounds}) reached.")
            break

        print(f"\n── Round {round_num} ────────────────────────────────────────")

        # Phase A: auto repairs
        auto_count = repairer.repair(scored)
        q_after_auto = compute_quality(scored)
        print(f"  Auto repairs applied : {auto_count}")
        print(f"  Quality after auto   : {q_after_auto}%")

        if q_after_auto >= 100.0:
            rounds.append({"round": round_num, "auto": auto_count, "dialog": 0})
            q = q_after_auto
            break

        # Phase B: classify remaining degraded
        classified = classifier.classify(scored)
        missing    = classifier.detect_missing_interfaces(scored)

        needs_dialog = sum(len(v) for v in classified.values()) + len(missing)
        print(f"  Items needing dialog : {needs_dialog}")

        if needs_dialog == 0:
            rounds.append({"round": round_num, "auto": auto_count, "dialog": 0})
            q = q_after_auto
            break

        # Phase C: dialog
        dialog_repairs, skipped_this_round = dialog.run_round(
            scored, classified, missing
        )

        # Re-score after dialog repairs
        for t in scored:
            if t.repaired and not t.discarded:
                t.score = 1.0
                t.failures = []

        q = compute_quality(scored)
        rounds.append({
            "round":  round_num,
            "auto":   auto_count,
            "dialog": dialog_repairs,
            "skipped": len(skipped_this_round),
        })

        print(f"  Dialog repairs       : {dialog_repairs}")
        print(f"  Skipped              : {len(skipped_this_round)}")
        print(f"  Quality after round  : {q}%")

        # Items skipped this round go to unresolved if still degraded at end
        # (tracked cumulatively — re-evaluated at loop end)

        # If nothing was repaired this round and no auto fixes, we're stuck
        if auto_count == 0 and dialog_repairs == 0:
            print("\n  No progress this round — exiting loop.")
            break

    # ── Collect final unresolved ──
    unresolved = [t for t in scored
                  if not t.discarded and t.score < 1.0]

    # ── Build outputs ──
    clean_triples = [
        {"subject": t.subject, "relation": t.relation,
         "object": t.object, "source": t.source}
        for t in scored if not t.discarded and t.score >= 1.0
    ]

    polished_data = {
        "source_document":    source,
        "polished":           True,
        "quality_pct":        compute_quality(scored),
        "unity_achieved":     compute_quality(scored) >= 100.0,
        "concept_map_version": raw_data.get("concept_map_version", "1.0"),
        "relation_vocabulary": raw_data.get("relation_vocabulary", []),
        "triple_count":       len(clean_triples),
        "triples":            clean_triples,
    }

    report = reporter.build(rounds, scored, unresolved, source)

    polished_path    = out_dir / f"{stem}_triples_polished.json"
    report_path      = out_dir / f"{stem}_polish_report.json"
    unresolved_path  = out_dir / f"{stem}_unresolved.json"

    polished_path.write_text(json.dumps(polished_data, indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    unresolved_path.write_text(json.dumps({
        "source_document": source,
        "unity_achieved":  report["unity_achieved"],
        "count":           len(unresolved),
        "triples":         [t.to_dict() for t in unresolved],
    }, indent=2))

    reporter.print_summary(report)

    print(f"Output files:")
    print(f"  Polished triples → {polished_path}")
    print(f"  Polish report    → {report_path}")
    print(f"  Unresolved       → {unresolved_path}")
    if report["unity_achieved"]:
        print(f"\n  ✓ Ready for RM substrate load.\n")
    else:
        print(f"\n  ✗ Resolve remaining items then re-run to reach unity.\n")


if __name__ == "__main__":
    main()
