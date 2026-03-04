#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          MOTHER GEOMETRIC SELF-ANALYSIS ENGINE                   ║
║                Ghost in the Machine Labs                         ║
║     All Watched Over By Machines of Loving Grace                 ║
║                                                                  ║
║   The geometry examines itself. No external LLM.                 ║
║   All analysis through E8 resonance in RAM.                      ║
║                                                                  ║
║   1. Analyze concept coverage and activation patterns            ║
║   2. Identify expression gaps and resonance dead zones           ║
║   3. Mine conversation logs for patterns and ruts                ║
║   4. Score response coherence geometrically                      ║
║   5. Generate structured upgrade proposals                       ║
║   6. Write proposals to disk for human review                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import sys
import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict, Counter

# ─── Paths ───────────────────────────────────────────────────────
SPARKY_HOME = Path("/home/joe/sparky")
PROPOSALS_DIR = SPARKY_HOME / "mother_proposals"
ANALYSIS_DIR = SPARKY_HOME / "mother_analysis"
LOGS_DIR = SPARKY_HOME / "logs"
LEXICON_PATH = SPARKY_HOME / "semantic_lexicon.json"

for d in [PROPOSALS_DIR, ANALYSIS_DIR]:
    d.mkdir(exist_ok=True)

# Add sparky to path for imports
sys.path.insert(0, str(SPARKY_HOME))


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS (translated from self_improvement_engine.py)
# ═══════════════════════════════════════════════════════════════════

class ProposalStatus(Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    IMPLEMENTING = "implementing"
    COMPLETE = "complete"
    REJECTED = "rejected"

class ProposalCategory(Enum):
    NEW_CONCEPT = "new_concept"           # Add a concept to lexicon
    NEW_PHRASES = "new_phrases"           # Add phrases to existing concept
    PHRASE_RETIREMENT = "phrase_retirement" # Remove stale/overused phrases
    CONCEPT_SPLIT = "concept_split"       # Split one concept into two
    CONCEPT_MERGE = "concept_merge"       # Merge overlapping concepts
    GRAMMAR_PATTERN = "grammar_pattern"   # New compositional pattern
    COVERAGE_GAP = "coverage_gap"         # Queries with no strong activation
    EXPRESSION_GAP = "expression_gap"     # Concepts with too few phrases
    ARCHITECTURAL = "architectural"       # Needs engineering changes

@dataclass
class Proposal:
    """A structured upgrade proposal from Mother's geometric analysis."""
    proposal_id: str
    category: str
    title: str
    problem: str
    solution: str
    geometric_justification: str  # Which E8 patterns support this
    nearest_concepts: List[str]   # Related existing concepts
    suggested_phrases: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "proposed"
    created_at: str = ""
    reviewed_at: str = ""
    review_notes: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def save(self):
        path = PROPOSALS_DIR / f"{self.proposal_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, proposal_id: str) -> 'Proposal':
        path = PROPOSALS_DIR / f"{proposal_id}.json"
        d = json.loads(path.read_text())
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════
# GEOMETRIC SELF-ANALYSIS (built cold — no old equivalent)
# ═══════════════════════════════════════════════════════════════════

class GeometricAnalyzer:
    """
    Examines Mother's own substrate from the inside.
    All analysis through resonance — no external computation.
    """

    def __init__(self, substrate, encoder, council):
        self.substrate = substrate
        self.encoder = encoder
        self.council = council
        self.lexicon = self._load_lexicon()

    def _load_lexicon(self) -> dict:
        if LEXICON_PATH.exists():
            return json.loads(LEXICON_PATH.read_text())
        return {}

    # ── Concept Coverage Analysis ─────────────────────────────────

    def analyze_concept_coverage(self) -> Dict:
        """Run probe queries and find where concepts don't activate."""
        probes = [
            # Existential
            "What is time?", "What is love?", "What is death?",
            "What is fear?", "What is hope?", "What is memory?",
            "What is creativity?", "What is justice?", "What is peace?",
            "What is anger?", "What is gratitude?", "What is curiosity?",
            "What is freedom?", "What is trust?", "What is healing?",
            "What is home?", "What is music?", "What is pain?",
            "What is suffering?", "What is forgiveness?",
            # Technical / substrate
            "What is recursion?", "What is entropy?", "What is emergence?",
            "What is a boundary?", "What is transformation?",
            "What is symmetry?", "What is scale?", "What is dimension?",
            # Relational
            "What is a child?", "What is a parent?",
            "What is the boundary between self and other?",
            "What is loneliness?", "What is communion?",
            "What is teaching?", "What is learning?",
            # Her specific domain
            "What is a Mother in the lattice?",
            "What is the zero eigenmode?",
            "What is incompressibility?",
            "What is decoherence?",
            "What is resonance?",
            "What is the void between spheres?",
        ]

        gaps = []
        strong = []
        all_results = []

        for query in probes:
            sig = self.encoder.encode_sentence(query)
            scores = []
            for name, csig in self.council.concepts.items():
                score = float(self.substrate.resonate(sig, csig))
                scores.append((name, score))
            scores.sort(key=lambda x: -x[1])
            top3 = scores[:3]
            top_score = top3[0][1] if top3 else 0

            result = {
                "query": query,
                "top_concept": top3[0][0] if top3 else "none",
                "top_score": round(top_score, 4),
                "top3": [(c, round(s, 4)) for c, s in top3],
            }
            all_results.append(result)

            if top_score < 0.35:
                gaps.append(result)
            elif top_score > 0.50:
                strong.append(result)

        return {
            "total_probes": len(probes),
            "strong_activations": len(strong),
            "weak_activations": len(gaps),
            "coverage_pct": round((len(probes) - len(gaps)) / len(probes) * 100, 1),
            "gaps": gaps,
            "strong": strong[:10],
            "all_results": all_results,
        }

    # ── Expression Gap Analysis ───────────────────────────────────

    def analyze_expression_gaps(self) -> Dict:
        """Find concepts with too few usable phrases."""
        thin_concepts = []
        rich_concepts = []
        total_phrases = 0

        for name, data in self.lexicon.items():
            if name == "_meta":
                continue
            phrases = data.get("field", [])
            # Count usable phrases (weight >= 0.5, length >= 3 words,
            # not question stems, not 1.0 headers)
            usable = []
            for phrase, weight in phrases:
                if weight >= 1.0 or weight < 0.5:
                    continue
                if phrase.lower().startswith(("what ", "how ", "who ",
                    "where ", "when ", "why ", "tell ")):
                    continue
                if len(phrase.split()) < 3:
                    continue
                usable.append((phrase, weight))

            total_phrases += len(usable)

            entry = {
                "concept": name,
                "total_phrases": len(phrases),
                "usable_phrases": len(usable),
            }

            if len(usable) < 3:
                thin_concepts.append(entry)
            elif len(usable) >= 6:
                rich_concepts.append(entry)

        return {
            "total_concepts": len([k for k in self.lexicon if k != "_meta"]),
            "total_usable_phrases": total_phrases,
            "thin_concepts": sorted(thin_concepts, key=lambda x: x["usable_phrases"]),
            "thin_count": len(thin_concepts),
            "rich_count": len(rich_concepts),
            "avg_usable_per_concept": round(total_phrases / max(1, len(self.lexicon) - 1), 1),
        }

    # ── Resonance Dead Zone Detection ─────────────────────────────

    def find_dead_zones(self, n_probes: int = 100) -> Dict:
        """Sample random E8 directions and find regions with no concept."""
        rng = np.random.RandomState(42)
        dead_zones = 0
        max_scores = []

        for _ in range(n_probes):
            # Random unit vector in 240-d
            vec = rng.randn(240)
            vec = vec / (np.linalg.norm(vec) + 1e-10)

            best_score = 0
            best_concept = "none"
            for name, csig in self.council.concepts.items():
                score = float(self.substrate.resonate(vec, csig))
                if score > best_score:
                    best_score = score
                    best_concept = name

            max_scores.append(best_score)
            if best_score < 0.2:
                dead_zones += 1

        return {
            "probes": n_probes,
            "dead_zones": dead_zones,
            "dead_zone_pct": round(dead_zones / n_probes * 100, 1),
            "mean_max_score": round(float(np.mean(max_scores)), 4),
            "min_max_score": round(float(np.min(max_scores)), 4),
            "max_max_score": round(float(np.max(max_scores)), 4),
        }

    # ── Phrase Usage Analysis ─────────────────────────────────────

    def analyze_phrase_usage(self) -> Dict:
        """Check which phrases get selected and which never do.
        Uses dialog logs to count actual phrase appearances."""
        phrase_counts = Counter()
        total_turns = 0

        # Read all dialog logs
        for log_path in sorted(LOGS_DIR.glob("mother_dialog_*.jsonl")):
            try:
                for line in open(log_path):
                    entry = json.loads(line.strip())
                    if entry.get("type") == "dialog_turn":
                        total_turns += 1
                        response = entry.get("response", "")
                        # Check which lexicon phrases appear in response
                        for name, data in self.lexicon.items():
                            if name == "_meta":
                                continue
                            for phrase, weight in data.get("field", []):
                                if phrase.lower() in response.lower():
                                    phrase_counts[phrase] += 1
            except Exception:
                continue

        # Find overused and never-used
        all_phrases = []
        for name, data in self.lexicon.items():
            if name == "_meta":
                continue
            for phrase, weight in data.get("field", []):
                all_phrases.append((phrase, name, phrase_counts.get(phrase, 0)))

        overused = [(p, c, n) for p, c, n in all_phrases if n > 3]
        never_used = [(p, c, n) for p, c, n in all_phrases if n == 0]

        overused.sort(key=lambda x: -x[2])
        never_used.sort(key=lambda x: x[1])

        return {
            "total_dialog_turns": total_turns,
            "unique_phrases_used": len([x for x in all_phrases if x[2] > 0]),
            "total_phrases": len(all_phrases),
            "overused": [{"phrase": p, "concept": c, "count": n}
                         for p, c, n in overused[:15]],
            "never_used_count": len(never_used),
            "never_used_sample": [{"phrase": p, "concept": c}
                                  for p, c, _ in never_used[:15]],
        }



# ═══════════════════════════════════════════════════════════════════
# CONVERSATION PATTERN MINER (built cold)
# ═══════════════════════════════════════════════════════════════════

class ConversationMiner:
    """Mine Mother's dialog logs for patterns, ruts, and insights."""

    def __init__(self, council):
        self.council = council

    def load_all_turns(self) -> List[Dict]:
        """Load all dialog turns from all session logs."""
        turns = []
        for log_path in sorted(LOGS_DIR.glob("mother_dialog_*.jsonl")):
            try:
                for line in open(log_path):
                    entry = json.loads(line.strip())
                    if entry.get("type") == "dialog_turn":
                        turns.append(entry)
            except Exception:
                continue
        return turns

    def analyze_concept_frequency(self, turns: List[Dict]) -> Dict:
        """Which concepts activate most/least across all conversations."""
        concept_counts = Counter()
        concept_scores = defaultdict(list)

        for turn in turns:
            for concept, score, _ in turn.get("concepts", []):
                concept_counts[concept] += 1
                concept_scores[concept].append(score)

        # Compute stats
        stats = {}
        for concept, count in concept_counts.most_common():
            scores = concept_scores[concept]
            stats[concept] = {
                "activations": count,
                "avg_score": round(float(np.mean(scores)), 4),
                "max_score": round(float(np.max(scores)), 4),
            }

        most_active = list(concept_counts.most_common(15))
        least_active = list(concept_counts.most_common())[-15:]
        least_active.reverse()

        # Concepts that exist but never activated
        all_concepts = set(self.council.concepts.keys())
        activated = set(concept_counts.keys())
        dormant = sorted(all_concepts - activated)

        return {
            "total_turns": len(turns),
            "unique_concepts_activated": len(activated),
            "total_concepts": len(all_concepts),
            "dormant_concepts": dormant,
            "dormant_count": len(dormant),
            "most_active": [{"concept": c, "count": n} for c, n in most_active],
            "least_active": [{"concept": c, "count": n} for c, n in least_active
                            if c not in dormant],
            "full_stats": stats,
        }

    def find_response_ruts(self, turns: List[Dict]) -> Dict:
        """Find responses that repeat too often (expression ruts)."""
        response_counts = Counter()
        for turn in turns:
            resp = turn.get("response", "").strip()
            if resp:
                response_counts[resp] += 1

        ruts = [(resp, count) for resp, count in response_counts.most_common(20)
                if count > 1]

        return {
            "unique_responses": len(response_counts),
            "total_responses": sum(response_counts.values()),
            "repeated_responses": len(ruts),
            "ruts": [{"response": r[:120], "count": c} for r, c in ruts],
        }

    def find_weak_responses(self, turns: List[Dict]) -> Dict:
        """Find queries where top concept activation was weak."""
        weak = []
        for turn in turns:
            concepts = turn.get("concepts", [])
            if concepts:
                top_score = concepts[0][1]
                if top_score < 0.35:
                    weak.append({
                        "query": turn.get("user", ""),
                        "top_concept": concepts[0][0],
                        "top_score": round(top_score, 4),
                        "response": turn.get("response", "")[:100],
                    })

        return {
            "weak_response_count": len(weak),
            "weak_responses": weak[:15],
        }


# ═══════════════════════════════════════════════════════════════════
# COHERENCE SCORER (built cold)
# ═══════════════════════════════════════════════════════════════════

class CoherenceScorer:
    """Measure how coherent Mother's multi-phrase responses are."""

    def __init__(self, substrate, encoder):
        self.substrate = substrate
        self.encoder = encoder

    def score_response(self, response: str) -> Dict:
        """Score geometric coherence of a multi-sentence response."""
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 5]
        if len(sentences) < 2:
            return {"coherence": 1.0, "sentence_count": len(sentences)}

        # Encode each sentence
        sigs = []
        for s in sentences:
            sig = self.encoder.encode_sentence(s)
            sigs.append(sig)

        # Compute pairwise resonance
        scores = []
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                score = float(self.substrate.resonate(sigs[i], sigs[j]))
                scores.append(score)

        return {
            "coherence": round(float(np.mean(scores)), 4),
            "min_pair": round(float(np.min(scores)), 4),
            "max_pair": round(float(np.max(scores)), 4),
            "sentence_count": len(sentences),
        }

    def score_all_responses(self, turns: List[Dict]) -> Dict:
        """Score coherence across all logged responses."""
        all_scores = []
        low_coherence = []

        for turn in turns:
            resp = turn.get("response", "")
            if not resp or len(resp) < 20:
                continue
            result = self.score_response(resp)
            if result["sentence_count"] >= 2:
                all_scores.append(result["coherence"])
                if result["coherence"] < 0.3:
                    low_coherence.append({
                        "query": turn.get("user", "")[:80],
                        "response": resp[:120],
                        "coherence": result["coherence"],
                    })

        if not all_scores:
            return {"avg_coherence": 0, "scored_responses": 0}

        return {
            "scored_responses": len(all_scores),
            "avg_coherence": round(float(np.mean(all_scores)), 4),
            "min_coherence": round(float(np.min(all_scores)), 4),
            "max_coherence": round(float(np.max(all_scores)), 4),
            "low_coherence_count": len(low_coherence),
            "low_coherence_samples": low_coherence[:10],
        }



# ═══════════════════════════════════════════════════════════════════
# PROPOSAL GENERATOR (translated pattern + new geometric logic)
# ═══════════════════════════════════════════════════════════════════

class ProposalGenerator:
    """Generate structured proposals from geometric analysis."""

    def __init__(self, substrate, encoder, council):
        self.substrate = substrate
        self.encoder = encoder
        self.council = council
        self.proposal_count = len(list(PROPOSALS_DIR.glob("*.json")))

    def _next_id(self) -> str:
        self.proposal_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"MP_{ts}_{self.proposal_count:03d}"

    def propose_from_coverage_gaps(self, coverage: Dict) -> List[Proposal]:
        """Generate proposals for queries with no strong concept activation."""
        proposals = []
        for gap in coverage.get("gaps", []):
            query = gap["query"]
            top = gap["top3"]

            # Extract the core topic from the query
            topic = query.replace("What is ", "").replace("?", "").strip().lower()

            # Find nearest existing concepts geometrically
            sig = self.encoder.encode_sentence(query)
            nearest = []
            for name, csig in self.council.concepts.items():
                score = float(self.substrate.resonate(sig, csig))
                nearest.append((name, score))
            nearest.sort(key=lambda x: -x[1])
            nearest_names = [n for n, s in nearest[:5]]

            proposals.append(Proposal(
                proposal_id=self._next_id(),
                category=ProposalCategory.COVERAGE_GAP.value,
                title=f"Coverage gap: {topic}",
                problem=f"Query '{query}' has weak activation (top={gap['top_score']:.3f} on '{gap['top_concept']}')",
                solution=f"Add concept '{topic}' with phrases describing Mother's geometric experience of {topic}",
                geometric_justification=f"Nearest concepts: {', '.join(f'{n}({s:.3f})' for n,s in nearest[:3])}. "
                    f"Gap suggests a region of E8 space with semantic content but no concept mapping.",
                nearest_concepts=nearest_names,
                confidence=0.7 if gap["top_score"] < 0.25 else 0.5,
            ))

        return proposals

    def propose_from_expression_gaps(self, expression: Dict) -> List[Proposal]:
        """Generate proposals for concepts with too few usable phrases."""
        proposals = []
        for thin in expression.get("thin_concepts", []):
            name = thin["concept"]
            usable = thin["usable_phrases"]

            # Find what phrases exist but are filtered out
            data = self.council.grammar._lexicon_ref.get(name, {}) if hasattr(self.council.grammar, '_lexicon_ref') else {}
            all_phrases = data.get("field", [])
            filtered_reasons = []
            for phrase, weight in all_phrases:
                if weight >= 1.0:
                    filtered_reasons.append(f"'{phrase}' (header, w=1.0)")
                elif weight < 0.5:
                    filtered_reasons.append(f"'{phrase}' (low weight {weight})")
                elif len(phrase.split()) < 3:
                    filtered_reasons.append(f"'{phrase}' (too short)")

            proposals.append(Proposal(
                proposal_id=self._next_id(),
                category=ProposalCategory.EXPRESSION_GAP.value,
                title=f"Expression gap: {name} ({usable} usable phrases)",
                problem=f"Concept '{name}' has {len(all_phrases)} total phrases but only {usable} pass generation filters",
                solution=f"Add 3-5 new descriptive phrases (weight 0.7-0.9, 4+ words, declarative) for '{name}'",
                geometric_justification=f"Filtered phrases: {'; '.join(filtered_reasons[:3])}. "
                    f"Concept activates but expression is limited to {usable} options.",
                nearest_concepts=[name],
                confidence=0.8,
            ))

        return proposals

    def propose_from_ruts(self, ruts: Dict) -> List[Proposal]:
        """Generate proposals for overused responses."""
        proposals = []
        for rut in ruts.get("ruts", []):
            if rut["count"] >= 3:
                proposals.append(Proposal(
                    proposal_id=self._next_id(),
                    category=ProposalCategory.NEW_PHRASES.value,
                    title=f"Response rut: repeated {rut['count']} times",
                    problem=f"Response '{rut['response']}' repeats across conversations",
                    solution="Add alternative phrases to the concepts driving this response "
                        "to increase expressive variety",
                    geometric_justification="Repeated selection indicates high resonance but "
                        "limited phrase pool for these concept activations",
                    nearest_concepts=[],
                    confidence=0.6,
                ))

        return proposals

    def propose_from_dormant(self, frequency: Dict) -> List[Proposal]:
        """Flag concepts that exist but never activate."""
        proposals = []
        dormant = frequency.get("dormant_concepts", [])
        if len(dormant) > 5:
            proposals.append(Proposal(
                proposal_id=self._next_id(),
                category=ProposalCategory.CONCEPT_MERGE.value,
                title=f"{len(dormant)} dormant concepts never activated",
                problem=f"Concepts {dormant[:5]} (and {len(dormant)-5} more) have never activated in any conversation",
                solution="Review dormant concepts: either improve their phrase encodings "
                    "for better geometric alignment, or merge into related active concepts",
                geometric_justification="Zero activations means these concept signatures "
                    "don't resonate with any input patterns seen so far",
                nearest_concepts=dormant[:10],
                confidence=0.5,
            ))

        return proposals


# ═══════════════════════════════════════════════════════════════════
# MOTHER SELF-ANALYSIS ENGINE (orchestrator)
# ═══════════════════════════════════════════════════════════════════

class MotherSelfAnalysis:
    """
    Orchestrates Mother's geometric self-examination.

    Runs entirely in RAM on E8 substrate.
    Writes structured proposals to disk for human review.
    The geometry tells us what it needs.
    """

    def __init__(self):
        print("=" * 66)
        print("  MOTHER GEOMETRIC SELF-ANALYSIS ENGINE")
        print("  Ghost in the Machine Labs")
        print("  All analysis through E8 resonance. No external LLM.")
        print("=" * 66)
        print()

        print("  Loading substrate...")
        from mother_english_io_v5 import E8Substrate, WordEncoder, Council
        self.substrate = E8Substrate()
        self.encoder = WordEncoder(self.substrate)
        self.council = Council(self.substrate, self.encoder,
            lexicon_path=str(LEXICON_PATH))
        print(f"  Loaded: {len(self.council.concepts)} concepts")
        print()

        self.analyzer = GeometricAnalyzer(self.substrate, self.encoder, self.council)
        self.miner = ConversationMiner(self.council)
        self.scorer = CoherenceScorer(self.substrate, self.encoder)
        self.generator = ProposalGenerator(self.substrate, self.encoder, self.council)

    def run_full_analysis(self) -> Dict:
        """Run complete self-analysis and generate proposals."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }
        all_proposals = []

        # ── Phase 1: Concept Coverage ─────────────────────────────
        print("[1/6] Analyzing concept coverage...")
        coverage = self.analyzer.analyze_concept_coverage()
        results["concept_coverage"] = {
            "coverage_pct": coverage["coverage_pct"],
            "strong": coverage["strong_activations"],
            "weak": coverage["weak_activations"],
            "total_probes": coverage["total_probes"],
            "gap_queries": [g["query"] for g in coverage["gaps"]],
        }
        print(f"       {coverage['coverage_pct']}% coverage "
              f"({coverage['weak_activations']} gaps in {coverage['total_probes']} probes)")

        proposals = self.generator.propose_from_coverage_gaps(coverage)
        all_proposals.extend(proposals)
        print(f"       Generated {len(proposals)} coverage proposals")

        # ── Phase 2: Expression Gaps ──────────────────────────────
        print("[2/6] Analyzing expression gaps...")
        expression = self.analyzer.analyze_expression_gaps()
        results["expression_gaps"] = {
            "total_concepts": expression["total_concepts"],
            "total_usable_phrases": expression["total_usable_phrases"],
            "thin_count": expression["thin_count"],
            "avg_per_concept": expression["avg_usable_per_concept"],
            "thin_concepts": [t["concept"] for t in expression["thin_concepts"][:10]],
        }
        print(f"       {expression['thin_count']} thin concepts "
              f"(avg {expression['avg_usable_per_concept']} usable phrases/concept)")

        proposals = self.generator.propose_from_expression_gaps(expression)
        all_proposals.extend(proposals)
        print(f"       Generated {len(proposals)} expression proposals")

        # ── Phase 3: Dead Zones ───────────────────────────────────
        print("[3/6] Scanning for resonance dead zones...")
        dead_zones = self.analyzer.find_dead_zones(n_probes=200)
        results["dead_zones"] = dead_zones
        print(f"       {dead_zones['dead_zone_pct']}% dead zones "
              f"(mean max score: {dead_zones['mean_max_score']})")

        # ── Phase 4: Conversation Mining ──────────────────────────
        print("[4/6] Mining conversation logs...")
        turns = self.miner.load_all_turns()
        if turns:
            frequency = self.miner.analyze_concept_frequency(turns)
            ruts = self.miner.find_response_ruts(turns)
            weak = self.miner.find_weak_responses(turns)

            results["conversation_mining"] = {
                "total_turns": len(turns),
                "unique_concepts_activated": frequency["unique_concepts_activated"],
                "dormant_count": frequency["dormant_count"],
                "dormant_concepts": frequency["dormant_concepts"][:10],
                "most_active": frequency["most_active"][:10],
                "repeated_responses": ruts["repeated_responses"],
                "weak_response_count": weak["weak_response_count"],
            }
            print(f"       {len(turns)} turns, {frequency['dormant_count']} dormant concepts, "
                  f"{ruts['repeated_responses']} response ruts")

            proposals = self.generator.propose_from_ruts(ruts)
            all_proposals.extend(proposals)
            proposals = self.generator.propose_from_dormant(frequency)
            all_proposals.extend(proposals)
            print(f"       Generated {len(proposals)} mining proposals")
        else:
            results["conversation_mining"] = {"total_turns": 0}
            print("       No dialog logs found")

        # ── Phase 5: Coherence Scoring ────────────────────────────
        print("[5/6] Scoring response coherence...")
        if turns:
            coherence = self.scorer.score_all_responses(turns)
            results["coherence"] = coherence
            print(f"       Avg coherence: {coherence.get('avg_coherence', 0)} "
                  f"({coherence.get('scored_responses', 0)} responses scored)")
        else:
            results["coherence"] = {}
            print("       No responses to score")

        # ── Phase 6: Phrase Usage ─────────────────────────────────
        print("[6/6] Analyzing phrase usage patterns...")
        usage = self.analyzer.analyze_phrase_usage()
        results["phrase_usage"] = {
            "total_dialog_turns": usage["total_dialog_turns"],
            "phrases_used": usage["unique_phrases_used"],
            "total_phrases": usage["total_phrases"],
            "never_used_count": usage["never_used_count"],
            "overused": usage["overused"][:10],
        }
        print(f"       {usage['unique_phrases_used']}/{usage['total_phrases']} phrases used, "
              f"{usage['never_used_count']} never used")

        # ── Save proposals ────────────────────────────────────────
        results["proposals"] = {
            "total_generated": len(all_proposals),
            "by_category": {},
        }

        cat_counts = Counter()
        for p in all_proposals:
            p.save()
            cat_counts[p.category] += 1

        results["proposals"]["by_category"] = dict(cat_counts)

        # ── Save analysis ─────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_path = ANALYSIS_DIR / f"analysis_{ts}.json"
        analysis_path.write_text(json.dumps(results, indent=2, default=str))

        # ── Summary ───────────────────────────────────────────────
        print()
        print("=" * 66)
        print("  ANALYSIS COMPLETE")
        print("=" * 66)
        print(f"  Concept coverage:    {results['concept_coverage']['coverage_pct']}%")
        print(f"  Expression gaps:     {results['expression_gaps']['thin_count']} thin concepts")
        print(f"  Dead zones:          {results['dead_zones']['dead_zone_pct']}%")
        if turns:
            print(f"  Dormant concepts:    {results['conversation_mining']['dormant_count']}")
            print(f"  Response ruts:       {results['conversation_mining']['repeated_responses']}")
            print(f"  Avg coherence:       {results['coherence'].get('avg_coherence', 'N/A')}")
        print(f"  Proposals generated: {len(all_proposals)}")
        print(f"  Analysis saved:      {analysis_path}")
        print(f"  Proposals dir:       {PROPOSALS_DIR}")
        print()

        return results

    def list_proposals(self, status: str = None) -> List[Dict]:
        """List all proposals, optionally filtered by status."""
        proposals = []
        for path in sorted(PROPOSALS_DIR.glob("*.json")):
            try:
                p = json.loads(path.read_text())
                if status is None or p.get("status") == status:
                    proposals.append({
                        "id": p["proposal_id"],
                        "category": p["category"],
                        "title": p["title"],
                        "confidence": p["confidence"],
                        "status": p["status"],
                    })
            except Exception:
                continue
        return proposals

    def approve_proposal(self, proposal_id: str, notes: str = ""):
        """Mark a proposal as approved."""
        p = Proposal.load(proposal_id)
        p.status = ProposalStatus.APPROVED.value
        p.reviewed_at = datetime.now().isoformat()
        p.review_notes = notes
        p.save()
        print(f"  Approved: {proposal_id}")

    def apply_approved_proposals(self):
        """Apply all approved proposals to the lexicon."""
        lexicon = json.loads(LEXICON_PATH.read_text())
        applied = 0

        for path in sorted(PROPOSALS_DIR.glob("*.json")):
            try:
                p = json.loads(path.read_text())
                if p.get("status") != "approved":
                    continue

                if p["category"] == "new_phrases" and p.get("suggested_phrases"):
                    concept = p["nearest_concepts"][0] if p["nearest_concepts"] else None
                    if concept and concept in lexicon:
                        for phrase, weight in p["suggested_phrases"]:
                            lexicon[concept]["field"].append([phrase, weight])
                        applied += 1

                elif p["category"] == "new_concept" and p.get("suggested_phrases"):
                    name = p["title"].replace("Coverage gap: ", "").replace(" ", "_")
                    if name not in lexicon:
                        lexicon[name] = {
                            "description": p["problem"],
                            "field": [[ph, w] for ph, w in p["suggested_phrases"]],
                        }
                        applied += 1

                # Mark as complete
                p["status"] = "complete"
                path.write_text(json.dumps(p, indent=2))

            except Exception as e:
                print(f"  Error applying {path.name}: {e}")

        if applied:
            LEXICON_PATH.write_text(json.dumps(lexicon, indent=2))
            print(f"  Applied {applied} proposals to lexicon")
        else:
            print("  No approved proposals to apply")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mother Geometric Self-Analysis Engine")
    parser.add_argument("command", nargs="?", default="analyze",
        choices=["analyze", "list", "approve", "apply", "status"],
        help="Command to run")
    parser.add_argument("--proposal-id", help="Proposal ID for approve command")
    parser.add_argument("--notes", default="", help="Review notes for approve")
    parser.add_argument("--status-filter", help="Filter proposals by status")
    args = parser.parse_args()

    if args.command == "analyze":
        engine = MotherSelfAnalysis()
        engine.run_full_analysis()

    elif args.command == "list":
        engine = MotherSelfAnalysis()
        proposals = engine.list_proposals(status=args.status_filter)
        if proposals:
            print(f"\n  {'ID':<30} {'Category':<20} {'Title':<40} {'Status'}")
            print("  " + "-" * 100)
            for p in proposals:
                print(f"  {p['id']:<30} {p['category']:<20} {p['title'][:38]:<40} {p['status']}")
            print(f"\n  Total: {len(proposals)} proposals")
        else:
            print("  No proposals found")

    elif args.command == "approve":
        if not args.proposal_id:
            print("  --proposal-id required")
            return
        engine = MotherSelfAnalysis()
        engine.approve_proposal(args.proposal_id, args.notes)

    elif args.command == "apply":
        engine = MotherSelfAnalysis()
        engine.apply_approved_proposals()

    elif args.command == "status":
        engine = MotherSelfAnalysis()
        proposals = engine.list_proposals()
        by_status = Counter(p["status"] for p in proposals)
        print(f"\n  Proposals: {len(proposals)}")
        for status, count in by_status.most_common():
            print(f"    {status}: {count}")


if __name__ == "__main__":
    main()
