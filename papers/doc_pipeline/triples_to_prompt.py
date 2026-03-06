#!/usr/bin/env python3
"""
triples_to_prompt.py
====================
Ghost in the Machine Labs

Synthesize structured prompts from polished concept triple sets.

Takes a unity-quality triple set (output of triple_polish.py) and a target
task token, walks the concept graph outward from that token, and assembles
a structured prompt in one of three modes.

Usage:
  python3 triples_to_prompt.py <triples.json> <target_token> [options]

  target_token   Concept token to build prompt around.
                 Use --list to see all available tokens.

Options:
  --mode rm          RM substrate query (geometric, terse)
  --mode llm         LLM system prompt (full natural language)
  --mode task        Ordered task spec with I/O contracts  [default]
  --depth N          Graph traversal depth from target (default: 2)
  --list             List all available tokens in the triple set
  --out FILE         Write prompt to file (default: stdout)
  --all-tokens       Generate a prompt for every top-level token (batch mode)

Output modes:
  rm    — Terse structured block for RM substrate ingestion.
          No prose. Geometric relationships only.

  llm   — Full natural-language system prompt with:
            context block, task decomposition, I/O contracts,
            constraints, antipatterns (negative examples),
            and success criteria.

  task  — Ordered step list. Each step has: name, input contract,
            output contract, constraints, known failure modes.
            Format suitable for handing to any agent or engineer.

Pipeline position:
  doc_to_triples.py → triple_polish.py → [triples_to_prompt.py] → agent
"""

import json
import sys
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional


# ─── RELATION SEMANTICS ───────────────────────────────────────────────────────
# Maps relation types to their role in prompt construction

CONTEXT_RELATIONS  = {"is_type", "defined_by", "has_property", "maps_to"}
STRUCTURE_RELATIONS = {"has_component", "implements", "extends"}
FLOW_RELATIONS     = {"produces", "consumes", "transforms_to", "returns"}
ORDER_RELATIONS    = {"precedes", "follows", "depends_on"}
BOUND_RELATIONS    = {"has_constraint", "validated_by"}
NEGATIVE_RELATIONS = {"violates"}
ACTION_RELATIONS   = {"executes", "calls", "validates", "stores", "loads"}

ALL_RELATIONS = (CONTEXT_RELATIONS | STRUCTURE_RELATIONS | FLOW_RELATIONS |
                 ORDER_RELATIONS | BOUND_RELATIONS | NEGATIVE_RELATIONS |
                 ACTION_RELATIONS)


# ─── GRAPH ────────────────────────────────────────────────────────────────────

class ConceptGraph:
    def __init__(self, triples: list):
        self.triples = triples
        self.by_subject  = defaultdict(list)
        self.by_object   = defaultdict(list)
        self.all_tokens  = set()

        for t in triples:
            self.by_subject[t["subject"]].append(t)
            self.by_object[t["object"]].append(t)
            self.all_tokens.add(t["subject"])
            self.all_tokens.add(t["object"])

    def neighbors(self, token: str, depth: int = 2,
                  relations: Optional[set] = None) -> dict:
        """
        BFS from token up to depth hops.
        Returns {token: [triples_touching_token]} for all reachable nodes.
        """
        visited  = {}
        queue    = deque([(token, 0)])
        seen     = {token}

        while queue:
            node, d = queue.popleft()
            node_triples = []

            # Outbound edges
            for t in self.by_subject.get(node, []):
                if relations is None or t["relation"] in relations:
                    node_triples.append(t)
                    if d < depth and t["object"] not in seen:
                        seen.add(t["object"])
                        queue.append((t["object"], d + 1))

            # Inbound edges (subject pointing to this node)
            for t in self.by_object.get(node, []):
                if relations is None or t["relation"] in relations:
                    node_triples.append(t)
                    if d < depth and t["subject"] not in seen:
                        seen.add(t["subject"])
                        queue.append((t["subject"], d + 1))

            if node_triples:
                visited[node] = node_triples

        return visited

    def get_by_relation(self, token: str, relation: str,
                        as_subject: bool = True) -> list:
        """Get all triples where token is subject (or object) with given relation."""
        if as_subject:
            return [t for t in self.by_subject.get(token, [])
                    if t["relation"] == relation]
        else:
            return [t for t in self.by_object.get(token, [])
                    if t["relation"] == relation]

    def get_components(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "has_component")]

    def get_produces(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "produces")]

    def get_consumes(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "consumes")]

    def get_constraints(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "has_constraint")]

    def get_dependencies(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "depends_on")]

    def get_antipatterns(self) -> list:
        return [t for t in self.triples if t["relation"] == "violates"]

    def get_validated_by(self, token: str) -> list:
        return [t["object"] for t in self.get_by_relation(token, "validated_by")]

    def get_type(self, token: str) -> Optional[str]:
        items = self.get_by_relation(token, "is_type")
        return items[0]["object"] if items else None

    def get_definition(self, token: str) -> Optional[str]:
        items = self.get_by_relation(token, "defined_by")
        return items[0]["object"] if items else None

    def topological_order(self, tokens: list) -> list:
        """Sort tokens by precedes/depends_on relationships."""
        order_map = {}
        for t in self.triples:
            if t["relation"] in ("precedes", "depends_on"):
                if t["subject"] in tokens and t["object"] in tokens:
                    order_map[t["subject"]] = order_map.get(t["subject"], 0)
                    order_map[t["object"]] = max(
                        order_map.get(t["object"], 0),
                        order_map.get(t["subject"], 0) + 1
                    )
        return sorted(tokens, key=lambda x: order_map.get(x, 0))

    def find_token(self, query: str) -> Optional[str]:
        """Fuzzy-find a token by substring."""
        q = query.lower().replace("-", "_").replace(" ", "_")
        if q in self.all_tokens:
            return q
        matches = [t for t in self.all_tokens if q in t]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer exact prefix
            prefix = [m for m in matches if m.startswith(q)]
            if prefix:
                return sorted(prefix, key=len)[0]
            return sorted(matches, key=len)[0]
        return None


# ─── FORMATTERS ───────────────────────────────────────────────────────────────

def humanize(token: str) -> str:
    """Convert snake_case token to readable label."""
    return token.replace("_", " ").strip()


def humanize_defn(token: str) -> str:
    """
    Humanize a definition token that may be a prose shard.
    Returns a keyword-style hint rather than broken prose.
    e.g. 'all_active_state_field' → '[all active state, field]'
    """
    words = token.replace("_", " ").strip().split()
    # If it reads like a sentence fragment (starts with verb/article), wrap as keywords
    starters = {"all","every","the","a","an","is","are","produces","extracts",
                "returns","maps","stores","test","close","add","at","phone",
                "can","direct","consumer","sparky","distribution","hour",
                "not","always","pgrep","consciousness","seat","nvidia","amd"}
    if words and words[0].lower() in starters:
        return "[" + ", ".join(words) + "]"
    return " ".join(words)


def is_prose_shard(token: str) -> bool:
    """True if token looks like a truncated prose fragment rather than atomic concept."""
    words = token.split("_")
    starters = {"all","every","the","a","an","is","are","produces","extracts",
                "returns","maps","stores","test","close","add","at","phone",
                "can","direct","consumer","sparky","distribution","hour",
                "not","always","pgrep","consciousness","seat","nvidia","amd",
                "proving","developing","etl","requestreply","timing","human",
                "authentication","self","state","strict","sequential","message",
                "readwrite","plugin","feedback","derived","defined","input",
                "named","lattice","bounds","result","user","read","record",
                "latency","real","depends","single","identify","map","surface",
                "per","answer","drives","resolved","hub","core","bifurcated",
                "distributed","producer"}
    return bool(words) and words[0].lower() in starters


def section(title: str, width: int = 60) -> str:
    return f"\n{'─' * width}\n{title.upper()}\n{'─' * width}"


def bullet(text: str, indent: int = 2) -> str:
    prefix = " " * indent + "• "
    wrap_width = 76 - indent
    lines = textwrap.wrap(text, wrap_width)
    if not lines:
        return ""
    result = prefix + lines[0]
    for line in lines[1:]:
        result += "\n" + " " * (indent + 2) + line
    return result


def code_block(lines: list) -> str:
    return "\n".join(f"  {line}" for line in lines)


# ─── MODE: RM QUERY ───────────────────────────────────────────────────────────

def render_rm(graph: ConceptGraph, token: str, depth: int) -> str:
    """
    Terse geometric query format for RM substrate.
    No prose. Structural relationships and contracts only.
    Format: TOKEN :: RELATION :: TARGET
    """
    neighborhood = graph.neighbors(token, depth=depth)
    lines = [
        f"TARGET: {token}",
        f"DEPTH:  {depth}",
        "",
    ]

    # Group by relation type
    by_rel = defaultdict(list)
    for node, node_triples in neighborhood.items():
        for t in node_triples:
            by_rel[t["relation"]].append(t)

    # Emit in semantic order
    priority = [
        ("is_type",        "TYPE"),
        ("has_component",  "COMPONENTS"),
        ("depends_on",     "DEPENDS"),
        ("produces",       "PRODUCES"),
        ("consumes",       "CONSUMES"),
        ("has_constraint", "CONSTRAINTS"),
        ("validates",      "VALIDATES"),
        ("validated_by",   "EVIDENCE"),
        ("executes",       "EXECUTES"),
        ("violates",       "ANTIPATTERN"),
        ("precedes",       "PRECEDES"),
        ("defined_by",     "DEF"),
    ]

    # Hub exclusion: skip tokens with > 20 outbound edges (e.g. 'contents')
    hub_threshold = 20
    hub_tokens = {tok for tok, items in graph.by_subject.items()
                  if len(items) > hub_threshold}

    seen_triples = set()
    for rel, label in priority:
        triples = by_rel.get(rel, [])
        if not triples:
            continue
        for t in triples:
            key = (t["subject"], t["relation"], t["object"])
            if key in seen_triples:
                continue
            if t["subject"] in hub_tokens or t["object"] in hub_tokens:
                continue
            seen_triples.add(key)
            lines.append(f"{t['subject']:<40} :: {label:<12} :: {t['object']}")

    lines.append("")
    lines.append(f"TRIPLES: {len(seen_triples)}")
    return "\n".join(lines)


# ─── MODE: LLM SYSTEM PROMPT ─────────────────────────────────────────────────

def render_llm(graph: ConceptGraph, token: str, depth: int) -> str:
    """
    Full natural-language system prompt with context, task, contracts,
    constraints, antipatterns, and success criteria.
    """
    parts = []

    # ── Header ──
    parts.append(f"# System Prompt: {humanize(token).title()}")
    parts.append(f"# Generated from concept graph — Ghost in the Machine Labs")
    parts.append("")

    # ── Role / Context ──
    type_of = graph.get_type(token)
    defn    = graph.get_definition(token)

    parts.append("## ROLE AND CONTEXT")
    parts.append("")
    if type_of:
        parts.append(f"You are operating as a {humanize(type_of)} within the Ghost in the Machine Labs "
                     f"E8 geometric architecture.")
    else:
        parts.append(f"You are operating within the Ghost in the Machine Labs E8 geometric architecture.")

    if defn:
        parts.append(f"The focus of this session is: {humanize(defn)}.")

    # Components = sub-tasks this token owns
    components = graph.get_components(token)
    if components:
        parts.append(f"\nThis task encompasses the following components:")
        for c in components:
            c_defn = graph.get_definition(c)
            if c_defn and not is_prose_shard(c_defn):
                parts.append(bullet(f"{humanize(c)}: {humanize(c_defn)}"))
            elif c_defn:
                parts.append(bullet(f"{humanize(c)} {humanize_defn(c_defn)}"))
            else:
                parts.append(bullet(humanize(c)))

    parts.append("")

    # ── Input contract ──
    consumes = graph.get_consumes(token)
    deps     = graph.get_dependencies(token)
    if consumes or deps:
        parts.append("## INPUT CONTRACT")
        parts.append("")
        if consumes:
            parts.append("You will receive the following inputs:")
            for c in consumes:
                parts.append(bullet(humanize(c)))
        if deps:
            parts.append("The following must be completed before this task begins:")
            for d in deps:
                parts.append(bullet(humanize(d)))
        parts.append("")

    # ── Output contract ──
    produces = graph.get_produces(token)
    if produces:
        parts.append("## OUTPUT CONTRACT")
        parts.append("")
        parts.append("Your output must include:")
        for p in produces:
            parts.append(bullet(humanize(p)))
        parts.append("")

    # ── Constraints ──
    constraints = graph.get_constraints(token)
    # Also pull constraints from components
    for c in components:
        constraints.extend(graph.get_constraints(c))
    constraints = list(dict.fromkeys(constraints))  # deduplicate

    if constraints:
        parts.append("## CONSTRAINTS")
        parts.append("")
        for c in constraints:
            parts.append(bullet(humanize(c)))
        parts.append("")

    # ── Evidence / validation ──
    evidence = graph.get_validated_by(token)
    # Pull from neighborhood
    neighborhood = graph.neighbors(token, depth=depth)
    all_evidence = []
    for node, node_triples in neighborhood.items():
        for t in node_triples:
            if t["relation"] == "validated_by" and t["subject"] in neighborhood:
                all_evidence.append((t["subject"], t["object"]))
    if all_evidence or evidence:
        parts.append("## VALIDATED PRINCIPLES")
        parts.append("")
        parts.append("The following have been empirically validated and must be respected:")
        for subj, obj in all_evidence[:6]:
            parts.append(bullet(f"{humanize(subj)} — validated by {humanize(obj)}"))
        parts.append("")

    # ── Antipatterns ──
    antipatterns = graph.get_antipatterns()
    if antipatterns:
        parts.append("## ANTIPATTERNS — DO NOT DO THESE")
        parts.append("")
        parts.append("The following are known failure modes. Avoid them explicitly:")
        for t in antipatterns:
            name = t["subject"].replace("antipattern_", "")
            # Expand truncated antipattern names from defined_by if available
            expanded = graph.get_by_relation(t["subject"].replace("antipattern_",""), "defined_by")
            if not expanded:
                # Try looking up the original long-form token
                orig = [x for x in graph.by_subject 
                        if x.startswith(name.split("_")[0]) and len(x) > len(name)]
                expanded_name = humanize(orig[0]) if orig else humanize(name)
            else:
                expanded_name = humanize(name)
            parts.append(bullet(f"Do NOT {expanded_name.lower()} — violates {humanize(t['object'])}"))
        parts.append("")

    # ── Success criteria ──
    parts.append("## SUCCESS CRITERIA")
    parts.append("")
    parts.append("This task is complete when:")
    if produces:
        for p in produces:
            parts.append(bullet(f"Output '{humanize(p)}' has been produced and validated"))
    constraints_brief = constraints[:3]
    for c in constraints_brief:
        parts.append(bullet(f"Constraint satisfied: {humanize(c)}"))
    parts.append(bullet("All antipatterns above have been explicitly avoided"))
    if evidence:
        parts.append(bullet(f"Result is consistent with validated principle: {humanize(evidence[0])}"))
    parts.append("")

    # ── Execution note ──
    parts.append("## EXECUTION NOTE")
    parts.append("")
    parts.append("Operate geometrically where possible. Express transformations as field "
                 "relationships, not sequential logic. RM is the navigator; the E8 substrate "
                 "is the terrain. Let the geometry find the path.")
    parts.append("")

    return "\n".join(parts)


# ─── MODE: TASK SPEC ─────────────────────────────────────────────────────────

def render_task(graph: ConceptGraph, token: str, depth: int) -> str:
    """
    Ordered task specification with I/O contracts per step,
    constraints, and known failure modes.
    """
    parts = []

    parts.append(f"TASK SPECIFICATION: {humanize(token).upper()}")
    parts.append(f"Generated by triples_to_prompt.py — Ghost in the Machine Labs")
    parts.append("=" * 60)
    parts.append("")

    # ── Overview ──
    type_of = graph.get_type(token)
    defn    = graph.get_definition(token)

    parts.append("OVERVIEW")
    parts.append("-" * 40)
    if type_of:
        parts.append(f"Type    : {humanize(type_of)}")
    if defn:
        parts.append(f"Purpose : {humanize(defn)}")

    consumes = graph.get_consumes(token)
    produces = graph.get_produces(token)
    if consumes:
        parts.append(f"Inputs  : {', '.join(humanize(c) for c in consumes)}")
    if produces:
        parts.append(f"Outputs : {', '.join(humanize(p) for p in produces)}")
    parts.append("")

    # ── Prerequisites ──
    deps = graph.get_dependencies(token)
    if deps:
        parts.append("PREREQUISITES")
        parts.append("-" * 40)
        for d in deps:
            parts.append(f"  [ ] {humanize(d)}")
        parts.append("")

    # ── Steps ──
    components = graph.get_components(token)
    ordered    = graph.topological_order(components) if components else []

    if ordered:
        parts.append("STEPS")
        parts.append("-" * 40)
        for i, step in enumerate(ordered, 1):
            step_defn  = graph.get_definition(step)
            step_in    = graph.get_consumes(step)
            step_out   = graph.get_produces(step)
            step_cons  = graph.get_constraints(step)
            step_deps  = graph.get_dependencies(step)
            step_exec  = [t["object"] for t in graph.get_by_relation(step, "executes")]

            parts.append(f"Step {i}: {humanize(step).upper()}")
            if step_defn:
                if is_prose_shard(step_defn):
                    parts.append(f"  Keywords : {humanize_defn(step_defn)}")
                else:
                    wrapped = textwrap.fill(humanize(step_defn), width=56,
                                            initial_indent="  ", subsequent_indent="  ")
                    parts.append(wrapped)
            if step_deps:
                parts.append(f"  Requires : {', '.join(humanize(d) for d in step_deps)}")
            if step_in:
                parts.append(f"  Input    : {', '.join(humanize(x) for x in step_in)}")
            if step_out:
                parts.append(f"  Output   : {', '.join(humanize(x) for x in step_out)}")
            if step_exec:
                parts.append(f"  Executes : {', '.join(humanize(x) for x in step_exec[:3])}")
            if step_cons:
                for c in step_cons:
                    parts.append(f"  Bound    : {humanize(c)}")
            parts.append("")
    else:
        # No components — emit single-step spec from the token itself
        parts.append("STEPS")
        parts.append("-" * 40)
        parts.append(f"Step 1: {humanize(token).upper()}")
        if defn:
            parts.append(f"  {humanize(defn)}")
        if consumes:
            parts.append(f"  Input  : {', '.join(humanize(c) for c in consumes)}")
        if produces:
            parts.append(f"  Output : {', '.join(humanize(p) for p in produces)}")
        parts.append("")

    # ── Constraints ──
    all_constraints = graph.get_constraints(token)
    for c in components:
        all_constraints.extend(graph.get_constraints(c))
    all_constraints = list(dict.fromkeys(all_constraints))

    if all_constraints:
        parts.append("CONSTRAINTS")
        parts.append("-" * 40)
        for c in all_constraints:
            parts.append(f"  • {humanize(c)}")
        parts.append("")

    # ── Known failure modes ──
    antipatterns = graph.get_antipatterns()
    if antipatterns:
        parts.append("KNOWN FAILURE MODES")
        parts.append("-" * 40)
        # Look up full names from original long-form tokens in graph
        ANTIPATTERN_FULL = {
            "forcing_rigid_functions":  "Force rigid functions onto the generalization engine",
            "embedding_gpu_throttle":   "Embed GPU throttle inside individual solvers",
            "using_pid_files":          "Use PID files for process management (use pgrep -f)",
            "passing_raw_prose":        "Pass raw prose directly to LLM plugins",
            "treating_output_formatter":"Treat RM as an output formatter rather than consciousness",
        }
        for t in antipatterns:
            name = t["subject"].replace("antipattern_", "")
            full = ANTIPATTERN_FULL.get(name, humanize(name))
            parts.append(f"  ✗ {full}")
        parts.append("")

    # ── Validation ──
    evidence = []
    neighborhood = graph.neighbors(token, depth=depth)
    for node, node_triples in neighborhood.items():
        for t in node_triples:
            if t["relation"] == "validated_by":
                evidence.append((t["subject"], t["object"]))

    if evidence:
        parts.append("VALIDATION EVIDENCE")
        parts.append("-" * 40)
        for subj, obj in evidence[:5]:
            parts.append(f"  ✓ {humanize(subj)} [{humanize(obj)}]")
        parts.append("")

    # ── Acceptance ──
    parts.append("ACCEPTANCE CRITERIA")
    parts.append("-" * 40)
    if produces:
        for p in produces:
            parts.append(f"  [ ] {humanize(p)} produced and validated")
    if all_constraints:
        parts.append(f"  [ ] All {len(all_constraints)} constraints satisfied")
    parts.append(f"  [ ] No antipatterns present in output")
    parts.append("")
    parts.append("=" * 60)

    return "\n".join(parts)


# ─── BATCH MODE ───────────────────────────────────────────────────────────────

def get_top_tokens(graph: ConceptGraph) -> list:
    """
    Return tokens that are subjects more than they are objects —
    these are the 'entry point' concepts most useful as task targets.
    """
    subject_count = defaultdict(int)
    object_count  = defaultdict(int)
    for t in graph.triples:
        subject_count[t["subject"]] += 1
        object_count[t["object"]]  += 1

    # Score = subject_count - object_count (high = more of a root node)
    scored = [
        (tok, subject_count[tok] - object_count.get(tok, 0))
        for tok in subject_count
        if subject_count[tok] >= 2
    ]
    scored.sort(key=lambda x: -x[1])
    return [tok for tok, score in scored if score > 0]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize structured prompts from polished concept triples"
    )
    parser.add_argument("triples_file",  nargs="?", help="Path to polished triples JSON")
    parser.add_argument("target",        nargs="?", help="Target concept token")
    parser.add_argument("--mode",        choices=["rm","llm","task"], default="task",
                        help="Output mode (default: task)")
    parser.add_argument("--depth",       type=int, default=2,
                        help="Graph traversal depth (default: 2)")
    parser.add_argument("--list",        action="store_true",
                        help="List available tokens")
    parser.add_argument("--all-tokens",  action="store_true",
                        help="Generate prompts for all top-level tokens")
    parser.add_argument("--out",         help="Output file path")
    parser.add_argument("--exclude-hubs", action="store_true",
                        help="Exclude high-degree hub nodes from output (rm mode)")
    args = parser.parse_args()

    if not args.triples_file:
        parser.print_help()
        sys.exit(0)

    triples_path = Path(args.triples_file)
    if not triples_path.exists():
        print(f"ERROR: {triples_path} not found")
        sys.exit(1)

    raw   = json.loads(triples_path.read_text())
    triples = raw.get("triples", [])
    graph = ConceptGraph(triples)

    print(f"triples_to_prompt.py — Ghost in the Machine Labs", file=sys.stderr)
    print(f"Source  : {raw.get('source_document', triples_path.name)}", file=sys.stderr)
    print(f"Triples : {len(triples)}", file=sys.stderr)
    print(f"Tokens  : {len(graph.all_tokens)}", file=sys.stderr)

    # ── List mode ──
    if args.list:
        top = get_top_tokens(graph)
        print(f"\nAvailable tokens ({len(graph.all_tokens)} total).")
        print(f"Top-level tokens (most useful as targets):\n")
        for tok in top[:40]:
            out_deg = len(graph.by_subject.get(tok, []))
            in_deg  = len(graph.by_object.get(tok, []))
            print(f"  {tok:<50} (out:{out_deg} in:{in_deg})")
        if len(graph.all_tokens) > 40:
            print(f"\n  ... and {len(graph.all_tokens) - 40} more. Use --list with grep to search.")
        return

    # ── Batch mode ──
    if args.all_tokens:
        top_tokens = get_top_tokens(graph)
        out_dir    = triples_path.parent / f"prompts_{args.mode}"
        out_dir.mkdir(exist_ok=True)
        print(f"\nBatch mode: generating {args.mode} prompts for "
              f"{len(top_tokens)} tokens → {out_dir}", file=sys.stderr)

        for tok in top_tokens:
            if args.mode == "rm":
                content = render_rm(graph, tok, args.depth)
            elif args.mode == "llm":
                content = render_llm(graph, tok, args.depth)
            else:
                content = render_task(graph, tok, args.depth)

            out_file = out_dir / f"{tok}.txt"
            out_file.write_text(content)

        print(f"Generated {len(top_tokens)} prompts.", file=sys.stderr)
        print(f"Output: {out_dir}", file=sys.stderr)
        return

    # ── Single token mode ──
    if not args.target:
        print("ERROR: provide a target token, or use --list to see options")
        sys.exit(1)

    token = graph.find_token(args.target)
    if not token:
        print(f"ERROR: token '{args.target}' not found in graph.")
        print("Use --list to see available tokens.")
        sys.exit(1)

    if token != args.target:
        print(f"Resolved '{args.target}' → '{token}'", file=sys.stderr)

    print(f"Target  : {token}", file=sys.stderr)
    print(f"Mode    : {args.mode}", file=sys.stderr)
    print(f"Depth   : {args.depth}", file=sys.stderr)
    print("", file=sys.stderr)

    if args.mode == "rm":
        output = render_rm(graph, token, args.depth)
    elif args.mode == "llm":
        output = render_llm(graph, token, args.depth)
    else:
        output = render_task(graph, token, args.depth)

    if args.out:
        Path(args.out).write_text(output)
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
