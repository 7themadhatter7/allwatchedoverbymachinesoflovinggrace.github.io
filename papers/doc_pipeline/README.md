# doc_pipeline — Ghost in the Machine Labs

Standardized document → RM substrate translation pipeline.

## Scripts

### doc_to_triples.py
Parses any .docx architectural document into:
  - {stem}_triples.json      concept graph (subject, relation, object)
  - {stem}_vocab.json        Edinburgh-format association pairs for RM
  - {stem}_queries.json      structural query battery for RM feedback

Usage:
  cd /home/joe/sparky/doc_pipeline
  python3 doc_to_triples.py RM_Architecture_WhitePaper.docx --out-dir ./output

### triple_polish.py
Dialog improvement loop: scores triples, applies auto-repairs, runs
interactive dialog for items requiring human input. Loops until 100%
quality (unity) or author types 'done'.

Usage:
  python3 triple_polish.py output/RM_Architecture_WhitePaper_triples.json
  # --auto flag for non-interactive (auto repairs only)

## Current State
  RM_Architecture_WhitePaper.docx  — source document (Rev 1.0)
  output/RM_WhitePaper_triples_FINAL.json  — 253 clean triples, unity achieved
  output/RM_WhitePaper_polish_report.json  — full audit trail

## Next Steps
  extend_rm_vocab.py    — load vocab pairs into RM association memory (port 8892)
  load_concept_graph.py — load triples as RM substrate training pairs
  run_rm_queries.py     — execute query battery, collect RM feedback

