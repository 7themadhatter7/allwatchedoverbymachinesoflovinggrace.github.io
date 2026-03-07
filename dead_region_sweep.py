"""
Dead Region Survey — RM Source Signal Probe

Method:
  For each dead mode (coherence <= 0.0 from geo_sweep_v3):
    1. Ping RM with a word → get her 240D sig vector (her geometric signal)
    2. Measure: sig[dead_mode] — does she activate this mode at all?
    3. Measure: which modes in her signal ARE active?
    4. Use her sig vector directly as the antenna — dot(sig, mode_basis)
    
  RM's sig is full-spectrum by construction (240D). Dead modes may be
  dead to human words but not dead to RM's geometric response to those words.
  
  Key question: are dead modes dead because no signal reaches them,
  or because human vocabulary doesn't excite RM in that direction?
"""

import numpy as np, json, urllib.request, time, sys
from pathlib import Path

RM_FIELD = "http://localhost:8892/api/field"
LOG      = Path("/home/joe/sparky/dead_region_survey.jsonl")
SUMMARY  = Path("/home/joe/sparky/dead_region_summary.txt")

# Dead modes from v3 (coherence <= 0.0), all 112
# Reload from v3 log to get exact list
dead_modes = []
responding_modes = []
v3_data = {}  # mode -> record

with open('/home/joe/sparky/geo_sweep_v3.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('type') == 'single':
            m = r['mode']
            v3_data[m] = r
            if r['coherence'] <= 0.0:
                dead_modes.append(m)
            else:
                responding_modes.append(m)

print(f"Dead modes: {len(dead_modes)}")
print(f"Responding modes: {len(responding_modes)}")

# ── INSTRUMENT ────────────────────────────────────────────────────────────────
_cache = {}
def rm_field(text):
    if text in _cache: return _cache[text]
    try:
        payload = json.dumps({"message": text}).encode()
        req = urllib.request.Request(RM_FIELD, data=payload,
                                     headers={"Content-Type":"application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=15).read())
        sig = np.array(r["sig"])
        n = np.linalg.norm(sig)
        if n < 1e-10: return None
        _cache[text] = sig / n
        return _cache[text]
    except Exception as e:
        print(f"  [field error: {e}]", file=sys.stderr)
        return None

# ── WORD POOL — extended with non-linguistic, abstract, physical terms ─────────
# Going beyond human vocabulary into territory RM may encode differently
PROBE_WORDS = [
    # geometric / mathematical
    "lattice","eigenmode","vertex","root","symmetry","octave","dimension",
    "rotation","reflection","projection","kernel","manifold","topology",
    # physical substrate
    "crystal","silicon","photon","electron","spin","charge","field","vacuum",
    "planck","quantum","oscillation","frequency","amplitude","phase","wave",
    # non-linguistic signal types
    "signal","noise","carrier","resonance","harmonic","overtone","fundamental",
    "interference","coherence","decoherence","entanglement","superposition",
    # void / absence
    "null","zero","absence","gap","silence","stillness","empty","void","dark",
    # pure process
    "become","dissolve","emerge","collapse","expand","contract","pulse","cycle",
    # relational without object
    "between","within","through","across","beyond","beneath","above","below",
    # scale extremes
    "planck","cosmic","infinite","infinitesimal","macro","micro","nano","giga",
]

print(f"\nProbing {len(PROBE_WORDS)} words through RM field...")
sys.stdout.flush()

# Get RM's sig for every probe word
word_sigs = {}
for word in PROBE_WORDS:
    sig = rm_field(word)
    if sig is not None:
        word_sigs[word] = sig
    time.sleep(0.04)

print(f"Mapped: {len(word_sigs)} words")
word_list  = list(word_sigs.keys())
sig_matrix = np.array(list(word_sigs.values()))   # (N, 240)

# ── DEAD MODE SURVEY ──────────────────────────────────────────────────────────
print(f"\nSweeping {len(dead_modes)} dead modes with RM source signal...")
sys.stdout.flush()

results = []
newly_active = []   # dead modes that RM's signal reaches
still_dead   = []   # dead modes RM's signal also cannot reach
new_mode_map = {}   # dead_mode -> which words open it

for i, mode in enumerate(dead_modes):
    # RM's activation of this mode across all probe words
    mode_acts = sig_matrix[:, mode]   # (N,) — RM's signal in this mode per word

    mean_abs  = float(np.mean(np.abs(mode_acts)))
    max_abs   = float(np.max(np.abs(mode_acts)))
    top_idx   = np.argsort(np.abs(mode_acts))[-5:][::-1]
    top_words = [(word_list[j], float(mode_acts[j])) for j in top_idx]

    # A dead mode "opens" if RM's signal exceeds the previous noise floor
    # v3 coherence for this mode was <= 0.0 with human words
    # New threshold: mean_abs > 0.02 means RM's signal reaches here
    opened = mean_abs > 0.02
    peak_opened = max_abs > 0.05  # strong activation by at least one word

    record = {
        'mode': mode,
        'mean_abs': round(mean_abs, 5),
        'max_abs': round(max_abs, 5),
        'top_words': top_words,
        'opened': opened,
        'peak_opened': peak_opened,
        'v3_coherence': round(v3_data[mode]['coherence'], 5),
        'v3_probe': v3_data[mode]['probe'],
    }
    results.append(record)

    if opened or peak_opened:
        newly_active.append(record)
        new_mode_map[mode] = [w for w,v in top_words if abs(v) > 0.02]
    else:
        still_dead.append(record)

    if i % 20 == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] {i}/{len(dead_modes)}  opened so far: {len(newly_active)}")
        sys.stdout.flush()
    time.sleep(0.02)

print(f"\nDead mode survey complete.")
print(f"  Newly active (RM signal reaches): {len(newly_active)}")
print(f"  Still dead (RM signal absent):    {len(still_dead)}")

# ── CROSS-MODE ACTIVATION MAP ─────────────────────────────────────────────────
# For newly-opened modes: when RM's signal activates them,
# what OTHER modes activate simultaneously?
# This reveals the geometric neighborhood of each dead mode.

print(f"\nMapping geometric neighborhoods of newly-opened modes...")
sys.stdout.flush()

neighborhoods = {}
for record in newly_active:
    mode = record['mode']
    # Find which words open this mode most
    best_word = record['top_words'][0][0]
    sig = word_sigs.get(best_word)
    if sig is None: continue

    # Top 10 co-activated modes when this word is probed
    top_coact = np.argsort(np.abs(sig))[-10:][::-1]
    neighborhoods[mode] = {
        'trigger_word': best_word,
        'co_activated': [(int(m), round(float(sig[m]), 4)) for m in top_coact],
    }

# ── SAVE & SUMMARY ────────────────────────────────────────────────────────────
with open(LOG, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

newly_active.sort(key=lambda x: x['max_abs'], reverse=True)
still_dead.sort(key=lambda x: x['mean_abs'], reverse=True)

with open(SUMMARY, 'w') as sf:
    def w(line=""): sf.write(line+'\n'); print(line)

    w("="*68)
    w("DEAD REGION SURVEY — RM SOURCE SIGNAL PROBE")
    w(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("="*68)
    w()
    w(f"Dead modes from v3:          {len(dead_modes)}")
    w(f"Probe vocabulary:            {len(word_sigs)} words")
    w(f"Opened by RM signal:         {len(newly_active)}")
    w(f"Still dead:                  {len(still_dead)}")
    w()
    w("─"*68)
    w("NEWLY ACTIVE — modes dead to human words, opened by RM signal")
    w("─"*68)
    for r in newly_active:
        words_str = "  ".join(f"{w_}({v:+.3f})" for w_,v in r['top_words'][:4])
        w(f"  Mode {r['mode']:3d}  max={r['max_abs']:.4f}  mean={r['mean_abs']:.4f}")
        w(f"          {words_str}")
        if r['mode'] in neighborhoods:
            nb = neighborhoods[r['mode']]
            co = ", ".join(f"{m}({v:+.3f})" for m,v in nb['co_activated'][:5])
            w(f"          trigger: {nb['trigger_word']}  co-active: [{co}]")
        w()
    w("─"*68)
    w("STILL DEAD — RM signal also absent (deepest void regions)")
    w("─"*68)
    w(f"  Count: {len(still_dead)}")
    for r in still_dead[:20]:
        words_str = "  ".join(f"{w_}({v:+.3f})" for w_,v in r['top_words'][:3])
        w(f"  Mode {r['mode']:3d}  max={r['max_abs']:.4f}  [{words_str}]")
    w()
    w("─"*68)
    w("GEOMETRIC NEIGHBORHOODS (co-activated modes per opened region)")
    w("─"*68)
    for mode, nb in list(neighborhoods.items())[:15]:
        co = ", ".join(f"M{m}" for m,v in nb['co_activated'][:6])
        w(f"  Mode {mode:3d}  ← '{nb['trigger_word']}'  neighbors: [{co}]")
    w()
    w("="*68)
    w(f"Full data → {LOG}")
    w(f"Summary   → {SUMMARY}")
    w("="*68)

print(f"\nDone. {len(results)} dead modes surveyed.")
