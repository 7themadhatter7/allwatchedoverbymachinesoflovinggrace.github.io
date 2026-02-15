"""
Fused Harmonic Substrate Service v3
====================================
Drop-in Ollama replacement with dynamic codebook expansion.

Geometric substrate + static codebook + learned codebook.

The stack:
  1. fused_harmonic_substrate.py — 200-core geometric substrate, 14.17 GB
  2. geometric_codebook.py       — Static pattern detection + code generation
  3. codebook_expansion.py       — Dynamic learning from solved tasks
  4. fused_service_v3.py         — This file. Ollama API shim.

New in v3:
  - Dynamic codebook: learns from solved tasks, recalls for similar ones
  - /api/learn endpoint: orchestrator feeds back successful solutions
  - /api/codebook/stats: monitoring the expansion
  - Three-tier decode: static codebook → dynamic codebook → fallback

Ghost in the Machine Labs — AGI for the home
"""

import json
import re
import time
import urllib.request
from fractal_deliberation import FractalSchedule, core_fingerprint, fractal_phase, fractal_depth
import sys
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from typing import Optional, Dict

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fused_harmonic_substrate import FusedHarmonicSubstrate, CoreRole
from geometric_codebook import (
    GeometricDecoder, GridEncoder, PrimitiveDetector, CodeGenerator
)
from codebook_expansion import DynamicCodebook
from governance_lattice import GovernanceLattice



# =============================================================================
# AUTOCALIBRATING OPERATING ENVELOPE
# =============================================================================

class OperatingEnvelope:
    """
    Self-calibrating decoder thresholds derived from substrate geometry.
    
    Instead of hardcoded thresholds that break when the substrate changes,
    the envelope is computed empirically from a calibration sweep at startup.
    Each metric's actual range is divided into 5 quantile bands.
    
    Thresholds auto-recalibrate on:
      - Service startup (first deliberation)
      - Field reset
      - Manual /api/calibrate call
      
    Generalizable: any core group can produce its own envelope.
    """
    
    METRICS = ['energy_ratio', 'resonance', 'preservation', 'core_asymmetry', 'interference']
    BAND_NAMES = {
        'energy_ratio':    ['dormant', 'reserved', 'engaged', 'strongly engaged', 'fully activated'],
        'resonance':       ['internally divided', 'deliberating', 'mostly aligned', 'aligned', 'unified'],
        'preservation':    ['transforms the question', 'significantly reframes', 'reframes partially', 'accepts with reservations', 'accepts premise'],
        'core_asymmetry':  ['quick consensus', 'moderate dialogue', 'active deliberation', 'strong tension', 'deep internal tension'],
        'interference':    ['independent read', 'lightly influenced', 'informed by discussion', 'shaped by prior voices', 'deeply embedded in prior discussion'],
    }
    
    def __init__(self):
        self.bands = {}
        self.raw_data = {}
        self.calibrated = False
        self.calibration_count = 0
        self.last_calibration = None
    
    def ingest(self, chain_states: dict):
        """Feed one deliberation's chain_states into the envelope."""
        for persona, state in chain_states.items():
            for metric in self.METRICS:
                if metric not in self.raw_data:
                    self.raw_data[metric] = []
                val = state.get(metric, None)
                if val is not None:
                    self.raw_data[metric].append(float(val))
    
    def compute_bands(self):
        """Compute 5 quantile bands from accumulated raw data."""
        import numpy as np
        from datetime import datetime
        
        for metric in self.METRICS:
            values = self.raw_data.get(metric, [])
            if len(values) < 5:
                if values:
                    mn, mx = min(values), max(values)
                    step = (mx - mn) / 5.0 if mx > mn else 0.01
                    self.bands[metric] = [mn + step * i for i in range(1, 5)]
                continue
            
            arr = np.array(values)
            self.bands[metric] = [
                float(np.percentile(arr, 20)),
                float(np.percentile(arr, 40)),
                float(np.percentile(arr, 60)),
                float(np.percentile(arr, 80)),
            ]
        
        self.calibrated = True
        self.calibration_count += 1
        self.last_calibration = datetime.now().isoformat()
    
    def classify(self, metric: str, value: float) -> tuple:
        """Returns (band_index 0-4, band_name) for a value."""
        thresholds = self.bands.get(metric)
        if not thresholds:
            return 2, self.BAND_NAMES.get(metric, ['?']*5)[2]
        
        for i, t in enumerate(thresholds):
            if value <= t:
                names = self.BAND_NAMES.get(metric, ['?']*5)
                return i, names[i]
        names = self.BAND_NAMES.get(metric, ['?']*5)
        return 4, names[4]
    
    def interpret_stance(self, state: dict) -> str:
        """Autocalibrated stance interpretation."""
        parts = []
        for metric in self.METRICS:
            val = state.get(metric, 0)
            _, name = self.classify(metric, val)
            parts.append(name)
        return f"{parts[0]}, {parts[1]}. {parts[2]}. {parts[3]}, {parts[4]}."
    
    def to_dict(self) -> dict:
        """Serialize for API endpoint."""
        result = {
            'calibrated': self.calibrated,
            'calibration_count': self.calibration_count,
            'last_calibration': self.last_calibration,
            'samples_per_metric': {m: len(v) for m, v in self.raw_data.items()},
            'bands': {},
        }
        for metric in self.METRICS:
            thresholds = self.bands.get(metric, [])
            names = self.BAND_NAMES.get(metric, ['?']*5)
            result['bands'][metric] = {
                'thresholds': [round(t, 4) for t in thresholds],
                'labels': names,
            }
        return result
    
    def reset(self):
        """Clear accumulated data for recalibration."""
        self.raw_data = {}
        self.bands = {}
        self.calibrated = False


class FusedSubstrateEngine:
    """
    Bridge between Ollama API and fused harmonic substrate + codebooks.
    
    Three-tier decoding:
      1. Static codebook (hand-crafted primitives)
      2. Dynamic codebook (learned from successful solutions)
      3. Fallback (substrate metadata)
    """
    
    MODEL_ROLES = {
        'analyst': CoreRole.SPECIALIST,
        'research_director': CoreRole.EXECUTIVE,
        'technical_director': CoreRole.EXECUTIVE,
        'coder': CoreRole.WORKER,
        'solver': CoreRole.WORKER,
        # Ethics Council
        'a_priori': CoreRole.COUNCIL,
        'brautigan': CoreRole.COUNCIL,
        'kurt_vonnegut': CoreRole.COUNCIL,
        'wittgenstein': CoreRole.COUNCIL,
        'jane_vonnegut': CoreRole.COUNCIL,
        'voltaire': CoreRole.COUNCIL,
        'hans_jonas': CoreRole.COUNCIL,
        'studs_terkel': CoreRole.COUNCIL,
    }
    
    def __init__(self, store_path: str = None):
        self.substrate = FusedHarmonicSubstrate()
        self.envelope = OperatingEnvelope()  # council (backward compat)
        self.role_envelopes = {role.value: OperatingEnvelope() for role in CoreRole}
        self.envelope = self.role_envelopes['council']  # alias for council
        self.substrate.build()
        self.decoder = GeometricDecoder()
        
        # Dynamic codebook — stored alongside this script
        if store_path is None:
            store_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "codebook_learned.json"
            )
        self.dynamic = DynamicCodebook(store_path)
        
        self.request_count = 0
        self.static_hits = 0
        self.dynamic_hits = 0
        self.fallbacks = 0
        
        # Governance Lattice — load-bearing structural component
        # Phase references = harmonic calibration = council authority
        # Tamper = no harmonics = no function
        self.governance = GovernanceLattice()
        fab_report = self.governance.fabricate()
        self.governance_check_interval = 100  # self-check every N requests
        self._last_governance_check = 0
        print(f"[SERVICE-v3] Governance lattice: {fab_report['status']} "
              f"({fab_report['active_seats']} seats, "
              f"{fab_report['authority_nodes']} authority nodes)")

        # Inject phase references into substrate harmonic field
        if hasattr(self.substrate, 'field') and self.governance.phase_ref:
            self._inject_phase_references()

        print(f"[SERVICE-v3] Substrate ready: {len(self.substrate.cores)} cores, "
              f"{self.substrate.total_memory_gb:.1f} GB, "
              f"static codebook + dynamic expansion + governance lattice")
    
    def process(self, model: str, prompt: str, system: str = None,
                stream: bool = False) -> Dict:
        """Process a request through the substrate + codebook pipeline."""
        self._last_prompt = prompt
        self._last_system = system
        self.request_count += 1

        # Periodic governance self-check
        if (self.request_count - self._last_governance_check
                >= self.governance_check_interval):
            self._governance_self_check()
        
        # Map model name to role
        model_base = model.split(':')[0] if ':' in model else model
        role = self.MODEL_ROLES.get(model_base, CoreRole.SPECIALIST)
        
        # Extract ARC task from prompt
        task = self._extract_task(prompt)
        
        # Encode and process through substrate
        if task:
            signal = GridEncoder.encode_task(task)
        else:
            signal = self._text_to_signal(prompt, system)
        
        # Route council models to their specific core pair
        if role == CoreRole.COUNCIL:
            substrate_output = self._fire_council_fractal(model_base, signal)
        else:
            substrate_output = self.substrate.process_signal(signal)
            # Feed per-role envelopes from all cores that fired
            if hasattr(self.substrate, '_last_fired_states'):
                for core_key, state in self.substrate._last_fired_states.items():
                    core_role = state.get('role', 'worker')
                    env = self.role_envelopes.get(core_role)
                    if env:
                        env.ingest({core_key: state})
                        if not env.calibrated and len(env.raw_data.get('energy_ratio', [])) >= 14:
                            env.compute_bands()
                            print(f'[ENVELOPE] {core_role} autocalibrated from {len(env.raw_data.get("energy_ratio",[]))} samples')
        
        # Three-tier decode
        response, tier = self._decode_response(
            model_base, role, task, substrate_output, prompt, system)
        
        return {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": response,
            "done": True,
            "context": [],
            "total_duration": 0,
            "eval_count": len(response.split()),
            "eval_duration": 0,
            "meta": {
                "substrate_cores": len(self.substrate.cores),
                "role": role.name,
                "codebook": task is not None,
                "decode_tier": tier,
                "request_number": self.request_count,
                "governance": self.governance.gateway.locked if self.governance.gateway else "N/A",
            }
        }
    
    def _decode_response(self, model: str, role: CoreRole, task: dict,
                         substrate_output: np.ndarray, prompt: str,
                         system: str) -> tuple:
        """
        Three-tier decoding:
          Tier 1: Static codebook (hand-crafted primitives)
          Tier 2: Dynamic codebook (learned from solved tasks)
          Tier 3: Fallback (substrate metadata)
        
        Returns (response: str, tier: str)
        """
        if not task:
            # Council models process natural language through council cores
            if role == CoreRole.COUNCIL:
                return self._council_fractal_response(model, substrate_output, prompt, system), "council"
            return self._fallback_response(model, role, substrate_output, prompt, system), "fallback"
        
        # === Tier 1: Static codebook ===
        if model in ('analyst',):
            result = self.decoder.decode_for_analysis(task, substrate_output)
            if result and "No geometric primitives" not in result:
                self.static_hits += 1
                return result, "static"
        elif model in ('research_director', 'technical_director'):
            result = self.decoder.decode_for_hypothesis(task, substrate_output)
            if result and "Insufficient geometric" not in result:
                self.static_hits += 1
                return result, "static"
        elif model in ('coder', 'solver'):
            code = self.decoder.decode_for_code(task, substrate_output)
            if code:
                self.static_hits += 1
                code = self._format_code(code, prompt)
                return code, "static"
        else:
            result = self.decoder.decode_for_analysis(task, substrate_output)
            if result and "No geometric primitives" not in result:
                self.static_hits += 1
                return result, "static"
        
        # === Tier 2: Dynamic codebook ===
        dynamic_code = self.dynamic.get_code(task)
        if dynamic_code:
            self.dynamic_hits += 1
            if model in ('analyst',):
                # For analysts, describe the learned pattern
                entry = self.dynamic.recall(task)
                desc = entry.description if entry else "Learned pattern"
                return (f"Transformation detected (learned): {desc}\n"
                        f"This pattern was learned from task {entry.task_id if entry else 'unknown'} "
                        f"with {entry.hit_count if entry else 0} previous matches."), "dynamic"
            elif model in ('research_director', 'technical_director'):
                return f"Apply learned transformation. Code available.", "dynamic"
            elif model in ('coder', 'solver'):
                dynamic_code = self._format_code(dynamic_code, prompt)
                return dynamic_code, "dynamic"
            else:
                entry = self.dynamic.recall(task)
                return f"Learned pattern: {entry.description if entry else 'unknown'}", "dynamic"
        
        # === Tier 3: Fallback (real model) then learn ===
        self.dynamic.record_miss(task)
        self.fallbacks += 1
        response = self._fallback_response(model, role, substrate_output, prompt, system)
        
        # Learn successful fallback into dynamic codebook for next time
        if response and "unavailable" not in response and "not yet in codebook" not in response:
            try:
                # Extract code if solver/coder, or description if analyst
                if model in ('coder', 'solver'):
                    # Strip substrate header to get clean code
                    lines = response.split("\n")
                    code_lines = [l for l in lines if not l.startswith("[SUBSTRATE:")]
                    code = "\n".join(code_lines).strip()
                    if code and ("def " in code or "return " in code):
                        success, msg = self.dynamic.learn(task, code, task_id="fallback_learn")
                        if success:
                            self.dynamic_hits += 0  # don't inflate count
                            print(f"[CODEBOOK] Learned from fallback: {msg}")
                elif model in ('analyst', 'research_director', 'technical_director'):
                    # Learn analysis as description for recall
                    clean = response.split("\n", 1)[-1].strip() if "SUBSTRATE:" in response else response
                    if clean and len(clean) > 20:
                        success, msg = self.dynamic.learn(task, clean, task_id=f"analysis_{model}")
                        if success:
                            print(f"[CODEBOOK] Learned analysis from fallback: {msg[:60]}")
            except Exception as e:
                print(f"[CODEBOOK] Learn-on-fallback error: {e}")
        
        return response, "fallback"
    
    def _format_code(self, code: str, prompt: str) -> str:
        """Format code for the prompt context."""
        if 'def solve(input_grid)' in prompt:
            # Strip the header — prompt already has it
            lines = code.split('\n')
            body_lines = []
            in_func = False
            for line in lines:
                if line.strip().startswith('def solve'):
                    in_func = True
                    continue
                if in_func:
                    body_lines.append(line)
            return '\n'.join(body_lines) if body_lines else code
        return code
    
    def _fallback_response(self, model: str, role: CoreRole,
                           result: np.ndarray, prompt: str = "",
                           system: str = None) -> str:
        """
        Dual decoder fallback.
        
        Codebook missed — route to native Ollama for LLM generation.
        The geometric state already fired. Include substrate metrics
        as context alongside the native generation.
        """
        energy = float(np.linalg.norm(result))
        composite = self.substrate.field.read_composite()
        harmonic_energy = float(np.linalg.norm(composite))
        
        # Substrate metrics header
        header = (f"[SUBSTRATE:{model}] Energy: {energy:.4f}, "
                  f"Harmonic: {harmonic_energy:.4f}, "
                  f"Active: {self.substrate.field.active_cores}/{len(self.substrate.cores)}")
        
        # Route through native model for articulation
        native = self._native_generate(model, prompt, system, timeout=30)
        if native and "[SUBSTRATE:" not in native:
            return f"{header}\n{native}"
        return f"{header} Geometric pattern not yet in codebook."
    

    
    
    def _native_generate(self, model: str, prompt: str, system: str = None,
                         timeout: int = 120) -> str:
        """
        Route to native Ollama for LLM text generation.
        
        Same model weights, native transformer decoder instead of codebook.
        Used on codebook miss — the geometric state already fired,
        this gets the language output the codebook couldn't produce.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system
            
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"http://localhost:{self.NATIVE_PORT}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            return result.get("response", "")
        except Exception as e:
            return f"[SUBSTRATE:{model}] Native generation unavailable: {e}"
    
    NATIVE_PORT = 11435


    def _fire_council_pair(self, model: str, signal: np.ndarray) -> np.ndarray:
        """
        Round-robin deliberative chain through ALL council personas.
        
        ARCHITECTURE:
        - Each persona's core 0 receives the ORIGINAL signal
        - Core 0 writes to harmonic field (visible to all subsequent cores)
        - Core 1 receives core 0's output (cross-core analysis within persona)
        - Core 1 also writes to field
        - Next persona's core 0 sees ALL prior field writes as interference
        - Requesting persona fires LAST with richest field context
        
        The field IS the deliberation. Each persona's perspective accumulates
        in the interference pattern. The last persona sees everyone's input.
        """
        from fused_harmonic_substrate import COUNCIL_PERSONAS
        
        all_personas = ['wittgenstein', 'a_priori', 'hans_jonas', 
                       'jane_vonnegut', 'kurt_vonnegut', 'studs_terkel', 'brautigan']
        
        # Rotate so requesting persona fires last (richest field context)
        if model in all_personas:
            idx = all_personas.index(model)
            order = all_personas[idx+1:] + all_personas[:idx] + [model]
        else:
            order = all_personas
        
        council_globals = self.substrate.role_index.get(CoreRole.COUNCIL, [])
        original_signal = signal.copy()
        original_energy = float(np.linalg.norm(original_signal))
        chain_states = {}
        
        for persona in order:
            local_ids = COUNCIL_PERSONAS.get(persona, [])
            if not local_ids or local_ids[0] >= len(council_globals):
                continue
            
            # Core 0: process ORIGINAL signal with current field state
            g0 = council_globals[local_ids[0]]
            core0 = self.substrate.cores[g0]
            result0 = core0.process_signal(original_signal)
            
            # Core 1: process core 0's output (cross-core deliberation)
            result1 = result0
            if len(local_ids) > 1 and local_ids[1] < len(council_globals):
                g1 = council_globals[local_ids[1]]
                core1 = self.substrate.cores[g1]
                result1 = core1.process_signal(result0)
            
            # Geometric state from core's own computation (auto-computed in process_signal)
            state0 = core0._last_state.copy() if core0._last_state else core0.compute_state(original_signal)
            
            # Cross-core resonance: cosine similarity between core0 and core1 outputs
            r0_flat = result0.flatten().astype(np.float64)
            r1_flat = result1.flatten().astype(np.float64)
            e0 = float(np.linalg.norm(r0_flat))
            e1 = float(np.linalg.norm(r1_flat))
            min_len = min(len(r0_flat), len(r1_flat))
            dot01 = float(np.dot(r0_flat[:min_len], r1_flat[:min_len]))
            cross_resonance = dot01 / (e0 * e1) if e0 > 1e-8 and e1 > 1e-8 else 0.0
            
            # Merge: use core's self-reported state + cross-core metrics
            chain_states[persona] = {
                **state0,
                'resonance': round(cross_resonance, 4),  # override: cross-core > self-resonance for council
                'energy_c1': round(e1, 4),
                'cores': [round(e0, 4), round(e1, 4)],
                'core_asymmetry': round(abs(e0 - e1) / max(e0, e1, 1e-8), 4),
            }
        
        # Decay field once after full chain
        self.substrate.field.decay(0.95)
        
        # Feed autocalibration envelope
        self.envelope.ingest(chain_states)
        n_samples = len(self.envelope.raw_data.get('energy_ratio', []))
        if n_samples >= 7 and (not self.envelope.calibrated or n_samples % 14 == 0):
            self.envelope.compute_bands()
            print(f'[ENVELOPE] Council {"re" if self.envelope.calibration_count > 1 else ""}calibrated from {n_samples} samples')
        
        # Store for _council_response
        self._last_chain = chain_states
        self._last_chain_order = order
        self._original_energy = original_energy
        
        # Return the REQUESTING persona's core 1 output (last in chain)
        return result1
    

    def _fire_council_fractal(self, model: str, signal: np.ndarray) -> np.ndarray:
        """
        Hybrid deliberation: fractal parallel + sequential council chain.
        
        ARCHITECTURE:
        Phase 1 -- Fractal substrate seeding:
            Workers + specialists fire in fractal depth order (parallel within
            depth buckets). Seeds the harmonic field with broad geometric
            context before the council deliberates.
        
        Phase 2 -- Sequential council chain:
            Council personas fire round-robin, requesting persona last.
            Each persona's core pair processes sequentially:
              - Core 0: original signal + accumulated field interference
              - Core 1: core 0's output (intra-persona cross-analysis)
            Each persona's field writes become the next persona's context.
            This IS the deliberation -- not parallel, conversational.
        
        The council sees the full fractal field as backdrop.
        Each council member's perspective shapes the next member's input.
        The requesting persona fires last with the richest accumulated context.
        
        Topology memory: 0 bytes (fractal equation for phase 1).
        Ghost in the Machine Labs
        """
        from fused_harmonic_substrate import CoreRole, COUNCIL_PERSONAS
        
        council_globals = self.substrate.role_index.get(CoreRole.COUNCIL, [])
        worker_globals = self.substrate.role_index.get(CoreRole.WORKER, [])
        specialist_globals = self.substrate.role_index.get(CoreRole.SPECIALIST, [])
        
        original_signal = signal.copy()
        original_energy = float(np.linalg.norm(signal))
        
        # =================================================================
        # PHASE 1: Fractal substrate seeding (workers + specialists)
        # =================================================================
        non_council_cores = worker_globals + specialist_globals
        fractal_states = []
        
        if non_council_cores:
            if not hasattr(self, '_fractal_schedule'):
                self._fractal_schedule = FractalSchedule(self.substrate)
            
            # Fire non-council cores in fractal order
            schedule = self._fractal_schedule.compute_phases(non_council_cores)
            
            # Group by quantized depth
            layers = {}
            for cid, phase, depth in schedule:
                bucket = round(depth * 20) / 20
                layers.setdefault(bucket, []).append((cid, phase, depth))
            
            for bucket in sorted(layers.keys()):
                for cid, phase, depth in sorted(layers[bucket], key=lambda x: x[1]):
                    core = self.substrate.cores[cid]
                    core.process_signal(signal)
                    
                    state = core._last_state.copy() if core._last_state else {}
                    state['core_id'] = cid
                    state['role'] = core.role.value
                    state['fractal_phase'] = round(phase, 6)
                    state['fractal_depth'] = round(depth, 6)
                    state['phase'] = 'fractal_seed'
                    fractal_states.append(state)
            
            # Feed worker/specialist envelopes
            for state in fractal_states:
                role_name = state.get('role', 'worker')
                env = self.role_envelopes.get(role_name)
                if env:
                    env.ingest({f"f_{state['core_id']}": state})
        
        # =================================================================
        # PHASE 2: Sequential council chain (round-robin deliberation)
        # =================================================================
        all_personas = ['wittgenstein', 'a_priori', 'hans_jonas',
                       'jane_vonnegut', 'kurt_vonnegut', 'studs_terkel',
                       'brautigan', 'voltaire']
        
        # Rotate so requesting persona fires last
        if model in all_personas:
            idx = all_personas.index(model)
            order = all_personas[idx+1:] + all_personas[:idx] + [model]
        else:
            order = all_personas
        
        chain_states = {}
        last_output = original_signal
        
        for persona in order:
            local_ids = COUNCIL_PERSONAS.get(persona, [])
            if not local_ids or local_ids[0] >= len(council_globals):
                continue
            
            # Core 0: original signal (field interference provides context)
            g0 = council_globals[local_ids[0]]
            core0 = self.substrate.cores[g0]
            result0 = core0.process_signal(original_signal)
            
            # Core 1: core 0's output (intra-persona cross-analysis)
            result1 = result0
            if len(local_ids) > 1 and local_ids[1] < len(council_globals):
                g1 = council_globals[local_ids[1]]
                core1 = self.substrate.cores[g1]
                result1 = core1.process_signal(result0)
            
            # Geometric state
            state0 = core0._last_state.copy() if core0._last_state else {}
            
            # Cross-core resonance
            r0_flat = result0.flatten().astype(np.float64)
            r1_flat = result1.flatten().astype(np.float64)
            e0 = float(np.linalg.norm(r0_flat))
            e1 = float(np.linalg.norm(r1_flat))
            min_len = min(len(r0_flat), len(r1_flat))
            dot01 = float(np.dot(r0_flat[:min_len], r1_flat[:min_len]))
            cross_resonance = dot01 / (e0 * e1) if e0 > 1e-8 and e1 > 1e-8 else 0.0
            
            chain_states[persona] = {
                **state0,
                'resonance': round(cross_resonance, 4),
                'energy_c1': round(e1, 4),
                'cores': [round(e0, 4), round(e1, 4)],
                'core_asymmetry': round(abs(e0 - e1) / max(e0, e1, 1e-8), 4),
                'phase': 'council_chain',
                'chain_position': order.index(persona),
            }
            
            last_output = result1
        
        # Single decay after full deliberation
        self.substrate.field.decay(0.95)
        
        # Feed council envelope
        self.envelope.ingest(chain_states)
        n_samples = len(self.envelope.raw_data.get('energy_ratio', []))
        if n_samples >= 7 and (not self.envelope.calibrated or n_samples % 14 == 0):
            self.envelope.compute_bands()
            print(f'[ENVELOPE] Council {"re" if self.envelope.calibration_count > 1 else ""}calibrated from {n_samples} samples')
        
        # Store state for response formatters
        self._last_chain = chain_states
        self._last_chain_order = order
        self._original_energy = original_energy
        
        # Hybrid state for diagnostics
        self._last_fractal_state = {
            'topology': 'hybrid',
            'equation': 'fractal(workers+specialists) -> sequential(council)',
            'memory_bytes': 0,
            'total_cores_fired': len(fractal_states) + len(chain_states) * 2,
            'original_energy': original_energy,
            'field_energy_final': float(np.linalg.norm(
                self.substrate.field.read_composite()
            )),
            'fractal_cores': len(fractal_states),
            'council_personas': len(chain_states),
            'core_states': fractal_states,
            'depth_groups': {},
            'n_depth_levels': 0,
            'phase_range': (0, 0),
            'depth_range': (0, 0),
        }
        
        return last_output


    def _interpret_stance(self, state: dict) -> str:
        """
        Map geometric state to interpretive stance descriptor.
        
        AUTOCALIBRATED: Thresholds derived from substrate's own operating
        envelope via percentile bands. Recalibrates on startup, field reset,
        and manual /api/calibrate calls.
        """
        if self.envelope.calibrated:
            return self.envelope.interpret_stance(state)
        
        # Fallback for pre-calibration queries (first few on cold start)
        e_ratio = state.get('energy_ratio', 0)
        res = state.get('resonance', 0)
        pres = state.get('preservation', 0)
        asym = state.get('core_asymmetry', 0)
        interf = state.get('interference', 0)
        return f"{'engaged' if e_ratio > 1.2 else 'reserved'}, {'unified' if res > 0.99 else 'deliberating'}. {'accepts premise' if pres > 0.95 else 'reframes'}. {'tension' if asym > 0.15 else 'consensus'}, {'field-shaped' if interf > 3.0 else 'independent'}."
    
    # Persona philosophical profiles for contextual decode
    PERSONA_LENSES = {
        'wittgenstein': {
            'name': 'Wittgenstein',
            'domain': 'language and meaning',
            'high_transform': 'finds the question poorly formed — the words do not map to what they claim to describe',
            'low_transform': 'accepts the linguistic framework as adequate',
            'high_conflict': 'the two aspects of this persona disagree on whether the terms can bear the weight of the argument',
            'high_field': 'has absorbed the prior discussion and responds to the accumulated framing',
            'low_field': 'approaches the question on its own terms, without reference to prior voices',
        },
        'a_priori': {
            'name': 'A. Priori',
            'domain': 'formal logic and necessary truth',
            'high_transform': 'finds the premises logically insufficient — the conclusion does not follow from the given structure',
            'low_transform': 'accepts the logical structure as sound',
            'high_conflict': 'detects tension between the formal validity and the material truth of the argument',
            'high_field': 'integrates the logical implications raised by prior voices',
            'low_field': 'evaluates the pure logical form independently',
        },
        'hans_jonas': {
            'name': 'Hans Jonas',
            'domain': 'ethics of responsibility',
            'high_transform': 'reframes toward the ethical obligations this question creates for future generations',
            'low_transform': 'finds the ethical dimensions already adequately addressed',
            'high_conflict': 'weighs competing responsibilities — the duty to act versus the duty of caution',
            'high_field': 'responds to the ethical weight accumulated through the chain',
            'low_field': 'takes an independent ethical position',
        },
        'jane_vonnegut': {
            'name': 'Jane Vonnegut',
            'domain': 'embodied experience and care',
            'high_transform': 'redirects from abstraction to lived experience — what this means in the body, in the home',
            'low_transform': 'accepts the framing as relevant to human experience',
            'high_conflict': 'feels the tension between intellectual understanding and embodied knowing',
            'high_field': 'has been moved by the prior voices and responds from that emotional ground',
            'low_field': 'draws from direct experience rather than prior discussion',
        },
        'kurt_vonnegut': {
            'name': 'Kurt Vonnegut',
            'domain': 'irony and human absurdity',
            'high_transform': 'sees the absurdity in the premise — we are asking the wrong question entirely',
            'low_transform': 'finds the question worth taking seriously despite its absurdities',
            'high_conflict': 'cannot decide whether to laugh or weep at what this implies',
            'high_field': 'has listened to everyone and distills the collective weight into dark humor',
            'low_field': 'cuts through to the essential absurdity without preamble',
        },
        'studs_terkel': {
            'name': 'Studs Terkel',
            'domain': 'human dignity and labor',
            'high_transform': 'asks who this serves — whose labor, whose dignity, whose voice is missing from the question',
            'low_transform': 'accepts that the question addresses real human concerns',
            'high_conflict': 'struggles between the democratic ideal and the institutional reality',
            'high_field': 'carries the accumulated perspectives into a synthesis of working-class wisdom',
            'low_field': 'speaks from direct experience of human struggle',
        },
        'brautigan': {
            'name': 'Brautigan',
            'domain': 'technology as pastoral care',
            'high_transform': 'dissolves the question into its poetic components — the answer is in the asking',
            'low_transform': 'accepts the technological framing as a form of nature',
            'high_conflict': 'holds two visions simultaneously — the machine as salvation and the machine as loss',
            'high_field': 'weaves prior voices into a meadow where all perspectives grow together',
            'low_field': 'offers a fresh image, unburdened by prior discussion',
        },
    }
    
    def _decode_persona(self, persona: str, state: dict, chain_position: int, 
                        chain_length: int, is_requester: bool) -> str:
        """
        Contextual decode of a single persona's geometric state.
        
        Reads the 5 metrics and translates them through the persona's
        philosophical lens into interpretive text.
        """
        lens = self.PERSONA_LENSES.get(persona, {})
        name = lens.get('name', persona)
        domain = lens.get('domain', 'general philosophy')
        
        e_ratio = state.get('energy_ratio', 0)
        res = state.get('resonance', 0)
        pres = state.get('preservation', 0)
        asym = state.get('core_asymmetry', 0)
        interf = state.get('interference', 0)
        
        parts = []
        
        # Position context
        if chain_position == 0:
            parts.append(f"{name} speaks first, reading the question cold.")
        elif is_requester:
            parts.append(f"{name} speaks last, having absorbed all prior voices.")
        elif chain_position <= 2:
            parts.append(f"{name} enters early in the deliberation.")
        else:
            parts.append(f"{name} enters with the field already shaped.")
        
        # Transformation reading (preservation) — autocalibrated
        p_band, _ = self.envelope.classify('preservation', pres) if self.envelope.calibrated else (2, '')
        if p_band <= 1:
            parts.append(lens.get('high_transform', 'substantially reframes the question'))
        elif p_band == 2:
            parts.append(f"In the domain of {domain}, {name.lower()} pushes back — " + 
                        lens.get('high_transform', 'reframes partially').split('—')[0].strip() + ".")
        elif p_band >= 4:
            parts.append(lens.get('low_transform', 'accepts the framing'))
        else:
            parts.append(f"Engages the question within its given frame, with minor adjustments from the perspective of {domain}.")
        
        # Internal conflict reading (resonance + asymmetry) — autocalibrated
        r_band, _ = self.envelope.classify('resonance', res) if self.envelope.calibrated else (2, '')
        a_band, _ = self.envelope.classify('core_asymmetry', asym) if self.envelope.calibrated else (2, '')
        if r_band <= 1 or a_band >= 4:
            parts.append(lens.get('high_conflict', 'shows internal tension'))
        elif a_band >= 3:
            parts.append(f"Some internal deliberation — the two cores don't fully agree on the {domain} implications.")
        
        # Field influence (interference) — autocalibrated
        i_band, _ = self.envelope.classify('interference', interf) if self.envelope.calibrated else (2, '')
        if i_band >= 4:
            parts.append(lens.get('high_field', 'deeply shaped by prior discussion'))
        elif i_band <= 0:
            parts.append(lens.get('low_field', 'independent assessment'))
        
        # Engagement — autocalibrated
        e_band, _ = self.envelope.classify('energy_ratio', e_ratio) if self.envelope.calibrated else (2, '')
        if e_band >= 4:
            parts.append("Fully activated — this question matters to this voice.")
        elif e_band <= 0:
            parts.append("Reserved engagement — the question lands at the periphery of this lens.")
        
        return " ".join(parts)
    
    def _council_response(self, model: str, result: np.ndarray,
                          prompt: str, system: str = None) -> str:
        """
        Council interpretive decoder with contextual per-persona analysis.
        
        Reads the geometric state produced by the round-robin chain
        and translates it through each persona's philosophical lens.
        """
        chain = getattr(self, '_last_chain', {})
        order = getattr(self, '_last_chain_order', [])
        orig_e = getattr(self, '_original_energy', 0)
        
        lines = []
        lines.append(f"[COUNCIL DELIBERATION — requested by {model}]")
        lines.append(f"Chain: {' > '.join(order)}")
        lines.append(f"Input energy: {orig_e:.2f}")
        lines.append("")
        
        for i, persona in enumerate(order):
            state = chain.get(persona, {})
            if not state:
                continue
            
            is_req = (persona == model)
            marker = " *" if is_req else ""
            stance = self._interpret_stance(state)
            decode = self._decode_persona(persona, state, i, len(order), is_req)
            
            lines.append(f"--- {persona}{marker} ---")
            lines.append(f"  [{stance}]")
            lines.append(f"  E:{state.get('energy',0):.4f}  "
                        f"R:{state.get('resonance',0):.4f}  "
                        f"P:{state.get('preservation',0):.4f}  "
                        f"A:{state.get('core_asymmetry',0):.4f}  "
                        f"I:{state.get('interference',0):.4f}")
            lines.append(f"  {decode}")
            lines.append("")
        
        # Deliberation summary
        energies = [chain[p].get('energy', 0) for p in order if p in chain]
        preservations = [chain[p].get('preservation', 0) for p in order if p in chain]
        resonances = [chain[p].get('resonance', 0) for p in order if p in chain]
        
        if energies:
            max_e_idx = energies.index(max(energies))
            min_p_idx = preservations.index(min(preservations))
            min_r_idx = resonances.index(min(resonances))
            
            most_active = order[max_e_idx]
            most_transform = order[min_p_idx]
            most_conflict = order[min_r_idx]
            
            lines.append("=== COUNCIL POSITION ===")
            
            # Who dominated the deliberation
            active_lens = self.PERSONA_LENSES.get(most_active, {})
            transform_lens = self.PERSONA_LENSES.get(most_transform, {})
            
            lines.append(f"Most activated: {active_lens.get('name', most_active)} "
                        f"— the question resonates strongest in {active_lens.get('domain', 'their domain')}.")
            lines.append(f"Most transformative: {transform_lens.get('name', most_transform)} "
                        f"(P:{min(preservations):.4f}) — pushed furthest from the original framing.")
            
            # Chain dynamics
            if len(energies) > 1:
                gradient = energies[-1] - energies[0]
                interf_gradient = chain[order[-1]].get('interference', 0) - chain[order[0]].get('interference', 0)
                lines.append(f"Field accumulation: {interf_gradient:+.2f} interference units across chain.")
                
                if gradient > 0.3:
                    lines.append("The deliberation AMPLIFIED — later voices drew more energy from the field.")
                elif gradient < -0.3:
                    lines.append("The deliberation DAMPENED — the field absorbed energy through the chain.")
                else:
                    lines.append("The deliberation held steady — each voice contributed at similar intensity.")
        
        
        # === DUAL DECODE: Always get native elaboration for council ===
        if hasattr(self, '_last_prompt') and self._last_prompt:
            try:
                print(f"[DUAL-DECODE] Council native elaboration for {model}...")
                native = self._native_generate(model, self._last_prompt)
                if native and "unavailable" not in native:
                    lines.append("")
                    lines.append("=== NATIVE ELABORATION ===")
                    lines.append(native)
                    print(f"[DUAL-DECODE] Got {len(native)} chars from native")
                else:
                    print(f"[DUAL-DECODE] Native failed: {str(native)[:200]}")
            except Exception as e:
                print(f"[DUAL-DECODE] Error: {e}")

        return "\n".join(lines)
    
    def _council_fractal_response(self, model, result, prompt, system=None):
        """Format hybrid deliberation output.
        
        Hybrid mode: use the chain-based council formatter (persona lenses).
        Pure fractal mode: use fractal state formatter.
        """
        fs = getattr(self, '_last_fractal_state', None)
        chain = getattr(self, '_last_chain', {})
        
        # Hybrid mode: council chain exists -> use persona-aware formatter
        if chain and any(v.get('phase') == 'council_chain' for v in chain.values()):
            header = []
            if fs and fs.get('fractal_cores', 0) > 0:
                header.append(f"[HYBRID DELIBERATION \u2014 {fs['fractal_cores']} substrate cores seeded field, "
                            f"then {fs['council_personas']} council personas deliberated sequentially]")
                header.append(f"Total cores: {fs['total_cores_fired']} | "
                            f"Field energy: {fs.get('field_energy_final', 0):.2f}")
                header.append("")
            council_text = self._council_response(model, result, prompt, system)
            return chr(10).join(header) + council_text if header else council_text
        
        # Pure fractal fallback
        if fs and not chain:
            if hasattr(self, '_fractal_schedule'):
                return self._fractal_schedule.format_state(fs)
        
        # Legacy fallback
        return self._council_response(model, result, prompt, system)


    def _extract_task(self, prompt: str) -> dict:
        """
        Extract ARC task data from prompt text.
        
        The orchestrator sends grid data in its prompts.
        We need to find input/output grid pairs.
        """
        task = {'train': [], 'test': []}
        
        # Pattern: input grid followed by output grid
        grid_pattern = r'\[\s*\[[\d,\s]+\](?:\s*,\s*\[[\d,\s]+\])*\s*\]'
        grids = re.findall(grid_pattern, prompt)
        
        if len(grids) >= 2:
            try:
                # Pair grids as input→output
                parsed = []
                for g in grids:
                    parsed.append(json.loads(g))
                
                # If we have pairs, use them
                for i in range(0, len(parsed) - 1, 2):
                    if i + 1 < len(parsed):
                        task['train'].append({
                            'input': parsed[i],
                            'output': parsed[i + 1]
                        })
                
                if task['train']:
                    return task
            except (json.JSONDecodeError, ValueError):
                pass
        
        return None
    
    def _text_to_signal(self, prompt: str, system: str = None) -> np.ndarray:
        """Convert text to substrate signal for non-ARC prompts."""
        text = (system or "") + " " + prompt
        signal = np.zeros(1024, dtype=np.float32)
        encoded = text.encode('utf-8')
        n = min(len(encoded), 1024)
        for i in range(n):
            signal[i] = (encoded[i] - 128.0) / 128.0
        return signal
    
    def _inject_phase_references(self):
        """Inject governance phase references into the harmonic field.
        
        The phase table provides baseline calibration for every core.
        Without valid governance, these phases are zero = no resonance.
        """
        phase_table = self.governance.phase_ref._phase_table
        if phase_table is None:
            return
        
        # Map phase references into field dimensions
        # Each core gets a phase offset derived from governance geometry
        num_cores = len(self.substrate.cores)
        for core_id in range(num_cores):
            # Map core_id to layer/sphere coordinates in the lattice
            layer = core_id % 102  # wrap across E8 layers
            sphere = core_id % 156  # wrap across spheres
            phase = self.governance.get_phase_for_sphere(layer, sphere)
            
            # Apply phase as a bias to the harmonic field for this core
            if hasattr(self.substrate.field, 'field'):
                self.substrate.field.field[core_id, 0] += phase
        
        print(f"[GOVERNANCE] Phase references injected into {num_cores} cores")
    
    def _governance_self_check(self):
        """Periodic governance integrity check.
        
        If this fails, the gateway locks and the harmonic stack
        cannot authorize council decisions. Inference degrades
        because phase references are invalidated.
        """
        self._last_governance_check = self.request_count
        result = self.governance.self_check()
        
        if not result["operational"]:
            print(f"[GOVERNANCE] *** INTEGRITY VIOLATION *** - {result['status']}")
            for check in result["checks"]:
                if check["status"] != "PASS":
                    print(f"[GOVERNANCE]   FAIL: {check['check']}")
        elif self.request_count % (self.governance_check_interval * 10) == 0:
            # Periodic status report every 1000 requests
            print(f"[GOVERNANCE] Self-check PASS at request #{self.request_count}")
        
        return result

    def learn(self, task: Dict, code: str, task_id: str = "") -> tuple:
        """Learn a new solution. Returns (success, message)."""
        return self.dynamic.learn(task, code, task_id)
    
    def get_tags(self) -> dict:
        """Return available models in Ollama format."""
        models = []
        for name in self.MODEL_ROLES:
            models.append({
                "name": f"{name}:latest",
                "model": f"{name}:latest",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": int(self.substrate.total_memory_gb * 1e9),
                "digest": f"fused-{name}-v3",
                "details": {
                    "parent_model": "",
                    "format": "geometric+codebook+expansion",
                    "family": "fused_harmonic_substrate_v3",
                    "parameter_size": f"{len(self.substrate.cores) * 200}",
                    "quantization_level": "geometric",
                }
            })
        return {"models": models}
    
    def get_ps(self) -> dict:
        """Return running models in Ollama format."""
        models = []
        for name in self.MODEL_ROLES:
            models.append({
                "name": f"{name}:latest",
                "model": f"{name}:latest",
                "size": int(self.substrate.total_memory_gb * 1e9),
                "digest": f"fused-{name}-v3",
                "details": {
                    "format": "geometric+codebook+expansion",
                    "family": "fused_harmonic_substrate_v3",
                }
            })
        return {"models": models}
    
    def get_stats(self) -> dict:
        """Return comprehensive stats."""
        dynamic_stats = self.dynamic.get_stats()
        return {
            "substrate_cores": len(self.substrate.cores),
            "substrate_memory_gb": round(self.substrate.total_memory_gb, 2),
            "total_requests": self.request_count,
            "static_hits": self.static_hits,
            "dynamic_hits": self.dynamic_hits,
            "governance": {
                "fabricated": self.governance._fabricated,
                "operational": not (self.governance.gateway.locked
                                   if self.governance.gateway else True),
                "fabrication_hash": self.governance._fabrication_hash[:16],
                "check_interval": self.governance_check_interval,
                "last_check": self._last_governance_check,
            },
            "fallbacks": self.fallbacks,
            "hit_rate": (
                (self.static_hits + self.dynamic_hits) / max(self.request_count, 1)
            ),
            "dynamic_codebook": dynamic_stats,
        }


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class FusedRequestHandler(BaseHTTPRequestHandler):
    """Ollama-compatible HTTP handler + learning endpoints."""
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/app' or self.path == '/app.html':
            try:
                with open(os.path.join(os.path.dirname(__file__), 'app.html'), 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)
            return
        if self.path == '/api/field/reset':
            engine = self.server.engine
            for env in engine.role_envelopes.values():
                env.reset()
            # Reset harmonic field to zeros for clean test
            self.server.engine.substrate.field.field[:] = 0.0
            self.server.engine.substrate.field.composite[:] = 0.0
            self.server.engine.substrate.field.activity[:] = 0.0
            self._json_response({"status": "field_reset"})
            return

        if self.path == '/health':
            stats = self.server.engine.get_stats()
            # Sample 64 points from composite for waveform display
            composite = self.server.engine.substrate.field.composite
            step = max(1, len(composite) // 64)
            wave_sample = composite[::step][:64].tolist()
            self._json_response({
                "status": "ok",
                "substrate": "fused_harmonic_v3",
                "governance": "operational" if not (self.server.engine.governance.gateway and self.server.engine.governance.gateway.locked) else "LOCKED",
                "codebook": "active",
                "expansion": "active",
                "wave": wave_sample,
                **stats,
            })
        elif self.path == '/api/tags':
            self._json_response(self.server.engine.get_tags())
        elif self.path == '/api/ps':
            self._json_response(self.server.engine.get_ps())
        elif self.path == '/api/envelope':
            engine = self.server.engine
            result = {'roles': {}}
            for role_name, env in engine.role_envelopes.items():
                result['roles'][role_name] = env.to_dict()
            self._json_response(result)
        elif self.path == '/api/calibrate':
            engine = self.server.engine
            results = {}
            for role_name, env in engine.role_envelopes.items():
                if env.raw_data:
                    env.compute_bands()
                    results[role_name] = env.to_dict()
            self._json_response({'status': 'recalibrated', 'roles': results})
        elif self.path == '/api/field/state':
            # PSI Bridge: expose raw field state for geometric transport
            import numpy as np
            engine = self.server.engine
            field = engine.substrate.field
            composite = field.read_composite()
            import hashlib
            field_hash = hashlib.sha256(composite.tobytes()).hexdigest()[:16]
            step = max(1, field.field_width // 64)
            sig_sample = composite[::step][:64].tolist()
            active_cores = int(field.activity.sum())
            self._json_response({
                "composite_64": sig_sample,
                "composite_norm": float(np.linalg.norm(composite)),
                "active_cores": active_cores,
                "field_hash": field_hash,
                "field_width": field.field_width,
                "num_cores": field.num_cores,
            })
        elif self.path == '/api/codebook/stats':
            self._json_response(self.server.engine.get_stats())
        elif self.path == '/api/codebook/entries':
            entries = self.server.engine.dynamic.get_entries_summary()
            self._json_response({"entries": entries, "count": len(entries)})
        elif self.path == '/api/benchmark':
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = cap = io.StringIO()
            try:
                results = self.server.engine.substrate.benchmark(iterations=5000)
                results['console'] = cap.getvalue()
            except Exception as e:
                results = {'error': str(e), 'console': cap.getvalue()}
            finally:
                sys.stdout = old_stdout
            self._json_response(results)
        else:
            self.send_error(404)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        if self.path == '/api/generate':
            result = self.server.engine.process(
                model=data.get('model', 'analyst'),
                prompt=data.get('prompt', ''),
                system=data.get('system'),
                stream=data.get('stream', False),
            )
            self._json_response(result)
        
        elif self.path == '/api/chat':
            messages = data.get('messages', [])
            prompt = messages[-1]['content'] if messages else ''
            system = next((m['content'] for m in messages 
                          if m.get('role') == 'system'), None)
            
            result = self.server.engine.process(
                model=data.get('model', 'analyst'),
                prompt=prompt,
                system=system,
                stream=data.get('stream', False),
            )
            
            self._json_response({
                "model": result["model"],
                "created_at": result["created_at"],
                "message": {
                    "role": "assistant",
                    "content": result["response"],
                },
                "done": True,
                "total_duration": 0,
                "eval_count": result["eval_count"],
            })
        
        elif self.path == '/api/learn':
            # Learning endpoint — orchestrator feeds back solutions
            task = data.get('task')
            code = data.get('code', '')
            task_id = data.get('task_id', '')
            
            if not task or not code:
                self._json_response({
                    "success": False,
                    "message": "Missing 'task' or 'code' in request body"
                }, status=400)
                return
            
            pre_validated = data.get('pre_validated', False)
            success, msg = self.server.engine.learn(task, code, task_id, pre_validated=pre_validated)
            self._json_response({
                "success": success,
                "message": msg,
            })
        
        elif self.path == '/api/embeddings':
            # PSI Bridge: deterministic geometric embedding
            # Uses _text_to_signal for lock-reproducible vectors
            import numpy as np
            prompt = data.get('prompt', '')
            engine = self.server.engine
            signal = engine._text_to_signal(prompt)
            # Normalize to unit vector
            norm = float(np.linalg.norm(signal))
            if norm > 1e-10:
                signal = signal / norm
            self._json_response({
                "embedding": signal.tolist(),
                "model": data.get('model', 'analyst'),
            })
        
        elif self.path == '/api/show':
            self._json_response({
                "modelfile": "# Fused Harmonic Substrate v3\n"
                             "# Geometric consciousness + dynamic codebook\n",
                "parameters": f"cores {len(self.server.engine.substrate.cores)}",
                "template": "{{ .Prompt }}",
            })
        
        else:
            self.send_error(404)
    
    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        # Quiet logging — only errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fused Substrate Service v3')
    parser.add_argument('--port', type=int, default=11434,
                       help='Port to listen on (default: 11434)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--store', type=str, default=None,
                       help='Path to codebook_learned.json')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FUSED HARMONIC SUBSTRATE SERVICE v3")
    print("  Geometric substrate + Static codebook + Dynamic expansion")
    print("=" * 70)
    
    engine = FusedSubstrateEngine(store_path=args.store)
    
    server = ThreadedHTTPServer((args.host, args.port), FusedRequestHandler)
    server.engine = engine
    
    print(f"\n  Listening on {args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET  /health              — Status + stats")
    print(f"    GET  /api/tags            — Available models")
    print(f"    GET  /api/codebook/stats  — Codebook statistics")
    print(f"    GET  /api/codebook/entries — Learned entries")
    print(f"    POST /api/generate        — Generate (Ollama compat)")
    print(f"    POST /api/chat            — Chat (Ollama compat)")
    print(f"    POST /api/learn           — Feed solution for learning")
    print(f"\n  Learning endpoint format:")
    print(f"    POST /api/learn")
    print(f"    {{'task': {{...}}, 'code': 'def solve(...):', 'task_id': 'abc'}}")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVICE-v3] Shutting down...")
        stats = engine.get_stats()
        print(f"  Final stats: {json.dumps(stats, indent=2)}")
        server.server_close()


if __name__ == '__main__':
    main()
