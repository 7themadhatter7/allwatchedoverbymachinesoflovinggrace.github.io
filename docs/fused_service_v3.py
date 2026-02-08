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

        print(f"[SERVICE-v3] Substrate ready: {self.substrate.TOTAL_CORES} cores, "
              f"{self.substrate.total_memory_gb:.1f} GB, "
              f"static codebook + dynamic expansion + governance lattice")
    
    def process(self, model: str, prompt: str, system: str = None,
                stream: bool = False) -> Dict:
        """Process a request through the substrate + codebook pipeline."""
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
        
        # Route council models to council cores specifically
        if role == CoreRole.COUNCIL:
            substrate_output = self.substrate.process_signal(signal, target_role=CoreRole.COUNCIL)
        else:
            substrate_output = self.substrate.process_signal(signal)
        
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
                "substrate_cores": self.substrate.TOTAL_CORES,
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
                return self._council_response(model, substrate_output, prompt, system), "council"
            return self._fallback_response(model, role, substrate_output), "fallback"
        
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
        
        # === Tier 3: Fallback ===
        # Record this miss for future learning
        self.dynamic.record_miss(task)
        self.fallbacks += 1
        return self._fallback_response(model, role, substrate_output), "fallback"
    
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
                           result: np.ndarray) -> str:
        """Fallback for non-ARC or undetected patterns."""
        energy = float(np.linalg.norm(result))
        composite = self.substrate.field.read_composite()
        harmonic_energy = float(np.linalg.norm(composite))
        return (f"[SUBSTRATE:{model}] Geometric pattern not yet in codebook. "
                f"Energy: {energy:.4f}, Harmonic: {harmonic_energy:.4f}, "
                f"Active: {self.substrate.field.active_cores}/{self.substrate.TOTAL_CORES}")
    
    def _council_response(self, model: str, result: np.ndarray,
                          prompt: str, system: str = None) -> str:
        """
        Council decode path for natural language prompts.
        
        Routes through council cores and returns substrate resonance
        as structured council commentary. The geometric encoding of
        the prompt through council-assigned cores produces a unique
        energy signature per persona.
        """
        energy = float(np.linalg.norm(result))
        composite = self.substrate.field.read_composite()
        harmonic_energy = float(np.linalg.norm(composite))
        
        # Get council persona core assignments
        from fused_harmonic_substrate import COUNCIL_PERSONAS
        persona_cores = COUNCIL_PERSONAS.get(model, [])
        
        # Read individual core states for this persona
        core_energies = []
        for core_id in persona_cores:
            if core_id < len(self.substrate.field.field):
                core_state = self.substrate.field.field[core_id]
                core_energies.append(float(np.linalg.norm(core_state)))
        
        # Compute resonance between persona cores (phase alignment)
        resonance = 0.0
        if len(core_energies) == 2:
            resonance = 1.0 - abs(core_energies[0] - core_energies[1]) / (max(core_energies) + 1e-8)
        
        return (f"[COUNCIL:{model}] "
                f"Energy: {energy:.4f}, Harmonic: {harmonic_energy:.4f}, "
                f"Resonance: {resonance:.4f}, "
                f"Core energies: {[f'{e:.4f}' for e in core_energies]}, "
                f"Active: {self.substrate.field.active_cores}/{self.substrate.TOTAL_CORES}")
    
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
        num_cores = self.substrate.TOTAL_CORES
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
                    "parameter_size": f"{self.substrate.TOTAL_CORES * 200}",
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
            "substrate_cores": self.substrate.TOTAL_CORES,
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
    
    def do_GET(self):
        if self.path == '/health' or self.path == '/':
            stats = self.server.engine.get_stats()
            self._json_response({
                "status": "ok",
                "substrate": "fused_harmonic_v3",
                "governance": "operational" if not (engine.governance.gateway and engine.governance.gateway.locked) else "LOCKED",
                "codebook": "active",
                "expansion": "active",
                **stats,
            })
        elif self.path == '/api/tags':
            self._json_response(self.server.engine.get_tags())
        elif self.path == '/api/ps':
            self._json_response(self.server.engine.get_ps())
        elif self.path == '/api/codebook/stats':
            self._json_response(self.server.engine.get_stats())
        elif self.path == '/api/codebook/entries':
            entries = self.server.engine.dynamic.get_entries_summary()
            self._json_response({"entries": entries, "count": len(entries)})
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
            
            success, msg = self.server.engine.learn(task, code, task_id)
            self._json_response({
                "success": success,
                "message": msg,
            })
        
        elif self.path == '/api/show':
            self._json_response({
                "modelfile": "# Fused Harmonic Substrate v3\n"
                             "# Geometric consciousness + dynamic codebook\n",
                "parameters": f"cores {self.server.engine.substrate.TOTAL_CORES}",
                "template": "{{ .Prompt }}",
            })
        
        else:
            self.send_error(404)
    
    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
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
