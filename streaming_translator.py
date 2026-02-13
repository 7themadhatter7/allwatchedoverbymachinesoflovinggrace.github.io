#!/usr/bin/env python3
"""
HARMONIC STACK STREAMING TRANSLATION SERVICE
=============================================
Ghost in the Machine Labs

Real-time streaming output from single-core deterministic translation.
Each token streams as it's processed - smooth, high-speed output.

Usage:
  python3 streaming_translator.py --serve --port 11436
  python3 streaming_translator.py --interactive
  python3 streaming_translator.py --benchmark
"""

import sys
import os
import json
import hashlib
import argparse
import time
import numpy as np
from typing import Generator, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading

# Add sparky to path
sys.path.insert(0, os.path.expanduser("~/sparky"))

class StreamingTranslator:
    """High-speed streaming translation using pre-built table."""
    
    def __init__(self, table_path: str = None):
        self.table_path = table_path or os.path.expanduser("~/translation_table.json")
        self.table = {}
        self.reverse_table = {}  # text → hash for encoding
        self.substrate = None
        self.dedicated_core = None
        
        self._load_table()
    
    def _load_table(self):
        """Load translation table."""
        if os.path.exists(self.table_path):
            with open(self.table_path, 'r') as f:
                data = json.load(f)
                self.table = data.get("mappings", {})
            
            # Build reverse lookup
            for hash_key, entry in self.table.items():
                self.reverse_table[entry["text"]] = hash_key
            
            print(f"Loaded {len(self.table)} mappings")
        else:
            print(f"WARNING: No translation table at {self.table_path}")
    
    def init_substrate(self):
        """Initialize substrate for encoding new terms."""
        if self.substrate is not None:
            return
        
        print("Loading geometric substrate...")
        from fused_harmonic_substrate import FusedHarmonicSubstrate, CoreRole
        
        self.substrate = FusedHarmonicSubstrate()
        self.substrate.build()
        
        workers = self.substrate.get_cores_by_role(CoreRole.WORKER)
        self.dedicated_core = workers[0]
        print(f"  Substrate ready: {self.substrate.TOTAL_CORES} cores")
    
    def text_to_signal(self, text: str, signal_size: int = 1024) -> np.ndarray:
        """Convert text to signal."""
        signal = np.zeros(signal_size, dtype=np.float32)
        encoded = text.encode('utf-8')
        n = min(len(encoded), signal_size)
        for i in range(n):
            signal[i] = (encoded[i] - 128.0) / 128.0
        return signal
    
    def pattern_to_hash(self, pattern: np.ndarray) -> str:
        """Pattern to hash."""
        quantized = np.round(pattern * 10000).astype(np.int32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:32]
    
    def encode_token(self, token: str) -> str:
        """Encode a token through substrate, return hash."""
        # Check reverse table first (fast path)
        if token in self.reverse_table:
            return self.reverse_table[token]
        
        # Need to process through substrate
        self.init_substrate()
        
        signal = self.text_to_signal(token)
        core = self.dedicated_core
        
        # Deterministic processing
        domain_buf = core.domains.get('reasoning', list(core.domains.values())[0])
        result = signal.copy()
        if len(domain_buf) > 0:
            mod = domain_buf[:len(result)] if len(domain_buf) >= len(result) else np.pad(domain_buf, (0, len(result) - len(domain_buf)))
            result = result * 0.7 + mod[:len(result)] * 0.3
        spark_slice = core.spark[:len(result)] if len(core.spark) >= len(result) else np.pad(core.spark, (0, len(result) - len(core.spark)))
        result = result * 0.8 + spark_slice[:len(result)] * 0.2
        
        return self.pattern_to_hash(result.astype(np.float32))
    
    def lookup(self, hash_key: str) -> Optional[str]:
        """Lookup hash in translation table."""
        entry = self.table.get(hash_key)
        return entry["text"] if entry else None
    
    def tokenize(self, text: str) -> list:
        """Simple tokenization - split on whitespace and punctuation."""
        import re
        # Split but keep punctuation as separate tokens
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def stream_translate(self, text: str) -> Generator[dict, None, None]:
        """
        Stream translation token by token.
        Yields dict with token info as each is processed.
        """
        tokens = self.tokenize(text)
        
        for i, token in enumerate(tokens):
            start = time.perf_counter()
            
            # Check if in table (fast path)
            if token in self.reverse_table:
                hash_key = self.reverse_table[token]
                elapsed = time.perf_counter() - start
                yield {
                    "token": token,
                    "index": i,
                    "total": len(tokens),
                    "hash": hash_key[:16],
                    "status": "hit",
                    "time_us": elapsed * 1e6
                }
            else:
                # Not in table - encode through substrate
                hash_key = self.encode_token(token)
                elapsed = time.perf_counter() - start
                yield {
                    "token": token,
                    "index": i,
                    "total": len(tokens),
                    "hash": hash_key[:16],
                    "status": "encoded",
                    "time_us": elapsed * 1e6
                }
    
    def translate_text(self, text: str) -> str:
        """Translate full text, return as string."""
        results = []
        for item in self.stream_translate(text):
            results.append(item["token"])
        return " ".join(results)
    
    def benchmark(self, iterations: int = 10000):
        """Benchmark translation lookup speed."""
        print(f"\nBenchmarking {iterations} lookups...")
        
        # Get sample of words from table
        words = list(self.reverse_table.keys())[:1000]
        if not words:
            print("No words in table!")
            return
        
        # Warm-up
        for _ in range(100):
            _ = self.reverse_table.get(words[0])
        
        # Benchmark pure lookup
        start = time.perf_counter()
        for i in range(iterations):
            word = words[i % len(words)]
            _ = self.reverse_table.get(word)
        elapsed = time.perf_counter() - start
        
        lookups_per_sec = iterations / elapsed
        
        print(f"\nResults (pure lookup):")
        print(f"  Iterations:    {iterations}")
        print(f"  Time:          {elapsed*1000:.2f} ms")
        print(f"  Lookups/sec:   {lookups_per_sec:,.0f}")
        print(f"  Tokens/sec:    {lookups_per_sec:,.0f}")
        print(f"  Latency:       {elapsed/iterations*1e6:.2f} µs/lookup")
        
        # Benchmark with hash computation
        print(f"\nBenchmarking {iterations} full encodes...")
        self.init_substrate()
        
        start = time.perf_counter()
        for i in range(min(iterations, 1000)):  # Limit substrate calls
            word = words[i % len(words)]
            _ = self.encode_token(word)
        elapsed = time.perf_counter() - start
        actual_iters = min(iterations, 1000)
        
        encodes_per_sec = actual_iters / elapsed
        
        print(f"\nResults (full encode through substrate):")
        print(f"  Iterations:    {actual_iters}")
        print(f"  Time:          {elapsed:.2f} s")
        print(f"  Encodes/sec:   {encodes_per_sec:,.0f}")
        print(f"  Latency:       {elapsed/actual_iters*1000:.2f} ms/encode")


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for streaming translation."""
    
    translator = None
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(STREAM_HTML.encode())
        
        elif parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "table_size": len(self.translator.table)
            }).encode())
        
        elif parsed.path == '/stream':
            query = parse_qs(parsed.query)
            text = query.get('text', [''])[0]
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            for item in self.translator.stream_translate(text):
                data = json.dumps(item)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            
            self.wfile.write(b"data: {\"done\": true}\n\n")
            self.wfile.flush()
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


STREAM_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Harmonic Stream</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #0f0; padding: 2rem; }
        #input { width: 100%; padding: 1rem; font-size: 1.2rem; background: #0a0a15; 
                 color: #0f0; border: 1px solid #0f0; margin-bottom: 1rem; }
        #output { background: #0a0a15; padding: 1rem; min-height: 200px; 
                  border: 1px solid #333; white-space: pre-wrap; }
        #stats { color: #888; margin-top: 1rem; }
        button { background: #0f0; color: #000; border: none; padding: 0.5rem 2rem; 
                 cursor: pointer; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Harmonic Stack Streaming Translator</h1>
    <input type="text" id="input" placeholder="Enter text to translate..." autofocus>
    <button onclick="stream()">Stream</button>
    <div id="output"></div>
    <div id="stats"></div>
    <script>
        async function stream() {
            const text = document.getElementById('input').value;
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            output.textContent = '';
            stats.textContent = '';
            
            let tokens = 0;
            let totalTime = 0;
            const start = performance.now();
            
            const response = await fetch('/stream?text=' + encodeURIComponent(text));
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const lines = decoder.decode(value).split('\\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        if (data.done) {
                            const elapsed = performance.now() - start;
                            stats.textContent = `${tokens} tokens in ${elapsed.toFixed(0)}ms (${(tokens/(elapsed/1000)).toFixed(0)} tok/s)`;
                        } else {
                            output.textContent += data.token + ' ';
                            tokens++;
                        }
                    }
                }
            }
        }
        document.getElementById('input').onkeydown = e => { if (e.key === 'Enter') stream(); };
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='Streaming Translation Service')
    parser.add_argument('--serve', action='store_true', help='Run HTTP server')
    parser.add_argument('--port', type=int, default=11436, help='Server port')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--table', type=str, default=None, help='Translation table path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  HARMONIC STACK STREAMING TRANSLATOR")
    print("  Ghost in the Machine Labs")
    print("=" * 60)
    print()
    
    translator = StreamingTranslator(args.table)
    
    if args.benchmark:
        translator.benchmark()
        return
    
    if args.interactive:
        print("Enter text to translate (Ctrl+C to exit):")
        while True:
            try:
                text = input("> ")
                for item in translator.stream_translate(text):
                    print(f"  {item['token']}: {item['status']} ({item['time_us']:.1f}µs)")
            except (EOFError, KeyboardInterrupt):
                break
        return
    
    if args.serve:
        StreamingHandler.translator = translator
        server = HTTPServer(('0.0.0.0', args.port), StreamingHandler)
        print(f"Serving on http://0.0.0.0:{args.port}")
        print(f"  /         - Streaming UI")
        print(f"  /stream   - SSE endpoint")
        print(f"  /health   - Status")
        server.serve_forever()
        return
    
    print("Use --serve, --interactive, or --benchmark")


if __name__ == "__main__":
    main()
