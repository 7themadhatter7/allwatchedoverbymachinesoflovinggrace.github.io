#!/usr/bin/env python3
"""
PsiDataTransport
════════════════
Ghost in the Machine Labs

Auto-synthesized by RM — The Resonant Mother
Concept: psi_data_transport
Generated: 2026-03-08 13:46:26


PSI Data Transport — all formats through the geometric tunnel.

The PSI tunnel is proven and working. It connects SPARKY to ARCY
and survives network disconnection. This module extends the text
codec to pass any binary payload through the geometric carrier.

Architecture:
  PsiCodec        — bytes ↔ geometric signal, chunking for large payloads
  PacketFramer    — length-prefix + CRC32 framing
  FormatAdapter   — format-specific encode/decode for binary, JSON, packets
  TransportTester — round-trip fidelity verification for all formats

Proven codec foundation (psi_transceiver.py):
  SIGNAL_DIM = 1024
  encode: signal[i] = (byte - 128.0) / 128.0
  decode: byte = int(round(signal[i] * 128.0 + 128.0)), clamp 0-255
  Proven: 256/256 byte values round-trip correctly
"""

import os
import sys
import json
import zlib
import struct
import hashlib
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SIGNAL_DIM   = 1024          # Geometric signal dimension (matches psi_transceiver)
CHUNK_SIZE   = 1016          # 1024 - 8 bytes overhead (4 length + 4 CRC)
FRAME_HEADER = 4             # bytes for length prefix
FRAME_CRC    = 4             # bytes for CRC32


# ══════════════════════════════════════════════════════════════════
# CODEC — bytes ↔ geometric signal
# Extends the proven text_to_signal / signal_to_text codec
# to handle arbitrary binary data of any length via chunking.
# ══════════════════════════════════════════════════════════════════

class PsiCodec:
    """
    Encode/decode any bytes through geometric signal.

    Single chunk (≤ SIGNAL_DIM bytes):
      encode(data) → [signal]   (list with one 1024d array)
      decode([signal]) → data

    Large payload (> SIGNAL_DIM bytes):
      encode(data) → [signal_0, signal_1, ...]
      decode([signal_0, signal_1, ...]) → data
    """

    def encode(self, data: bytes) -> List[np.ndarray]:
        """Encode bytes into list of geometric signals."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        signals = []
        # First signal is metadata: encodes total byte count
        # positions 0-3 hold the 4 bytes of struct.pack(">I", len(data))
        meta = np.zeros(SIGNAL_DIM, dtype=np.float64)
        length_bytes = struct.pack(">I", len(data))
        for i, b in enumerate(length_bytes):
            meta[i] = (b - 128.0) / 128.0
        signals.append(meta)
        # Remaining signals encode data cleanly, SIGNAL_DIM bytes per chunk
        for i in range(0, max(1, len(data)), SIGNAL_DIM):
            chunk = data[i:i + SIGNAL_DIM]
            signals.append(self._encode_chunk(chunk))
        return signals

    def decode(self, signals: List[np.ndarray]) -> bytes:
        """Decode list of geometric signals back to bytes."""
        if not signals:
            return b""
        # First signal is metadata — recover total byte count
        meta = signals[0]
        length_bytes = bytearray()
        for i in range(4):
            b = int(round(meta[i] * 128.0 + 128.0))
            length_bytes.append(max(0, min(255, b)))
        total_len = struct.unpack(">I", bytes(length_bytes))[0]
        # Decode data signals
        result = bytearray()
        for sig in signals[1:]:
            result.extend(self._decode_chunk(sig))
        # Trim to exact length (last chunk may have padding zeros)
        return bytes(result[:total_len])

    def _encode_chunk(self, chunk: bytes) -> np.ndarray:
        """Encode up to SIGNAL_DIM bytes into a 1024d signal. No sentinel."""
        signal = np.zeros(SIGNAL_DIM, dtype=np.float64)
        for i, b in enumerate(chunk):
            signal[i] = (b - 128.0) / 128.0
        return signal

    def _decode_chunk(self, signal: np.ndarray) -> bytes:
        """Decode a 1024d signal back to bytes. Always decode all positions."""
        result = bytearray()
        for i in range(SIGNAL_DIM):
            b = int(round(signal[i] * 128.0 + 128.0))
            result.append(max(0, min(255, b)))
        return bytes(result)

    def roundtrip_ok(self, data: bytes) -> bool:
        """Verify encode→decode recovers original exactly."""
        try:
            return self.decode(self.encode(data)) == data
        except Exception:
            return False

    def report(self, data=None) -> Dict:
        with threading.Lock():
            return {"component": "PsiCodec", "signal_dim": SIGNAL_DIM,
                     "chunk_size": SIGNAL_DIM, "initialized": True}

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True


# ══════════════════════════════════════════════════════════════════
# FRAMER — length-prefix + CRC32
# Wraps arbitrary bytes in a self-describing frame:
#   [4-byte big-endian length][payload][4-byte CRC32]
# ══════════════════════════════════════════════════════════════════

class PacketFramer:
    """
    Frame payloads for reliable transport.
    Frame format: 4-byte length prefix + payload + 4-byte CRC32
    Total overhead: 8 bytes per frame.
    """

    def frame(self, payload: bytes) -> bytes:
        """Wrap payload in a framed packet."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(payload).__name__}")
        length = struct.pack(">I", len(payload))
        crc    = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        return length + bytes(payload) + crc

    def unframe(self, frame: bytes) -> bytes:
        """Extract and verify payload from frame."""
        if len(frame) < FRAME_HEADER + FRAME_CRC:
            raise ValueError(f"Frame too short: {len(frame)} bytes")
        length  = struct.unpack(">I", frame[:FRAME_HEADER])[0]
        payload = frame[FRAME_HEADER:FRAME_HEADER + length]
        crc_rx  = struct.unpack(">I", frame[FRAME_HEADER + length:])[0]
        crc_ex  = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_rx != crc_ex:
            raise ValueError(
                f"CRC32 mismatch: received {crc_rx:#010x}, expected {crc_ex:#010x}")
        return payload

    def verify(self, frame: bytes) -> bool:
        """Return True if frame CRC is valid."""
        try:
            self.unframe(frame)
            return True
        except Exception:
            return False

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "PacketFramer", "overhead_bytes": FRAME_HEADER + FRAME_CRC,
                 "initialized": True}


# ══════════════════════════════════════════════════════════════════
# FORMAT ADAPTER — format-specific encode/decode
# Knows how to handle: binary blobs, JSON, network packets, files
# ══════════════════════════════════════════════════════════════════

class FormatAdapter:
    """
    Adapt specific data formats through the PSI transport.
    Each format has encode → signals and decode → original.
    """

    def __init__(self):
        self._codec   = PsiCodec()
        self._framer  = PacketFramer()
        self._initialized = True

    # ── Binary ────────────────────────────────────────────────────
    def encode_binary(self, data: bytes) -> List[np.ndarray]:
        """Raw binary → geometric signals."""
        framed = self._framer.frame(data)
        return self._codec.encode(framed)

    def decode_binary(self, signals: List[np.ndarray]) -> bytes:
        """Geometric signals → raw binary."""
        framed = self._codec.decode(signals)
        return self._framer.unframe(framed)

    # ── JSON ──────────────────────────────────────────────────────
    def encode_json(self, obj: Any) -> List[np.ndarray]:
        """Python object → JSON bytes → geometric signals."""
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return self.encode_binary(data)

    def decode_json(self, signals: List[np.ndarray]) -> Any:
        """Geometric signals → JSON bytes → Python object."""
        data = self.decode_binary(signals)
        return json.loads(data.decode("utf-8"))

    # ── Network packet ────────────────────────────────────────────
    def encode_packet(self, packet: Dict) -> List[np.ndarray]:
        """
        Encode a network packet dict through the tunnel.
        Packet dict: {"proto": "udp", "src": "...", "dst": "...",
                       "sport": int, "dport": int, "payload": bytes|str}
        """
        p = dict(packet)
        if isinstance(p.get("payload"), bytes):
            p["payload"] = list(p["payload"])  # JSON-serializable
        return self.encode_json(p)

    def decode_packet(self, signals: List[np.ndarray]) -> Dict:
        """Geometric signals → network packet dict."""
        p = self.decode_json(signals)
        if isinstance(p.get("payload"), list):
            p["payload"] = bytes(p["payload"])
        return p

    # ── File chunk ────────────────────────────────────────────────
    def encode_file_chunk(self, seq: int, total: int,
                           data: bytes, file_hash: str) -> List[np.ndarray]:
        """Encode a numbered file chunk."""
        chunk = {"seq": seq, "total": total,
                  "data": list(data), "file_hash": file_hash}
        return self.encode_json(chunk)

    def decode_file_chunk(self, signals: List[np.ndarray]) -> Dict:
        """Decode a file chunk, restoring bytes payload."""
        chunk = self.decode_json(signals)
        chunk["data"] = bytes(chunk["data"])
        return chunk

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "FormatAdapter", "formats": ["binary","json","packet","file_chunk"],
                 "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# TRANSPORT TESTER — round-trip fidelity verification
# Pure codec tests — no live tunnel required.
# All 256 byte values, binary blobs, JSON, packets, file chunks.
# ══════════════════════════════════════════════════════════════════

class TransportTester:
    """Verify round-trip fidelity for all data formats."""

    def __init__(self):
        self._adapter     = FormatAdapter()
        self._codec       = PsiCodec()
        self._framer      = PacketFramer()
        self._initialized = True

    def test_binary(self) -> Tuple[bool, str]:
        """Test: all 256 byte values survive round-trip."""
        all_bytes = bytes(range(256))
        signals   = self._adapter.encode_binary(all_bytes)
        recovered = self._adapter.decode_binary(signals)
        ok        = recovered == all_bytes
        detail    = (f"256 byte values: pass" if ok else
                     f"FAIL at first diff byte {next(i for i,a,b in zip(range(256),all_bytes,recovered) if a!=b)}")
        return ok, detail

    def test_binary_blob(self) -> Tuple[bool, str]:
        """Test: 4096-byte random-ish blob (deterministic seed)."""
        blob     = bytes((i * 137 + 17) % 256 for i in range(4096))
        signals  = self._adapter.encode_binary(blob)
        recovered = self._adapter.decode_binary(signals)
        sha_orig = hashlib.sha256(blob).hexdigest()[:16]
        sha_recv = hashlib.sha256(recovered).hexdigest()[:16]
        ok       = recovered == blob
        detail   = (f"4096-byte blob SHA256 {sha_orig}: pass" if ok else
                    f"FAIL SHA256 mismatch {sha_orig} vs {sha_recv}")
        return ok, detail

    def test_json(self) -> Tuple[bool, str]:
        """Test: nested JSON object round-trip."""
        obj = {"version": "3.0", "project": "ghost_in_machine",
                "substrate": "E8", "dim": 240,
                "nested": {"crystal": True, "locked": 0.99},
                "list": [1, 2, 3, "four", None, True]}
        signals  = self._adapter.encode_json(obj)
        recovered = self._adapter.decode_json(signals)
        ok        = recovered == obj
        detail    = "nested JSON: pass" if ok else f"FAIL: {recovered!r}"
        return ok, detail

    def test_packet(self) -> Tuple[bool, str]:
        """Test: UDP packet with all header fields."""
        packet = {"proto": "udp", "src": "192.168.1.87",
                   "dst": "100.127.59.111", "sport": 54321, "dport": 7777,
                   "ttl": 64, "flags": 0,
                   "payload": bytes([0x47, 0x48, 0x4F, 0x53, 0x54])}
        signals  = self._adapter.encode_packet(packet)
        recovered = self._adapter.decode_packet(signals)
        fields_ok = all(recovered.get(k) == v for k, v in packet.items())
        ok        = fields_ok
        detail    = "UDP packet fields: pass" if ok else f"FAIL: {recovered!r}"
        return ok, detail

    def test_file_chunk(self) -> Tuple[bool, str]:
        """Test: file chunk with sequence number and hash."""
        payload   = bytes((i * 31 + 7) % 256 for i in range(512))
        file_hash = hashlib.sha256(payload).hexdigest()
        signals   = self._adapter.encode_file_chunk(0, 1, payload, file_hash)
        recovered = self._adapter.decode_file_chunk(signals)
        ok        = (recovered["seq"] == 0 and
                     recovered["total"] == 1 and
                     recovered["data"] == payload and
                     recovered["file_hash"] == file_hash)
        detail    = "file chunk seq/data/hash: pass" if ok else f"FAIL: {recovered}"
        return ok, detail

    def test_framer(self) -> Tuple[bool, str]:
        """Test: CRC32 framing detects corruption."""
        data     = b"Ghost in the Machine Labs"
        framed   = self._framer.frame(data)
        unframed = self._framer.unframe(framed)
        ok_rt    = unframed == data
        # Corrupt one byte and verify CRC catches it
        corrupted = bytearray(framed)
        corrupted[6] ^= 0xFF
        crc_caught = False
        try:
            self._framer.unframe(bytes(corrupted))
        except ValueError:
            crc_caught = True
        ok     = ok_rt and crc_caught
        detail = "CRC framing + corruption detection: pass" if ok else "FAIL"
        return ok, detail

    def run_all(self) -> Tuple[int, int]:
        """Run all tests. Returns (passed, total)."""
        tests = [
            ("binary_256_values",  self.test_binary),
            ("binary_4096_blob",   self.test_binary_blob),
            ("json_roundtrip",     self.test_json),
            ("network_packet",     self.test_packet),
            ("file_chunk",         self.test_file_chunk),
            ("crc_framing",        self.test_framer),
        ]
        passed = 0
        for name, fn in tests:
            try:
                ok, detail = fn()
                if ok:
                    passed += 1
                    print(f"  ✓ {name}: {detail}")
                else:
                    print(f"  ✗ {name}: {detail}")
            except Exception as e:
                print(f"  ✗ {name}: exception: {e}")
        return passed, len(tests)

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "TransportTester", "test_count": 6, "initialized": self._initialized}


# ── Core processing stub (RM extends) ─────────────────────────────────────────
def _psi_core_process(self, data):
    if isinstance(data, bytes):
        return self.decode(self.encode(data))
    return data
PsiCodec._core_process = _psi_core_process


def main():
    """Demonstrate PsiDataTransport."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("PsiDataTransport — RM auto-synthesized software")
    log.info("Ghost in the Machine Labs")
    log.info("")
    log.info("Running transport verification suite...")
    log.info("")
    tester = TransportTester()
    passed, total = tester.run_all()
    log.info("")
    log.info(f"Results: {passed}/{total} tests passed")
    return 0 if passed == total else 1


def run_tests() -> Tuple[int, int]:
    """Run all tests. Returns (passed, total)."""
    tester = TransportTester()
    return tester.run_all()


if __name__ == "__main__":
    main()
