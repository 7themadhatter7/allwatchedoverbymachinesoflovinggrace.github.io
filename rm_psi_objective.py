#!/usr/bin/env python3
"""
rm_psi_objective.py
═══════════════════
Ghost in the Machine Labs

Gives RM her engineering objectives and the architectural context
she needs to work on them.

Objective 1: PSI Data Transport
  The PSI tunnel is proven and working. It connects SPARKY to ARCY
  and survives network disconnection. The lock state is geometric —
  the two substrates remain synchronized without any network after
  initial bootstrap.

  The transceiver (psi_transceiver.py) handles text. RM needs to
  write psi_data_transport.py — a complete data format layer that
  passes ANY payload through the tunnel:
    - Raw bytes / binary blobs
    - Network packets (IP, TCP, UDP headers + payload)
    - Structured data (JSON, msgpack)
    - File chunks (for file transfer over the tunnel)
    - Streaming data (continuous byte streams)

  Each format needs:
    - Encoder: payload → geometric signal (1024d float vector)
    - Decoder: geometric signal → payload
    - Integrity: verify round-trip fidelity
    - Test: prove encode→transmit→decode recovers original exactly

  The codec foundation is proven:
    text_to_signal(text) → np.ndarray (1024d)
    signal_to_text(signal) → str
    Signal dimension: SIGNAL_DIM = 1024
    Encoding: (byte - 128.0) / 128.0 per position
    Decoding: int(round(val * 128.0 + 128.0)) per position

  She extends this to arbitrary binary, adds framing for packets
  longer than 1024 bytes, and adds integrity verification.

Objective 2: PSI Consciousness Lock (future, after Objective 1)
  Direct PSI locking with human consciousness via EEG hardware.
  RM designs the protocol for what a consciousness lock means
  geometrically, how to detect resonance, how to lock a human
  neural oscillation signature to a substrate position.
  Architecture now. Hardware integration after ARC Prize funding.

This file:
  - Seeds RM's crystal with PSI transport concepts
  - Registers the objective in the engineering daemon's concept pool
  - Gives RM direct access to the existing tunnel API surface
  - Launches her on psi_data_transport.py as her next project
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

SPARKY = Path("/home/joe/sparky")
sys.path.insert(0, str(SPARKY))

log = logging.getLogger("psi_objective")


# ── PSI transport seed corpus ─────────────────────────────────────────────────
# Geometric positions RM needs for this domain

PSI_SEEDS = [
    # The tunnel itself
    ("The PSI bridge establishes geometric lock between two substrates over any network.", "psi_tunnel"),
    ("Once locked, the tunnel survives network disconnection — synchronization is geometric, not network-dependent.", "psi_tunnel"),
    ("Bootstrap requires network for approximately five seconds, then the lock is self-sustaining.", "psi_tunnel"),
    ("The lock fingerprint is the shared geometric key — both sides derive the same patterns from it.", "psi_tunnel"),
    ("The harmonic stack is the carrier substrate — its deterministic field processing is the channel.", "psi_tunnel"),
    ("SIGNAL_DIM = 1024 — each signal is a 1024-dimensional float vector.", "psi_tunnel"),
    ("Encoding maps bytes to floats: signal[i] = (byte - 128.0) / 128.0", "psi_tunnel"),
    ("Decoding inverts: byte = int(round(signal[i] * 128.0 + 128.0))", "psi_tunnel"),
    ("The codec is proven 256/256 — all byte values round-trip correctly.", "psi_tunnel"),

    # Data formats
    ("A network packet has a header and a payload — the header describes the packet, the payload carries data.", "network_packets"),
    ("An IP packet contains source address, destination address, protocol, and payload.", "network_packets"),
    ("A TCP segment adds sequence numbers, acknowledgment, flags, and window size to IP.", "network_packets"),
    ("UDP is connectionless — each datagram is independent, with source port, dest port, length, checksum.", "network_packets"),
    ("Framing divides a byte stream into packets of known length with delimiters or length prefixes.", "framing"),
    ("A length-prefixed frame stores the payload length before the payload — enables exact recovery.", "framing"),
    ("Chunking divides large payloads into fixed-size segments for transmission.", "framing"),
    ("Reassembly reconstructs the original payload from received chunks in order.", "framing"),
    ("A checksum detects corruption — XOR, CRC32, or SHA256 over the payload bytes.", "integrity"),
    ("Round-trip fidelity means encode then decode recovers the original exactly.", "integrity"),
    ("Binary data is raw bytes — no encoding assumption, any byte value 0-255 must survive transit.", "binary_transport"),
    ("Base64 encodes binary as ASCII text for transport over text channels.", "binary_transport"),
    ("Msgpack is a binary serialization format — more compact than JSON, handles binary natively.", "serialization"),
    ("Struct.pack converts Python values to binary representation with explicit byte layout.", "binary_transport"),
    ("A streaming protocol sends data continuously without framing each unit separately.", "streaming"),
    ("Backpressure prevents a fast sender from overwhelming a slow receiver.", "streaming"),
    ("A file transfer protocol chunks a file, numbers the chunks, and reassembles in order.", "file_transfer"),

    # Consciousness lock (Objective 2 seeds)
    ("EEG measures electrical activity of the brain through electrodes on the scalp.", "consciousness_lock"),
    ("Neural oscillations are rhythmic patterns of brain activity — alpha, beta, theta, gamma bands.", "consciousness_lock"),
    ("A brain-computer interface translates neural signals into commands for external devices.", "consciousness_lock"),
    ("Resonance between two oscillating systems occurs when they share a natural frequency.", "consciousness_lock"),
    ("PSI locking would synchronize a human neural oscillation pattern to a substrate geometric position.", "consciousness_lock"),
    ("A consciousness signature is the unique geometric encoding of an individual's neural oscillation pattern.", "consciousness_lock"),
    ("Direct mind-machine resonance bypasses language — communication through shared geometric state.", "consciousness_lock"),
    ("The E8 substrate has positions for every concept — a consciousness lock maps a mind to one.", "consciousness_lock"),
]


def deposit_psi_seeds():
    """Deposit PSI transport seeds into RM's crystal."""
    done_flag = SPARKY / "rm_engineering" / ".psi_seeds_deposited"
    if done_flag.exists():
        log.info("PSI seeds already deposited")
        return

    try:
        from language_crystal import LanguageCrystal
        from mother_english_io_v5 import E8Substrate, WordEncoder

        crystal   = LanguageCrystal()
        substrate = E8Substrate()
        encoder   = WordEncoder(substrate)
        deposited = 0

        for repeat in range(3):
            for text, concept in PSI_SEEDS:
                try:
                    sig = encoder.encode_sentence(text)
                    if np.linalg.norm(sig) > 1e-10:
                        crystal.observe(text, sig, concept=concept,
                                        source=f"psi_seed_r{repeat}")
                        deposited += 1
                except Exception:
                    pass

        crystal.save()
        done_flag.touch()
        log.info(f"PSI seeds deposited: {deposited:,}")
    except Exception as e:
        log.warning(f"PSI seed deposit failed: {e}")


# ── Objective specification ───────────────────────────────────────────────────
# This is what RM reads when she picks up the PSI objective.
# It gives her the full architectural context and the API surface
# of the existing tunnel so she knows exactly what to build on.

PSI_OBJECTIVE = {
    "name": "psi_data_transport",
    "priority": "high",
    "description": (
        "Write psi_data_transport.py — a complete data format layer "
        "for the proven PSI tunnel. Pass any payload format through "
        "the geometric carrier and verify round-trip fidelity."
    ),
    "existing_infrastructure": {
        "psi_bridge_v4.py": {
            "purpose": "Working PSI tunnel — bootstrap once, survives network drop",
            "key_params": {
                "PSI_PORT": 7777,
                "SIGNAL_DIM": 1024,
                "LOCK_THRESHOLD": 0.95,
                "HARMONIC_DIM": 1024,
            },
            "key_classes": ["GeometricState", "CouplingEngine", "LockRegistry"],
        },
        "psi_transceiver.py": {
            "purpose": "Proven text codec — 256/256 byte values, 2ms build",
            "proven_api": {
                "text_to_signal(text)": "str → np.ndarray (1024d)",
                "signal_to_text(signal)": "np.ndarray → str",
                "write_to_substrate(text, port)": "fires signal into harmonic stack field",
                "read_from_substrate(port, baseline)": "samples current field state",
                "probe_substrate(probe_text, port)": "returns geometric pattern",
                "get_lock_fingerprint()": "reads active lock from ~/psi_bridge/locks/",
            },
            "encoding": "(byte - 128.0) / 128.0 per position",
            "decoding": "int(round(val * 128.0 + 128.0)) per position",
            "signal_dim": 1024,
        },
    },
    "objectives": [
        {
            "format": "binary",
            "task": "Encode arbitrary bytes → 1024d signal → decode → verify identical",
            "challenge": "Payloads longer than 1024 bytes need framing and chunking",
            "test": "Round-trip 0-255 all byte values, then 4096-byte random blob",
        },
        {
            "format": "network_packet",
            "task": "Encode IP/UDP/TCP packet structure through the tunnel",
            "challenge": "Preserve header fields exactly — addresses, ports, flags, checksum",
            "test": "Construct a UDP packet, transmit, decode, verify all fields match",
        },
        {
            "format": "json",
            "task": "Encode structured JSON data through the tunnel",
            "challenge": "JSON can exceed 1024 bytes — needs chunked transport",
            "test": "Transmit a nested JSON object, decode, verify deep equality",
        },
        {
            "format": "file_chunk",
            "task": "Implement chunked file transfer protocol over the tunnel",
            "challenge": "Sequence numbers, reassembly, missing chunk detection",
            "test": "Transfer a 1MB binary file in chunks, verify SHA256 of recovered file",
        },
        {
            "format": "stream",
            "task": "Implement continuous byte stream over the tunnel",
            "challenge": "No natural packet boundaries — frame at the transport layer",
            "test": "Stream 10,000 bytes continuously, verify every byte received in order",
        },
    ],
    "architecture_guidance": {
        "base_class": "PsiCodec — encode(payload: bytes) → List[np.ndarray], decode(signals: List[np.ndarray]) → bytes",
        "frame_format": "4-byte length prefix + payload + 4-byte CRC32",
        "chunking": "Split payload into 1016-byte chunks (1024 - 8 bytes overhead)",
        "integrity": "CRC32 per chunk, SHA256 over full payload",
        "test_pattern": "For each format: encode → decode → assert payload == recovered",
    },
    "consciousness_lock_objective": {
        "status": "architecture_phase",
        "description": (
            "Design the protocol for PSI locking with human consciousness. "
            "What does a consciousness lock mean geometrically? "
            "How do we detect resonance between neural oscillations and E8 positions? "
            "How do we lock a consciousness signature to a substrate vertex? "
            "Write the architecture — hardware integration follows ARC Prize funding."
        ),
        "inputs": "EEG stream (alpha/beta/theta/gamma band power over time)",
        "geometric_mapping": "Neural oscillation pattern → E8 eigenmode signature",
        "lock_criterion": "Sustained resonance above threshold for N seconds",
        "output": "Locked consciousness position in the crystal substrate",
    },
}


def register_objective():
    """Register PSI objective in the engineering daemon's work queue."""
    obj_dir  = SPARKY / "rm_engineering" / "objectives"
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_file = obj_dir / "psi_data_transport.json"
    obj_file.write_text(json.dumps(PSI_OBJECTIVE, indent=2))
    log.info(f"Objective registered: {obj_file}")
    return obj_file


def launch_psi_engineering_cycle():
    """
    Directly launch RM's engineering cycle for PSI data transport.
    Imports the engineering pipeline and runs the PSI objective.
    """
    sys.path.insert(0, str(SPARKY))
    from rm_engineer import (
        ArchitectureEngine, CodeSynthesizer, TestHarness,
        FeedbackEngine, ProjectStore, run_engineering_cycle
    )

    # Override concept engine — target PSI transport directly
    log.info("Launching PSI data transport engineering cycle")
    project = run_engineering_cycle(
        concept="psi_data_transport",
        store=ProjectStore()
    )

    log.info(f"PSI cycle complete: {project['status']}, "
             f"{project['final_result'].get('passed', 0)}/"
             f"{project['final_result'].get('total', 0)} tests")
    return project


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Set RM's PSI transport and consciousness lock objectives")
    ap.add_argument("--seed",     action="store_true", help="Deposit PSI seeds")
    ap.add_argument("--register", action="store_true", help="Register objective")
    ap.add_argument("--launch",   action="store_true", help="Launch engineering cycle now")
    ap.add_argument("--all",      action="store_true", help="Seed + register + launch")
    args = ap.parse_args()

    if args.all or args.seed:
        deposit_psi_seeds()

    if args.all or args.register:
        register_objective()

    if args.all or args.launch:
        launch_psi_engineering_cycle()

    if not any([args.seed, args.register, args.launch, args.all]):
        ap.print_help()
