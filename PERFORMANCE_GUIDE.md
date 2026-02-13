
---

## Multi-Slot Scaling Note

The Harmonic Stack supports 16 model slots for specialized personas (Executive, Analyst, Coder, etc.). However, throughput is **not additive** across slots due to:

- Shared harmonic field coordination
- Router core overhead per dispatch
- Field decay calculations per operation
- Council governance validation

Multi-slot performance characteristics will be documented when the full orchestration layer is optimized.
