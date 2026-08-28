"""VRAM estimation, profiling helpers, nvidia-smi logging, MFU calculation
(DESIGN.md §2 and §7). Imported by src.finetune and src.serve; owns no
pipeline logic of its own. First real user: P2 (§7.2 knob ablation,
predicted-vs-measured peak VRAM). scripts/gpu_probe.py is the standalone
entry point for the P0 device sanity check, not part of this package.
"""
