"""Ingestion, translation, SFT-set generation, preference-pair generation,
train/val splitting (DESIGN.md §2). Filled in starting P1.

One genuine coupling: preference-pair generation samples from the SFT
adapter, so it calls src.serve's generation function rather than a
second inference path (DESIGN.md §2) -- a divergence would mean the
preference pairs came from a different model than the one evaluated.
"""
