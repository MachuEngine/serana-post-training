"""vLLM + FastAPI; owns the inference path (DESIGN.md §2).

`pipeline.run` is the single entry point all three configs (B/SFT/DPO)
go through -- see DESIGN.md §1 and the single-pipeline rule in CLAUDE.md.
"""
