"""Quality metrics + judges (DESIGN.md §2). Filled in starting P5.

Circularity guard (DESIGN.md §4.4): `preference_judge` (pairwise, P3) and
`judge_pcs` (absolute rating, P5) are separate prompts/rubrics -- don't
collapse them into one function. Judge validation against the 50 human
labels (Spearman >= ~0.6) must pass before any judge-backed number is
trusted.
"""
