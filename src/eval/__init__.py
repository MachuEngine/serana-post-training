"""Quality metrics + judges (DESIGN.md §2). Filled in starting P5.

Circularity guard (DESIGN.md §4.4): `preference_judge` (best/worst of N
candidate replies, trains DPO) and `judge_pcs` (a single absolute rating,
scores results in P5) are separate prompts/rubrics -- don't collapse them
into one function. Judge validation against the 50 human
labels (Spearman >= ~0.6) must pass before any judge-backed number is
trusted.
"""
