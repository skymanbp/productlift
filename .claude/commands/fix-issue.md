---
description: Work a failing exercise test to green, the disciplined way
argument-hint: "[test path, e.g. tests/test_metrics.py]"
---

An exercise test is failing and we want it passing. Target: $ARGUMENTS

Follow this loop — the same discipline as real ML work:

1. **Read the test first.** It's the spec. Run it, read the failing assertion,
   state in one sentence what's required.
2. **Read the module docstring.** Each exercise has LEARNING GOAL / KEY CONCEPT /
   INTERVIEW ANGLE. Understand the concept — the point is to learn it, not just
   green the test.
3. **Implement the minimal correct version.** No gold-plating.
4. **Run the single test** until green, then the whole file for regressions.
5. **Run `ruff check` and `mypy`** on the file you touched.
6. **Explain in 2–3 sentences** what you did, why, and — since this is interview
   prep — how you'd describe the choice out loud to an interviewer.

Never edit a test to make it pass unless it's demonstrably wrong (say so first).
