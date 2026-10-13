# Learn ModelOps Watch

This is an AI-assisted portfolio project created on 14 October 2026. Before describing it as your own engineering expertise in an interview, run it, read every function and complete the exercise below. Use “portfolio prototype” rather than “production experience.”

## Three learning sessions

1. **Run and inspect (45 minutes):** create the environment, run tests and the evaluation, compare the saved report, try one successful and one invalid API request.
2. **Trace the design (60 minutes):** start from the endpoint, follow validation into the core, draw the data flow and explain where parameters are fitted and where evaluation data enters.
3. **Change and defend (90 minutes):** implement the exercise, add a regression test, rerun evaluation and write down one result that worsened.

## Design questions

- How does training-only scaling prevent leakage?
- Why export numeric weights instead of pickle?
- How do feature drift, label drift and concept drift differ?
- What would make model rollback safe?

## Practical exercise

Run five split seeds and report variation. Compare a dummy classifier and add calibration plots. Add a simulated label shift and show why feature-only PSI cannot detect every performance failure.

## A truthful interview introduction

“This is a recent AI-assisted portfolio prototype. It demonstrates stratified split → train-only scaler → logistic regression → json artifact → validated api → distribution review. I can show the implementation and evaluation, and I distinguish its small demo results from operational performance.”

## Readiness check

Explain the baseline, identify two failure cases, change one behavior without assistance, write a test that fails before your fix, and explain why the reported numbers are limited. Record your learning and changes in a new commit; never backdate the project.
