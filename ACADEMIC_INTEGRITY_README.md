# Before you submit this: read this file

This repository was scaffolded with AI assistance. Your assignment's own
Academic Integrity Note says submissions with **highly similar
implementations** — identical feature selections, EDA visualizations, model
pipelines, Docker/Kubernetes configs, or reports — may be treated as copied
and referred for academic integrity review, independent of whether you
personally wrote every line.

If another student in your cohort asks an AI assistant for the same
assignment brief, there is a real chance the output looks like this one.
Treat everything below as a reference implementation to learn from and
rebuild in your own words/choices, not a file to zip and upload.

## Minimum personalization checklist

- [ ] **Re-derive, don't just read, the preprocessing decisions.** Why median
      imputation and not mean? Why one-hot and not ordinal for `slope`? Write
      your own justification — even if you land on the same answer, arrive
      at it yourself.
- [ ] **Change at least one modeling choice.** Different hyperparameter grid,
      a 4th model, a different train/test split ratio, different CV fold
      count — something that changes your numbers slightly from the table in
      README.md.
- [ ] **Write the EDA narrative yourself.** The figures in `reports/figures/`
      are real, but *why* each pattern matters clinically/statistically
      should be your own observation, not a caption I wrote.
- [ ] **Redesign the monitoring dashboard layout** if you want it to look
      like yours rather than a template — different panels, different
      thresholds, different alerting rules.
- [ ] **Do the deployment yourself, on your own machine/cloud account**, and
      take your own screenshots. I cannot do this part for you — Minikube,
      real cloud credentials, and screen recording all require your actual
      environment.
- [ ] **Record your own video walkthrough** narrating what YOU built and why,
      not reading a script.
- [ ] **Write the report's reflection/discussion sections yourself** —
      model-selection reasoning, what you'd do differently, what surprised
      you. The CV-vs-test-ROC-AUC discrepancy noted in the README is real and
      you're welcome to build on it, but write it in your own words.

## What you still have to do that nothing here can do for you

1. Push this to your own GitHub repository.
2. Actually deploy to Minikube/Docker Desktop/a real cloud account and
   capture real screenshots (`reports/screenshots/`).
3. Record the short video of the overall pipeline.
4. Fill in the report placeholders marked `[STUDENT: ...]` in
   `reports/Final_Report.docx`.
5. Re-read both the assignment PDF and FAQ PDF yourself once more before
   submitting — this scaffold is thorough but is not a substitute for you
   knowing what you handed in.
