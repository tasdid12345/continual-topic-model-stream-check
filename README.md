# Continual Topic Model vs Sequential Online LDA on a Text Stream

**Motivation.** Topic models are often trained on document streams whose topics drift over time, yet standard online LDA treats each time slice mostly as a fresh update. The "Continual neural topic model" (2026, https://aclanthology.org/2026.eacl-long.312/) proposes carrying a global prior forward across time steps so that topics stay coherent and consistent as new data arrives. This experiment asks whether a simple version of that idea helps a classical model.

**Question.** Can a continual topic model that carries a prior forward across time slices beat a sequential online LDA baseline on topic coherence and topic stability over a text stream?

**Dataset.** 20 Newsgroups, loaded with scikit-learn's `fetch_20newsgroups` (headers, footers and quotes removed). This is a public corpus and is not the paper's own dataset. It is split 80/20 into train and test with a fixed seed. The training part is cut into 5 random time slices, because the corpus has no timestamps.

**Baseline vs improved.**
- *Baseline:* standard sequential online LDA (scikit-learn `LatentDirichletAllocation`, `partial_fit` on each slice, 20 topics). It stands in for the paper's neural model and is not the paper's method.
- *Improved:* the same online LDA. Before each new slice, the previous topic-word matrix is mixed 50/50 with the current slice's topic-word estimate, acting as a continual global prior.

Both models share the same vectorizer, slices, seeds and evaluation code. The main metric is NPMI coherence on held-out test documents. Secondary metrics are topic stability and the macro-F1 of a logistic regression on document-topic proportions.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** written to `results.json` (and `figures/coherence_bar.png`) when the script finishes.

## Reference

This project is motivated by the following paper:

> CK James, W Mustafa, M Monteiro (2026). *Continual neural topic model*. Proceedings of the. https://aclanthology.org/2026.eacl-long.312/

Paper author: Prof. Sophie Fellenz, RPTU Kaiserslautern-Landau (fellenz@cs.uni-kl.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
