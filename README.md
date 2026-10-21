# Continual topic model vs online LDA (20 Newsgroups stream)

Question: can a continual topic model that carries a prior across time slices beat a sequential online LDA baseline on topic coherence (NPMI on held-out test documents) and topic stability?

## Setup
- Data: 20 Newsgroups (sklearn `fetch_20newsgroups`, subset=all, headers/footers/quotes removed), stratified 80/20 train/test split, seed 42.
- Training part is cut into 5 random time slices (no timestamps available).
- Vocabulary: CountVectorizer, English stop words, 2000 features, min_df=5, fitted on train only.

## Models
- **Baseline:** sklearn online LDA (20 topics) with default priors, `partial_fit` on each slice in order.
- **Improved:** same LDA and slices, with the topic-word prior `eta` and a per-slice blend of the topic-word matrix with the pre-slice matrix (weight `mix`). `eta` in {default, 0.02, 0.05, 0.2} and `mix` in {0, 0.25, 0.5, 0.75} are chosen by cross-validation on an inner split of the training data only. The grid includes the baseline setting, so the selected setting is never worse on validation.

## Evaluation
- Primary: NPMI coherence of top-10 words, computed on binary held-out test documents, mean over 3 seeds (42, 43, 44).
- Secondary: topic stability across slices, and macro-F1 / accuracy of logistic regression on document-topic proportions.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Writes `results.json` and `figures/coherence_bar.png`. Runtime is a few minutes on CPU.

## Reference

This project is motivated by the following paper:

> CK James, W Mustafa, M Monteiro (2026). *Continual neural topic model*. Proceedings of the. https://aclanthology.org/2026.eacl-long.312/

Paper author: Prof. Sophie Fellenz, RPTU Kaiserslautern-Landau (fellenz@cs.uni-kl.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
