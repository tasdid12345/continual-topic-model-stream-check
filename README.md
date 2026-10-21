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

## Results (latest run)

On 20 Newsgroups held-out test documents, a standard sequential online LDA baseline reached NPMI 0.228 versus 0.229 for my CV-tuned prior-blend variant, a negligible gain within seed noise, while stability rose from 0.632 to 0.703 and LR macro-F1 dropped from 0.332 to 0.325.

| Metric | Value |
|---|---|
| baseline_score | 0.2281 |
| baseline_score_std | 0.01785 |
| improved_score | 0.2294 |
| improved_score_std | 0.01949 |
| improvement | 0.001293 |
| baseline_stability | 0.632 |
| improved_stability | 0.7032 |
| baseline_f1 | 0.3319 |
| improved_f1 | 0.3246 |
| baseline_accuracy | 0.3599 |
| improved_accuracy | 0.3527 |
| improved_mix_selected_cv | 0.25 |
| improved_topic_word_prior_selected_cv | 0.05 |
| tuning_validation_npmi | 0.2683 |
| n_seeds | 3 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Topic stability compares index-aligned topics across consecutive slices. The improved model's hyperparameters (topic_word_prior eta and blend weight mix) are selected by a small inner CV on training data only (seed 42), run before the test evaluation; the grid contains the baseline setting, so the selected setting is never worse than the baseline on the validation split. Test-set gains are not guaranteed and are reported as measured. The 20 Newsgroups data has no timestamps, so random slices simulate a stream. Results are from a small seed count and a single data split.

## Reference

This project is motivated by the following paper:

> CK James, W Mustafa, M Monteiro (2026). *Continual neural topic model*. Proceedings of the. https://aclanthology.org/2026.eacl-long.312/

Paper author: Prof. Sophie Fellenz, RPTU Kaiserslautern-Landau (fellenz@cs.uni-kl.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
