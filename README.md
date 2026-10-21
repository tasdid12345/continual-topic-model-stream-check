# Continual topic model vs sequential online LDA (20 Newsgroups stream)

## Question
Can a continual topic model that carries a prior forward across time slices beat a sequential online LDA baseline on topic coherence (NPMI of top-10 words on held-out test documents) and topic stability?

## Dataset
20 Newsgroups via scikit-learn (`subset="all"`, headers/footers/quotes removed). Stratified 80/20 train/test split with seed 42. The training part is cut into 5 random time slices (no timestamps exist in this data), identical for both models.

## Models
- **Baseline:** scikit-learn online LDA (20 topics), `partial_fit` on each slice in order.
- **Improved:** the same LDA; after each slice the topic-word matrix is blended with the matrix from before that slice (continual prior). The blend weight is chosen by cross-validation on an inner split of the training data over {0, 0.25, 0.5, 0.75}. The test set is not used for selection. The improved model performs no extra LDA fits.

## Evaluation
NPMI coherence of top-10 words on held-out test documents (primary, higher is better), topic stability across slices, and macro-F1 of a logistic regression on document-topic proportions. Results are averaged over seeds 42, 43, 44.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/coherence_bar.png`.

## Caveats
This is a stand-in for the paper's neural CoNTM, not a reproduction of it. The results come from one data split and three seeds.

## Reference

This project is motivated by the following paper:

> CK James, W Mustafa, M Monteiro (2026). *Continual neural topic model*. Proceedings of the. https://aclanthology.org/2026.eacl-long.312/

Paper author: Prof. Sophie Fellenz, RPTU Kaiserslautern-Landau (fellenz@cs.uni-kl.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
