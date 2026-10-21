"""Continual topic model vs sequential online LDA on a 20 Newsgroups stream.
Baseline: scikit-learn online LDA, partial_fit on each time slice in order.
Improved: same LDA and same data/slices, but after each partial_fit the topic-word matrix is
blended with the matrix from before that slice (a continual global prior). The blend weight
is chosen by cross-validation on the training split only (inner held-out training docs).
No extra LDA fits or extra data are used by the improved model."""
import json, os, socket, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import psi
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
SEEDS = [42, 43, 44]          # LDA is randomized: each model runs with all three seeds
N_TOPICS, N_SLICES, TOP_N = 20, 5, 10
MIX = 0.5                     # default prior weight (overridden by CV selection in main)
MIX_GRID = [0.0, 0.25, 0.5, 0.75]  # 0.0 == no prior, i.e. the baseline behaviour
EPS = 1e-12
URL = "https://ndownloader.figshare.com/files/5975967"
socket.setdefaulttimeout(60)  # 60 s timeout for the sklearn download (urlretrieve)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    # Retry with exponential backoff on network / HTTP errors (429, 5xx raise in urllib).
    for attempt in range(5):
        try:
            return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                      shuffle=True, random_state=SEED)
        except Exception as e:
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("could not download 20 Newsgroups")


def set_components(lda, comp):
    # Set topic-word matrix and the matching cached expectation used by sklearn's E-step.
    lda.components_ = comp
    lda.exp_dirichlet_component_ = np.exp(psi(comp) - psi(comp.sum(1, keepdims=True)))


def run(method, seed, Xtr, slices, mix=MIX):
    """Sequentially fit one LDA over the time slices; return model and top-word snapshots per slice.
    For 'improved', after each slice the new topic-word matrix is blended with the pre-slice matrix:
    comp = mix * previous + (1 - mix) * updated. No additional fits are performed."""
    lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online", random_state=seed)
    snaps = []
    for idx in slices:
        before = None if not hasattr(lda, "components_") else lda.components_.copy()
        lda.partial_fit(Xtr[idx])
        if method == "improved" and before is not None:
            set_components(lda, mix * before + (1 - mix) * lda.components_)
        snaps.append(np.argsort(-lda.components_, axis=1)[:, :TOP_N])
    return lda, snaps


def npmi_coherence(top, Xb):
    """Mean NPMI over word pairs of each topic's top-10 words, using binary held-out docs."""
    n = Xb.shape[0]
    df = np.asarray(Xb.mean(axis=0)).ravel()
    iu = np.triu_indices(TOP_N, 1)
    scores = []
    for words in top:
        B = Xb[:, words].toarray()
        P = (B.T @ B) / n                                   # pairwise document co-occurrence prob.
        pi = df[words]
        pmi = np.log((P + EPS) / (np.outer(pi, pi) + EPS))
        scores.append((pmi / -np.log(P + EPS))[iu].mean())
    return float(np.mean(scores))


def stability(snaps):
    """Mean Jaccard overlap of top-10 words for the same topic index between consecutive slices."""
    vals = [len(set(a) & set(b)) / len(set(a) | set(b))
            for s1, s2 in zip(snaps[:-1], snaps[1:]) for a, b in zip(s1, s2)]
    return float(np.mean(vals))


def tune_mix(Xtr, ytr, grid):
    """Select the prior weight on an inner stratified split of the TRAINING data only.
    The test set is never touched here."""
    fit_idx, val_idx = train_test_split(np.arange(Xtr.shape[0]), test_size=0.2,
                                        stratify=ytr, random_state=SEED)
    rng = np.random.default_rng(SEED)
    slices = [fit_idx[s] for s in np.array_split(rng.permutation(len(fit_idx)), N_SLICES)]
    Xval = (Xtr[val_idx] > 0).astype(np.float64).tocsr()
    scores = {}
    for mix in grid:
        lda, snaps = run("improved", SEED, Xtr, slices, mix)
        scores[mix] = npmi_coherence(snaps[-1], Xval)
        log(f"tune mix={mix}: validation NPMI={scores[mix]:.4f}")
    best = max(grid, key=lambda m: scores[m])   # first maximum on ties
    log(f"selected mix={best} (validation NPMI={scores[best]:.4f})")
    return best, scores[best]


def main():
    data = load_data()
    X_text, y = np.array(data.data, dtype=object), np.asarray(data.target)
    # Single stratified 80/20 split with fixed seed (test part is only used for final evaluation).
    tr_txt, te_txt, ytr, yte = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=SEED)
    log(f"rows={len(y)} train={len(ytr)} test={len(yte)}")

    vec = CountVectorizer(stop_words="english", max_features=2000, min_df=5)
    Xtr = vec.fit_transform(tr_txt)                          # vectorizer fitted on train only
    Xte = vec.transform(te_txt)
    Xte_bin = (Xte > 0).astype(np.float64).tocsr()           # binary docs for NPMI
    # No timestamps: random time slices of the training part, same for both models.
    rng = np.random.default_rng(SEED)
    slices = np.array_split(rng.permutation(Xtr.shape[0]), N_SLICES)
    log(f"vocab={Xtr.shape[1]} slices={[len(s) for s in slices]}")

    best_mix, best_val = tune_mix(Xtr, ytr, MIX_GRID)

    res = {m: {"coh": [], "stab": [], "f1": [], "acc": []} for m in ("baseline", "improved")}
    for seed in SEEDS:
        for m in res:
            lda, snaps = run(m, seed, Xtr, slices, best_mix)
            coh = npmi_coherence(snaps[-1], Xte_bin)
            stab = stability(snaps)
            # Secondary metric: logistic regression on document-topic proportions (train fit, test eval).
            clf = LogisticRegression(max_iter=2000, random_state=seed).fit(lda.transform(Xtr), ytr)
            pred = clf.predict(lda.transform(Xte))
            f1, acc = f1_score(yte, pred, average="macro"), accuracy_score(yte, pred)
            for k, v in zip(("coh", "stab", "f1", "acc"), (coh, stab, f1, acc)):
                res[m][k].append(float(v))
            log(f"seed {seed} {m}: NPMI={coh:.4f} stability={stab:.4f} macroF1={f1:.4f}")

    agg = {m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for m, d in res.items()}
    b, i = agg["baseline"], agg["improved"]
    metrics = {
        "baseline_score": b["coh"][0], "baseline_score_std": b["coh"][1],
        "improved_score": i["coh"][0], "improved_score_std": i["coh"][1],
        "improvement": i["coh"][0] - b["coh"][0],   # higher is better
        "baseline_stability": b["stab"][0], "improved_stability": i["stab"][0],
        "baseline_f1": b["f1"][0], "improved_f1": i["f1"][0],
        "baseline_accuracy": b["acc"][0], "improved_accuracy": i["acc"][0],
        "improved_mix_selected_cv": float(best_mix),
        "tuning_validation_npmi": float(best_val),
        "n_seeds": len(SEEDS),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        log("non-finite metrics; cannot produce meaningful results")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline (online LDA)", "improved (prior-blended)"],
            [b["coh"][0], i["coh"][0]], yerr=[b["coh"][1], i["coh"][1]], capsize=5,
            color=["grey", "tab:blue"])
    plt.ylabel("NPMI coherence (test, higher better)")
    plt.tight_layout()
    plt.savefig("figures/coherence_bar.png", dpi=120)
    plt.close()

    out = {
        "question": "Can a continual topic model that carries a prior forward across time slices beat a sequential online LDA baseline on topic coherence and topic stability over a text stream?",
        "method": "Stratified 80/20 split of 20 Newsgroups; training part cut into 5 random time slices; "
                  "sequential online LDA (20 topics) vs the same LDA where, after each slice, the topic-word "
                  "matrix is blended with the pre-slice matrix (weight chosen by cross-validation on an inner "
                  "split of the training data over {0, 0.25, 0.5, 0.75}); evaluated by NPMI on held-out test docs, "
                  "stability and LR macro-F1; 3 seeds.",
        "dataset": {
            "name": "20 Newsgroups (sklearn fetch_20newsgroups, subset=all, headers/footers/quotes removed)",
            "source_url": URL, "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "NPMI topic coherence of top-10 words on held-out test documents",
        "higher_is_better": True,
        "baseline": "Standard sequential online LDA (sklearn LatentDirichletAllocation, partial_fit per slice, 20 topics); stand-in, not the paper's neural CoNTM.",
        "improved": "Same online LDA and same slices, but after each partial_fit the topic-word matrix is blended with the pre-slice matrix as a continual prior; blend weight selected by CV on training data only. No extra fits or data.",
        "metrics": metrics,
        "notes": "Topic stability compares index-aligned topics across consecutive slices. The improved model performs "
                 "no extra LDA fits; its only additional cost is the blend step and a small inner CV over 4 blend weights "
                 "on training data (seed 42 only), which is run before the test evaluation. The baseline (mix=0 equivalent) "
                 "is part of the grid, so the selected setting is never worse than the baseline on the validation split. "
                 "The 20 Newsgroups data has no timestamps, so random slices simulate a stream. Results are from a small "
                 "seed count and a single data split.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"baseline NPMI={b['coh'][0]:.4f}  improved NPMI={i['coh'][0]:.4f}  wrote results.json")


if __name__ == "__main__":
    main()
