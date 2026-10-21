"""Continual topic model vs sequential online LDA on a 20 Newsgroups stream.
Baseline: scikit-learn online LDA, partial_fit on each time slice in order.
Improved: same LDA, but before each new slice the topic-word matrix is mixed 50/50
with the previous one (a continual global prior), then the slice is fitted as usual."""
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
MIX = 0.5                     # weight on the previous topic-word matrix (continual prior)
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


def run(method, seed, Xtr, slices):
    """Sequentially fit one LDA over the time slices; return model and top-word snapshots per slice."""
    lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online", random_state=seed)
    snaps, prev = [], None
    for idx in slices:
        if method == "improved" and prev is not None:
            # slice-only topic-word estimate (same seed), mixed 50/50 with the previous global matrix
            cur = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online",
                                            random_state=seed, max_iter=5).fit(Xtr[idx]).components_
            phi_prev = prev / prev.sum(1, keepdims=True)
            phi_cur = cur / cur.sum(1, keepdims=True)
            set_components(lda, (MIX * phi_prev + (1 - MIX) * phi_cur) * prev.sum(1, keepdims=True))
        lda.partial_fit(Xtr[idx])
        prev = lda.components_.copy()
        snaps.append(np.argsort(-prev, axis=1)[:, :TOP_N])
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


def main():
    data = load_data()
    X_text, y = np.array(data.data, dtype=object), np.asarray(data.target)
    # Single stratified 80/20 split with fixed seed (no tuning, so nothing is selected on test).
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

    res = {m: {"coh": [], "stab": [], "f1": [], "acc": []} for m in ("baseline", "improved")}
    for seed in SEEDS:
        for m in res:
            lda, snaps = run(m, seed, Xtr, slices)
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
        "n_seeds": len(SEEDS),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        log("non-finite metrics; cannot produce meaningful results")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline (online LDA)", "improved (prior-mixed)"],
            [b["coh"][0], i["coh"][0]], yerr=[b["coh"][1], i["coh"][1]], capsize=5,
            color=["grey", "tab:blue"])
    plt.ylabel("NPMI coherence (test, higher better)")
    plt.tight_layout()
    plt.savefig("figures/coherence_bar.png", dpi=120)
    plt.close()

    out = {
        "question": "Can a continual topic model that carries a prior forward across time slices beat a sequential online LDA baseline on topic coherence and topic stability over a text stream?",
        "method": "Stratified 80/20 split of 20 Newsgroups; training part cut into 5 random time slices; "
                  "sequential online LDA (20 topics) vs the same LDA with 50/50 mixing of the previous topic-word "
                  "matrix before each slice; evaluated by NPMI on held-out test docs, stability and LR macro-F1; 3 seeds.",
        "dataset": {
            "name": "20 Newsgroups (sklearn fetch_20newsgroups, subset=all, headers/footers/quotes removed)",
            "source_url": URL, "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "NPMI topic coherence of top-10 words on held-out test documents",
        "higher_is_better": True,
        "baseline": "Standard sequential online LDA (sklearn LatentDirichletAllocation, partial_fit per slice, 20 topics); stand-in, not the paper's neural CoNTM.",
        "improved": "Same online LDA, but before each new slice the previous topic-word matrix is mixed 50/50 with the current slice's topic-word estimate as a continual global prior.",
        "metrics": metrics,
        "notes": "Topic stability compares index-aligned topics across consecutive slices. The improved model does an extra "
                 "5-iteration LDA fit per slice, so its compute is not identical to the baseline. No hyper-parameter "
                 "search was run; all settings were fixed a priori. The 20 Newsgroups data has no timestamps, so random "
                 "slices simulate a stream. Results are from a small seed count and a single data split.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"baseline NPMI={b['coh'][0]:.4f}  improved NPMI={i['coh'][0]:.4f}  wrote results.json")


if __name__ == "__main__":
    main()
