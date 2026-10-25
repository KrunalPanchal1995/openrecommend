"""Session 2 (Level 2): similarity between users, prediction, and honest evaluation.

Run:  python start.py 2
"""
# %% [markdown]
# # Session 2: measure similarity, predict, then test honestly
# **A. Real-world problem.** Session 1 compared 5 people by hand. A real service has
# thousands of users, each with a few dozen ratings. How do we score "similar taste" with a
# formula, turn it into a predicted rating, and know that the predictions are any good?
#
# **Plan.** (1) make a seeded synthetic ratings matrix from hidden taste factors, (2) cosine
# and Pearson similarity on co-rated items, (3) predict as a similarity-weighted average,
# user-based and item-based, (4) hide 20% of known ratings and score with RMSE and hit rate.
# All data is synthetic; the "tastes" are random numbers.

# %%
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path(".")) / "out"


# %% [markdown]
# ## Step 1: a synthetic world with hidden tastes
# Every user has 3 hidden taste numbers and every item has 3 hidden "flavour" numbers. The
# true liking is `3.2 + user bias + item bias + 0.8 * (taste . flavour)`, plus a little noise,
# rounded to whole stars 1..5. Only some cells are *observed*, and popular (high item bias)
# items are observed more often, as in real life.

# %%
def make_ratings(n_users=100, n_items=60, factors=3, density=0.35, seed=0):
    """Return (R, M): R is the full 1..5 matrix, M marks which cells were observed."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, factors))
    V = rng.normal(size=(n_items, factors))
    user_bias = rng.normal(0, 0.3, n_users)
    item_bias = rng.normal(0, 0.5, n_items)
    raw = 3.2 + user_bias[:, None] + item_bias[None, :] + 0.8 * U @ V.T + rng.normal(0, 0.3, (n_users, n_items))
    R = np.clip(np.rint(raw), 1, 5)
    w = np.exp(1.2 * item_bias)
    p = np.clip(density * n_items * w / w.sum(), 0, 0.9)      # exposure of each item
    M = rng.random((n_users, n_items)) < p[None, :]
    return R, M


def split_observed(M, test_fraction=0.2, seed=1):
    """Hide a random share of the observed cells. Returns (train_mask, test_mask)."""
    rng = np.random.default_rng(seed)
    test = M & (rng.random(M.shape) < test_fraction)
    return M & ~test, test


# %% [markdown]
# ## Step 2: similarity on co-rated items
# For users a and b take only the items both rated, giving rating lists x and y.
#
#     cosine  = (x . y) / (|x| |y|)
#     pearson = the same formula after subtracting each list's own mean
#
# Pearson removes "this user rates everything high". With plain 1..5 ratings, cosine is
# always positive and barely tells people apart. We need at least `min_common` shared items.

# %%
def similarity(R, M, method="pearson", min_common=3):
    """Square matrix of similarities between the rows of R (zero where too little is shared)."""
    n = R.shape[0]
    S = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            both = M[a] & M[b]
            if both.sum() < min_common:
                continue
            x, y = R[a, both], R[b, both]
            if method == "pearson":
                x, y = x - x.mean(), y - y.mean()
            den = np.linalg.norm(x) * np.linalg.norm(y)
            S[a, b] = S[b, a] = (x @ y) / den if den > 0 else 0.0
    return S


# %% [markdown]
# ## Step 3: predict = own average + similarity-weighted opinions of neighbours
#
#     pred(u, i) = mean_u + sum_v s(u,v) * (r(v,i) - mean_v) / sum_v |s(u,v)|
#
# The sums run over the `k` most similar users v (positive similarity) who rated item i.
# If nobody qualifies we return the user's own mean. For **item-based** prediction we run
# the very same code on the transposed table (items play the role of users).

# %%
def top_k_neighbours(S, k):
    """Keep only the k largest positive similarities in each row."""
    out = np.zeros_like(S)
    for a in range(S.shape[0]):
        row = S[a].copy()
        row[a] = 0.0
        idx = np.argsort(row)[::-1][:k]
        idx = idx[row[idx] > 0]
        out[a, idx] = row[idx]
    return out


def predict_all(R, M, S, k=20):
    """Predicted rating for every cell of R, from the observed cells in M."""
    counts = np.maximum(M.sum(axis=1), 1)
    means = (R * M).sum(axis=1) / counts
    centred = (R - means[:, None]) * M
    Sk = top_k_neighbours(S, k)
    num = Sk @ centred
    den = np.abs(Sk) @ M.astype(float)
    adjust = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.clip(means[:, None] + adjust, 1, 5)


def predict_user_based(R, M, k=20, method="pearson"):
    return predict_all(R, M, similarity(R, M, method), k)


def predict_item_based(R, M, k=20, method="pearson"):
    return predict_all(R.T, M.T, similarity(R.T, M.T, method), k).T


# %% [markdown]
# ## Step 4: evaluate by hiding known ratings
# * **RMSE** = sqrt(mean((prediction - truth)^2)) over the hidden cells, in stars. Smaller is better.
# * **Hit rate at N**: for each user who has at least one hidden item rated 4 or 5 ("liked"),
#   rank all items they have not rated in the training data; a hit means a liked hidden item
#   appears in the top N (we use N = 5).

# %%
def rmse(pred, R, mask):
    return float(np.sqrt(np.mean((pred[mask] - R[mask]) ** 2)))


def global_mean_prediction(R, train):
    return np.full(R.shape, R[train].mean())


def item_mean_prediction(R, train):
    counts = np.maximum(train.sum(axis=0), 1)
    means = np.where(train.sum(axis=0) > 0, (R * train).sum(axis=0) / counts, R[train].mean())
    return np.tile(means, (R.shape[0], 1))


def hit_rate(score, R, train, test, n=5, liked=4):
    """Share of eligible users with a liked hidden item in their top-n list."""
    hits, users = 0, 0
    for u in range(R.shape[0]):
        target = test[u] & (R[u] >= liked)
        if not target.any():
            continue
        users += 1
        s = np.where(train[u], -np.inf, score[u])
        top = np.argsort(-s)[:n]
        hits += bool(target[top].any())
    return hits / users


def random_hit_rate(R, train, test, n=5, liked=4, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([hit_rate(rng.random(R.shape), R, train, test, n, liked) for _ in range(trials)]))


def popularity_score(train):
    """How many training ratings each item has, repeated for every user."""
    return np.tile(train.sum(axis=0).astype(float), (train.shape[0], 1))


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    R, M = make_ratings()
    train, test = split_observed(M)
    print(f"{R.shape[0]} users x {R.shape[1]} items, {M.sum()} observed ratings, {test.sum()} hidden for testing")
    results = {}
    results["global mean"] = rmse(global_mean_prediction(R, train), R, test)
    results["item mean"] = rmse(item_mean_prediction(R, train), R, test)
    Pu = predict_user_based(R, train)
    Pi = predict_item_based(R, train)
    results["user-based (Pearson)"] = rmse(Pu, R, test)
    results["item-based (Pearson)"] = rmse(Pi, R, test)
    results["user-based (cosine)"] = rmse(predict_user_based(R, train, method="cosine"), R, test)
    for name, v in results.items():
        print(f"RMSE {name:22s} {v:.3f} stars")
    print(f"hit rate@5  random {random_hit_rate(R, train, test):.3f}  popularity {hit_rate(popularity_score(train), R, train, test):.3f}"
          f"  user-based {hit_rate(Pu, R, train, test):.3f}  item-based {hit_rate(Pi, R, train, test):.3f}")

    ks = [1, 2, 5, 10, 20, 40, 80]
    S = similarity(R, train)
    curve = [rmse(predict_all(R, train, S, k), R, test) for k in ks]
    print("RMSE versus k:", [f"{k}:{c:.3f}" for k, c in zip(ks, curve)])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(similarity(R, train, "cosine")[np.triu_indices(100, 1)], bins=30, alpha=0.6, label="cosine")
    axes[0].hist(S[np.triu_indices(100, 1)], bins=30, alpha=0.6, label="Pearson")
    axes[0].set_xlabel("similarity between two users (no unit)")
    axes[0].set_ylabel("number of user pairs")
    axes[0].set_title("Cosine is bunched near 1; Pearson spreads out")
    axes[0].legend()
    names = list(results)
    axes[1].barh(names, [results[n] for n in names], color="#2a9d8f")
    axes[1].set_xlabel("test RMSE (stars)")
    axes[1].set_title("Hidden-rating error by method")
    axes[2].plot(ks, curve, "o-")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("neighbours k")
    axes[2].set_ylabel("test RMSE (stars)")
    axes[2].set_title("User-based error versus k")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "similarity_and_error.png", dpi=120)

# %% [markdown]
# ## Where this stops
# Neighbour methods look at raw pairs of users. Session 3 learns a small set of hidden tastes
# directly (matrix factorisation) and also faces cold start and feedback loops.
