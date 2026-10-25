"""Session 3 (Level 3): hidden tastes (matrix factorisation), cold start and feedback loops.

Run:  python start.py 3
"""
# %% [markdown]
# # Session 3: learn the hidden tastes, then break the recommender on purpose
# **A. Real-world problem.** Neighbour methods (Session 2) compare raw pairs of users. A
# stronger idea: assume every user and every item can be described by a few hidden numbers
# (taste factors), and learn them so that "user factors . item factors" reproduces the known
# ratings. Then we ask the awkward questions: how many factors is too many, what do we show a
# brand-new user, and what happens to diversity when the system leans on popularity?
# This is education only: real systems add much more and must respect people's privacy.
#
# **Plan.** (1) alternating least squares (ALS), (2) choose rank k and regularisation lambda
# on a validation set, (3) overfitting and data leakage, (4) cold start, (5) popularity
# bias and diversity.

# %%
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path(".")) / "out"


# %% [markdown]
# ## Step 1: a synthetic world
# Same recipe as Session 2 (3 hidden taste factors, item and user biases, whole-star ratings
# 1..5, popular items observed more often), but with 240 users and 100 items. The first
# `n_cold` users are kept aside as *new users* for the cold-start tests.

# %%
def make_world(n_users=240, n_items=100, factors=3, density=0.15, seed=0):
    """Return (R, M): the full 1..5 matrix and the mask of observed cells."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, factors))
    V = rng.normal(size=(n_items, factors))
    user_bias = rng.normal(0, 0.3, n_users)
    item_bias = rng.normal(0, 0.5, n_items)
    raw = 3.2 + user_bias[:, None] + item_bias[None, :] + 0.8 * U @ V.T + rng.normal(0, 0.3, (n_users, n_items))
    R = np.clip(np.rint(raw), 1, 5)
    w = np.exp(1.2 * item_bias)
    p = np.clip(density * n_items * w / w.sum(), 0, 0.9)
    M = rng.random((n_users, n_items)) < p[None, :]
    return R, M


def three_way_split(M, val=0.15, test=0.15, seed=1):
    """Split observed cells into train / validation / test masks."""
    u = np.random.default_rng(seed).random(M.shape)
    te = M & (u < test)
    va = M & (u >= test) & (u < test + val)
    tr = M & (u >= test + val)
    return tr, va, te


def build_world(n_cold=40):
    """The world used in this session: full matrix, train/validation/test masks, and the ids of the
    first `n_cold` users, who are hidden from training and play the role of brand-new users."""
    R, M = make_world()
    cold = np.arange(n_cold)
    M = M.copy()
    M[cold] = False
    train, val, test = three_way_split(M)
    return R, train, val, test, cold


# %% [markdown]
# ## Step 2: alternating least squares
# Model: `rating(u, i) ~ mu + p_u . q_i` where `mu` is the average training rating and
# `p_u`, `q_i` are vectors of k numbers. We minimise
#
#     sum over known cells (r - mu - p_u . q_i)^2  +  lambda * (|p_u|^2 + |q_i|^2)
#
# If the item vectors are frozen, the best p_u is a small ridge regression with a closed
# form: `p_u = (Q^T Q + lambda I)^-1 Q^T r`, using only the items u rated. Then we freeze the
# users and solve for every q_i the same way. Repeat. Each half step cannot raise the
# objective, so it only goes down.

# %%
def _ridge_rows(X, mask, F, lam):
    """Solve each row's ridge regression against the factor matrix F (only observed cells)."""
    k = F.shape[1]
    out = np.zeros((X.shape[0], k))
    for r in range(X.shape[0]):
        idx = mask[r]
        if idx.any():
            Fi = F[idx]
            out[r] = np.linalg.solve(Fi.T @ Fi + lam * np.eye(k), Fi.T @ X[r, idx])
    return out


def rmse(pred, R, mask):
    return float(np.sqrt(np.mean((pred[mask] - R[mask]) ** 2)))


def predict_from(mu, P, Q):
    return np.clip(mu + P @ Q.T, 1, 5)


def als(R, train, k=3, lam=5.0, iters=15, seed=0, watch=None):
    """Fit by ALS. `watch` is an optional mask; its RMSE is recorded after each iteration.
    Returns (mu, P, Q, train_curve, watch_curve)."""
    rng = np.random.default_rng(seed)
    mu = float(R[train].mean())
    X = (R - mu) * train
    P = rng.normal(0, 0.1, (R.shape[0], k))
    Q = rng.normal(0, 0.1, (R.shape[1], k))
    tr_curve, wa_curve = [], []
    for _ in range(iters):
        P = _ridge_rows(X, train, Q, lam)
        Q = _ridge_rows(X.T, train.T, P, lam)
        pred = predict_from(mu, P, Q)
        tr_curve.append(rmse(pred, R, train))
        if watch is not None:
            wa_curve.append(rmse(pred, R, watch))
    return mu, P, Q, tr_curve, wa_curve


def objective(R, train, mu, P, Q, lam):
    """The regularised squared error that ALS minimises (before clipping to 1..5)."""
    err = (R - mu - P @ Q.T)[train]
    return float((err ** 2).sum() + lam * ((P ** 2).sum() + (Q ** 2).sum()))


# %% [markdown]
# ## Step 3: choose k and lambda on validation data, then report once on test data
# Training error always falls as k grows (more freedom). The **validation** error is what
# tells us when the model starts memorising noise instead of learning tastes.

# %%
def sweep(R, train, val, ks, lams, iters=15):
    """Validation RMSE for every (k, lambda); returns a len(ks) x len(lams) array and the train RMSE array."""
    V = np.zeros((len(ks), len(lams)))
    T = np.zeros_like(V)
    for a, k in enumerate(ks):
        for b, lam in enumerate(lams):
            _, _, _, tr, va = als(R, train, k, lam, iters, watch=val)
            V[a, b], T[a, b] = va[-1], tr[-1]
    return V, T


# %% [markdown]
# ## Step 4: cold start
# A new user has no ratings, so `p_u` has nothing to learn from. We fall back to
# *popularity*: recommend the items with the highest mean training rating, among items with
# at least 5 ratings. As soon as the user rates m items we can "fold in": solve the same ridge
# problem for p_u with the item vectors Q frozen.
# Score = **precision at 3**: the share of the 3 recommended unseen items that the user likes
# (true rating 4 or 5 in the full synthetic matrix). We use precision rather than "at least
# one hit" because with 3 slots and many liked items, "at least one hit" is almost always met.

# %%
def popularity_order(R, train, min_count=5):
    """Item indices from best to worst mean training rating (items with few ratings last)."""
    count = train.sum(axis=0)
    mean = np.where(count > 0, (R * train).sum(axis=0) / np.maximum(count, 1), 0.0)
    mean = np.where(count >= min_count, mean, -1.0 + 0.001 * count)
    return np.argsort(-mean)


def fold_in(ratings, items, mu, Q, lam):
    """Item-factor ridge fit for a new user who rated `items` with `ratings`."""
    k = Q.shape[1]
    if len(items) == 0:
        return np.zeros(k)
    Fi = Q[items]
    return np.linalg.solve(Fi.T @ Fi + lam * np.eye(k), Fi.T @ (ratings - mu))


def recommend_new_user(given_items, given_ratings, mu, Q, order, lam, n=3):
    """Top-n unseen items. With no ratings: popularity. Otherwise: folded-in prediction."""
    given = set(int(i) for i in given_items)
    if len(given) == 0:
        ranked = list(order)
    else:
        p = fold_in(np.asarray(given_ratings, float), np.asarray(given_items, int), mu, Q, lam)
        ranked = list(np.argsort(-(Q @ p)))
    return [int(i) for i in ranked if int(i) not in given][:n]


def cold_start_curve(R, train, users, mu, Q, lam, ms, n=3, liked=4, seed=0):
    """Precision at n for ALS fold-in and for popularity as the new users supply m ratings.
    Each new user's m given items are random; unseen items are scored against the full matrix."""
    rng = np.random.default_rng(seed)
    order = popularity_order(R, train)
    als_p, pop_p = [], []
    for m in ms:
        a = b = 0.0
        for u in users:
            items = rng.choice(R.shape[1], size=m, replace=False)
            rec_a = recommend_new_user(items, R[u, items], mu, Q, order, lam, n)
            rec_p = [int(i) for i in order if int(i) not in set(int(x) for x in items)][:n]
            a += float((R[u, rec_a] >= liked).mean())
            b += float((R[u, rec_p] >= liked).mean())
        als_p.append(a / len(users))
        pop_p.append(b / len(users))
    return als_p, pop_p


# %% [markdown]
# ## Step 5: popularity bias and shrinking diversity
# Diversity of a set of top-N lists: **coverage** = distinct items shown divided by catalogue
# size; **top-10 share** = fraction of all recommendation slots filled by the 10 most-rated
# items. We blend personal scores with popularity: `score = (1 - a) * z(prediction) +
# a * z(popularity)`, where z means "standardised" (subtract the mean, divide by the spread).
# a = 0 is pure personal prediction, a = 1 is pure popularity.

# %%
def top_n_lists(score, seen, n=5):
    s = np.where(seen, -np.inf, score)
    return np.argsort(-s, axis=1)[:, :n]


def coverage(lists, n_items):
    return len(np.unique(lists)) / n_items


def top_share(lists, count, top=10):
    popular = set(np.argsort(-count)[:top].tolist())
    return float(np.mean([int(i) in popular for i in lists.ravel()]))


def blended_scores(pred, count, a):
    zp = (pred - pred.mean(axis=1, keepdims=True)) / pred.std(axis=1, keepdims=True)
    zc = (count - count.mean()) / count.std()
    return (1 - a) * zp + a * zc[None, :]


def diversity_study(R, train, pred, users, alphas, n=5, liked=4):
    """For each blend a: (coverage, top-10 share, precision at n) over the given users."""
    count = train.sum(axis=0).astype(float)
    out = []
    for a in alphas:
        lists = top_n_lists(blended_scores(pred, count, a)[users], train[users], n)
        prec = float(np.mean([(R[u, lists[i]] >= liked).mean() for i, u in enumerate(users)]))
        out.append((coverage(lists, R.shape[1]), top_share(lists, count), prec))
    return out


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    R, train, val, test, cold = build_world()
    N_COLD = len(cold)
    print(f"{train.sum() + val.sum() + test.sum()} observed ratings in the world: train {train.sum()}, validation {val.sum()}, test {test.sum()}")

    mu, P, Q, trc, tec = als(R, train, k=3, lam=5.0, iters=15, watch=val)
    objs = [objective(R, train, *als(R, train, 3, 5.0, i)[:3], 5.0) for i in (1, 2, 3, 5, 8)]
    print("objective after 1, 2, 3, 5, 8 iterations:", [round(o) for o in objs])
    print(f"ALS k=3 lambda=5: train RMSE {trc[0]:.3f} -> {trc[-1]:.3f}, validation {tec[0]:.3f} -> {tec[-1]:.3f}")

    ks = [1, 2, 3, 4, 6, 8, 12, 20]
    lams = [0.1, 1.0, 5.0, 20.0]
    V, T = sweep(R, train, val, ks, lams)
    a, b = np.unravel_index(np.argmin(V), V.shape)
    print(f"best validation RMSE {V[a, b]:.3f} at k={ks[a]}, lambda={lams[b]}")
    for kk in (1, 3, 20):
        print(f"  k={kk:2d} lambda=0.1: train {T[ks.index(kk), 0]:.3f}  validation {V[ks.index(kk), 0]:.3f}")
    mu, P, Q, _, _ = als(R, train, ks[a], lams[b])
    print(f"test RMSE of the chosen model {rmse(predict_from(mu, P, Q), R, test):.3f}; item-mean baseline "
          f"{rmse(np.tile(np.where(train.sum(0) > 0, (R * train).sum(0) / np.maximum(train.sum(0), 1), mu), (R.shape[0], 1)), R, test):.3f}")

    # leakage: train on train+test, then 'evaluate' on test
    leak_train = train | test
    mu2, P2, Q2, _, _ = als(R, leak_train, 20, 0.1)
    print(f"LEAK: k=20 lambda=0.1 trained with test ratings -> 'test' RMSE {rmse(predict_from(mu2, P2, Q2), R, test):.3f}")
    mu3, P3, Q3, _, _ = als(R, train, 20, 0.1)
    print(f"honest: same model trained on train only    -> test RMSE {rmse(predict_from(mu3, P3, Q3), R, test):.3f}")

    # cold start
    mu, P, Q, _, _ = als(R, train, ks[a], lams[b])
    ms = [0, 1, 2, 3, 5, 10]
    ah, ph = cold_start_curve(R, train, cold, mu, Q, lams[b], ms)
    print("cold start precision@3 ALS fold-in:", [f"{m}:{h:.2f}" for m, h in zip(ms, ah)])
    print("cold start precision@3 popularity :", [f"{m}:{h:.2f}" for m, h in zip(ms, ph)])
    print(f"share of liked cells for new users (random pick would score): {(R[cold] >= 4).mean():.2f}")

    users = np.arange(N_COLD, R.shape[0])
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    div = diversity_study(R, train, predict_from(mu, P, Q), users, alphas)
    for al, (cv, sh, pr) in zip(alphas, div):
        print(f"popularity weight a={al:.2f}: coverage {cv:.2f}  top-10 share {sh:.2f}  precision@5 {pr:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(range(1, 16), trc, "o-", label="train")
    axes[0].plot(range(1, 16), tec, "s-", label="validation")
    axes[0].set_xlabel("ALS iteration")
    axes[0].set_ylabel("RMSE (stars)")
    axes[0].set_title("ALS: error falls over iterations")
    axes[0].legend()
    for j, lam in enumerate(lams):
        axes[1].plot(ks, V[:, j], "o-", label=f"validation, lambda={lam}")
    axes[1].plot(ks, T[:, 0], "k--", label="train, lambda=0.1")
    axes[1].set_xlabel("rank k (hidden factors)")
    axes[1].set_ylabel("RMSE (stars)")
    axes[1].set_title("Validation error has a minimum")
    axes[1].legend(fontsize=7)
    axes[2].plot(ms, ah, "o-", label="ALS fold-in")
    axes[2].plot(ms, ph, "s-", label="popularity")
    axes[2].set_xlabel("ratings given by the new user")
    axes[2].set_ylabel("precision at 3")
    axes[2].set_title("Cold start")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "als_validation_coldstart.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.plot(alphas, [d[0] for d in div], "o-", label="coverage")
    ax2.plot(alphas, [d[1] for d in div], "s-", label="top-10 share")
    ax2.plot(alphas, [d[2] for d in div], "^-", label="precision at 5")
    ax2.set_xlabel("popularity weight a (0 = personal, 1 = popularity only)")
    ax2.set_ylabel("fraction (no unit)")
    ax2.set_title("More popularity, less diversity")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "popularity_vs_diversity.png", dpi=120)

# %% [markdown]
# ## Where this stops
# Real recommenders add implicit feedback (clicks), time, side information and fairness
# checks. The road from here to the Target animation is in `promo/roadmap.md`.
