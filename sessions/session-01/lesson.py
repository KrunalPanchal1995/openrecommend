"""Session 1 (Level 1): people who liked the same things like the same new things.

Run:  python start.py 1
"""
# %% [markdown]
# # Session 1: people who agree tend to keep agreeing
# **A. Real-world problem.** A video app has thousands of films and you have seen only a few.
# Which one should it show you next? It cannot read your mind, but it can look at *other
# people* who rated the same films you did and see what they liked next.
#
# **Plan.** (1) write a tiny 5 people x 6 films table, (2) measure how far apart two people's
# tastes are, (3) predict a missing rating from the closest person, (4) recommend one film.
# All data here is made up for teaching.

# %%
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path(".")) / "out"

PEOPLE = ["Ana", "Ben", "Chen", "Dia", "Eli"]
FILMS = ["Space Quest", "Robot Dawn", "Love Letters", "Summer Vows", "Dark Alley", "Night Chase"]
NAN = np.nan
# ratings from 1 (hated it) to 5 (loved it); NaN means "has not seen it"
RATINGS = np.array([
    [5, 4, 1, 2, 4, NAN],      # Ana  (has not seen Night Chase)
    [4, 5, 2, 1, 5, 5],        # Ben
    [1, 2, 5, 5, 2, 1],        # Chen
    [2, 1, 4, 5, 1, 2],        # Dia
    [5, 5, 2, 1, 4, 4],        # Eli
], dtype=float)


# %% [markdown]
# ## Step 1: how far apart are two people?
# Look only at films **both** have seen (co-rated films). Take the gap between their ratings
# on each film, ignore the sign, and average. Small number = similar taste.
#
# By hand for Ana and Eli on their 5 shared films:
# gaps are 0, 1, 1, 1, 0, so the average gap is 3 / 5 = 0.6.

# %%
def taste_gap(table, a, b):
    """Average absolute rating gap between persons a and b over films both have rated."""
    both = ~np.isnan(table[a]) & ~np.isnan(table[b])
    if not both.any():
        return float("inf")          # nothing in common: we cannot compare them
    return float(np.abs(table[a, both] - table[b, both]).mean())


def all_gaps(table, person):
    """Gap from `person` to everyone else (the person's own entry is infinity)."""
    gaps = np.array([taste_gap(table, person, other) for other in range(table.shape[0])])
    gaps[person] = float("inf")
    return gaps


# %% [markdown]
# ## Step 2: predict a missing rating
# Find the `k` people with the smallest gap who *have* seen the film, and average their ratings.

# %%
def predict(table, person, film, k=1):
    gaps = all_gaps(table, person)
    seen = ~np.isnan(table[:, film])
    gaps = np.where(seen, gaps, np.inf)
    nearest = np.argsort(gaps)[:k]
    nearest = nearest[np.isfinite(gaps[nearest])]
    if len(nearest) == 0:
        return float("nan")
    return float(table[nearest, film].mean())


def recommend(table, person, k=1):
    """Unseen film with the highest predicted rating, and that prediction."""
    best, best_score = None, -1.0
    for film in np.where(np.isnan(table[person]))[0]:
        score = predict(table, person, film, k)
        if score > best_score:
            best, best_score = int(film), score
    return best, best_score


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    gaps = all_gaps(RATINGS, 0)
    for name, g in zip(PEOPLE, gaps):
        print(f"gap Ana -> {name:5s} {g:.2f}")
    print("Ana's predicted rating for Night Chase, 1 neighbour :", predict(RATINGS, 0, 5, k=1))
    print("Ana's predicted rating for Night Chase, 2 neighbours:", predict(RATINGS, 0, 5, k=2))
    print("recommendation for Ana:", FILMS[recommend(RATINGS, 0)[0]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(np.ma.masked_invalid(RATINGS), cmap="viridis", vmin=1, vmax=5)
    axes[0].set_xticks(range(6), [f.replace(" ", "\n") for f in FILMS], fontsize=7)
    axes[0].set_yticks(range(5), PEOPLE)
    for i in range(5):
        for j in range(6):
            axes[0].text(j, i, "?" if np.isnan(RATINGS[i, j]) else f"{RATINGS[i, j]:.0f}", ha="center", va="center", color="w")
    axes[0].set_title("Ratings table (1 to 5, ? = not seen)")
    fig.colorbar(im, ax=axes[0], label="rating (stars)")
    axes[1].bar(PEOPLE[1:], gaps[1:], color=["#2a9d8f", "#d9534f", "#d9534f", "#2a9d8f"])
    axes[1].set_ylabel("average rating gap to Ana (stars)")
    axes[1].set_title("Who is closest to Ana? (smaller = closer)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ratings_and_gaps.png", dpi=120)

# %% [markdown]
# ## Where this stops
# The gap only works when two people share films. With thousands of films and few ratings
# each, we need a better similarity (Session 2) and a model of hidden tastes (Session 3).
