# Roadmap: the path to the openrecommend Target

This is the path, not the code. Each milestone says what you learn, what you build, and how you know it works. Pick one and open a pull request. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Session 4: Add bias terms

**You will learn:** How a per-user and per-item offset lets the hidden factors focus on taste instead of 'this item is popular'.

**You will build:** ALS where the model is mu plus a user bias plus an item bias plus the factor match, each bias solved in the same alternating loop.

**Checkpoint (not met by the shipped code):** Validation RMSE of at most 0.72 stars on the Session 3 split (the shipped model's best is 0.758).

**Difficulty:** intermediate

## Session 5: A smarter cold start

**You will learn:** Combining a new user's few ratings with a prior from popularity instead of switching between them.

**You will build:** A recommender for new users that blends fold-in scores with item average ratings.

**Checkpoint (not met by the shipped code):** Precision at 3 of at least 0.80 for new users who gave exactly 1 rating (the shipped fold-in gives 0.72, popularity 0.74).

**Difficulty:** intermediate

## Session 6: Variety-aware ranking

**You will learn:** Re-ranking so that the lists across all users cover more of the catalogue without losing accuracy.

**You will build:** A re-ranker that penalises items that have already been recommended to many users.

**Checkpoint (not met by the shipped code):** Coverage of at least 0.85 with precision at 5 of at least 0.93 (the shipped blend gives 0.67 at best, and 0.46 when it keeps precision above 0.96).

**Difficulty:** advanced

## Session 7: Closed-loop simulation

**You will learn:** What happens over many rounds when users only rate what they were shown, and how to measure drift in what the system shows.

**You will build:** A simulation of 8 rounds where every user is shown 5 unseen items, rates them with the true rating, and the model is refit.

**Checkpoint (not met by the shipped code):** Round-8 coverage of the popularity-only policy stays below 0.30 while a personalised policy reaches at least 0.80.

**Difficulty:** advanced

## The final step

Combine the milestones: a recommender with bias terms and a variety-aware ranking, tested on hidden ratings, with a 2D taste map and an animation of the error falling over ALS sweeps, like the Target. If your result looks like the Target animation, open a pull request and tell us how you did it.
