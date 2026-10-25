"""Checkpoint for Session 2. Prints PASS or FAIL."""
import numpy as np


def run_checks(L):
    """L is the lesson module (or notebook namespace). Returns a list of (label, passed)."""
    checks = []
    nan = np.nan

    # --- worked example by hand: 4 users x 5 items, user 0 has not rated item 4
    T = np.array([[5, 3, 4, 4, nan],
                  [3, 1, 2, 2, 3],
                  [1, 5, 2, 4, 2],
                  [4, 3, 4, 3, 5]], dtype=float)
    M = ~np.isnan(T)
    R = np.nan_to_num(T)
    cos = L.similarity(R, M, "cosine")
    pea = L.similarity(R, M, "pearson")
    hand_cos = 34 / (np.sqrt(25 + 9 + 16 + 16) * np.sqrt(9 + 1 + 4 + 4))
    checks.append((f"cosine of users 0 and 1 on 4 shared items is 34/(8.124*4.243) = 0.986 (got {cos[0, 1]:.3f})", abs(cos[0, 1] - hand_cos) < 1e-9 and abs(cos[0, 1] - 0.986) < 5e-4))
    checks.append(("Pearson of users 0 and 1 is exactly 1 (same pattern, shifted)", abs(pea[0, 1] - 1.0) < 1e-12))
    checks.append(("Pearson of users 0 and 2 is -4/(1.414*3.162) = -0.894", abs(pea[0, 2] + 0.894) < 5e-4))
    checks.append(("Pearson of users 0 and 3 is 0.707", abs(pea[0, 3] - 0.7071) < 5e-4))
    checks.append(("similarity matrix is symmetric", np.allclose(pea, pea.T)))
    pred = L.predict_all(R, M, pea, k=3)
    # mean_0 = 4; user 1 contributes 1.0 * (3 - 2.2); user 3 contributes 0.7071 * (5 - 3.8); user 2 is negative and ignored
    hand = 4 + (1.0 * 0.8 + 0.7071 * 1.2) / (1.0 + 0.7071)
    checks.append((f"worked example: prediction for user 0, item 4 is 4 + (0.8 + 0.8485)/1.7071 = 4.966 (got {pred[0, 4]:.3f})", abs(pred[0, 4] - hand) < 1e-3 and abs(pred[0, 4] - 4.966) < 5e-4))
    pred1 = L.predict_all(R, M, pea, k=1)
    checks.append(("with k = 1 only the closest user counts: 4 + 0.8 = 4.8", abs(pred1[0, 4] - 4.8) < 1e-9))
    checks.append(("predictions stay inside 1..5", bool(pred.min() >= 1 and pred.max() <= 5)))

    # --- edges
    few = np.array([[5, 4, nan, nan], [4, 5, 3, nan]], dtype=float)
    Mf = ~np.isnan(few)
    checks.append(("edge: only 2 shared items gives similarity 0 (below min_common = 3)", L.similarity(np.nan_to_num(few), Mf)[0, 1] == 0.0))
    lone = L.predict_all(np.nan_to_num(few), Mf, L.similarity(np.nan_to_num(few), Mf), k=5)
    checks.append(("edge: no neighbours means the user's own mean (4.5)", abs(lone[0, 2] - 4.5) < 1e-12))
    flat = np.array([[3, 3, 3, 3], [1, 2, 3, 4]], dtype=float)
    checks.append(("edge: a user who rates everything 3 gives Pearson 0, not NaN", L.similarity(flat, np.ones_like(flat, bool))[0, 1] == 0.0))
    ii = L.predict_item_based(R, M, k=3)
    checks.append(("item-based = user-based code on the transposed table", np.allclose(ii, L.predict_all(R.T, M.T, L.similarity(R.T, M.T), 3).T)))

    # --- assessment answers and the app challenge
    X = np.array([[5, 3, 1], [4, 4, 1]], dtype=float)
    MX = np.ones_like(X, dtype=bool)
    checks.append(("exercise B: cosine 0.971 and Pearson 0.866 for (5,3,1) and (4,4,1)",
                   round(L.similarity(X, MX, "cosine")[0, 1], 3) == 0.971 and round(L.similarity(X, MX)[0, 1], 3) == 0.866))
    Y = np.array([[5, 3, 4, 4], [3, 5, 4, 4]], dtype=float)
    MY = np.ones_like(Y, dtype=bool)
    checks.append(("app challenge: B = (3,5,4,4) has cosine 0.939 but Pearson -1",
                   round(L.similarity(Y, MY, "cosine")[0, 1], 3) == 0.939 and abs(L.similarity(Y, MY)[0, 1] + 1) < 1e-9))

    # --- the synthetic world and honest evaluation
    R, M = L.make_ratings()
    train, test = L.split_observed(M)
    checks.append(("seeded world is reproducible", np.array_equal(R, L.make_ratings()[0])))
    checks.append(("world: 100 users x 60 items with 2034 observed ratings", R.shape == (100, 60) and int(M.sum()) == 2034))
    checks.append(("2034 observed ratings split into 434 hidden and 1600 for training", int(test.sum()) == 434 and int(train.sum()) == 1600))
    checks.append(("train and test cells never overlap", not (train & test).any()))
    checks.append(("ratings are whole stars 1..5", bool(set(np.unique(R)) <= {1.0, 2.0, 3.0, 4.0, 5.0})))
    gm = L.rmse(L.global_mean_prediction(R, train), R, test)
    im = L.rmse(L.item_mean_prediction(R, train), R, test)
    Pu = L.predict_user_based(R, train)
    Pi = L.predict_item_based(R, train)
    Pc = L.predict_user_based(R, train, method="cosine")
    ru, ri, rc = (L.rmse(p, R, test) for p in (Pu, Pi, Pc))
    checks.append((f"RMSE: global mean 1.242 > item mean 1.179 (got {gm:.3f}, {im:.3f})", round(gm, 3) == 1.242 and round(im, 3) == 1.179))
    checks.append((f"RMSE: user-based Pearson 0.873, item-based 0.906, cosine 0.937 (got {ru:.3f}, {ri:.3f}, {rc:.3f})", (round(ru, 3), round(ri, 3), round(rc, 3)) == (0.873, 0.906, 0.937)))
    checks.append(("neighbour methods beat the item-mean baseline; Pearson beats cosine", ru < im and ri < im and ru < rc))
    # no leakage: changing the hidden ratings must not change the predictions
    R2 = R.copy()
    R2[test] = 1.0
    checks.append(("no leakage: hidden ratings do not influence predictions", np.allclose(L.predict_user_based(R2, train), Pu)))
    # the cosine values are all positive and bunched; Pearson spreads out
    iu = np.triu_indices(100, 1)
    Sc, Sp = L.similarity(R, train, "cosine")[iu], L.similarity(R, train)[iu]
    checks.append(("cosine on 1..5 ratings is never negative; Pearson takes both signs", bool(Sc.min() >= 0) and bool(Sp.min() < 0 < Sp.max())))
    checks.append(("Pearson spread is larger than cosine spread", Sp.std() > Sc.std()))
    ks = [1, 2, 5, 10, 20, 40, 80]
    S = L.similarity(R, train)
    curve = [L.rmse(L.predict_all(R, train, S, k), R, test) for k in ks]
    checks.append((f"RMSE against k = 1,2,5,10,20,40,80 is 1.229, 1.163, 1.059, 0.921, 0.873, 0.862, 0.872 (got {[round(c, 3) for c in curve]})",
                   [round(c, 3) for c in curve] == [1.229, 1.163, 1.059, 0.921, 0.873, 0.862, 0.872] and int(np.argmin(curve)) == 5))
    hr = (L.random_hit_rate(R, train, test), L.hit_rate(L.popularity_score(train), R, train, test), L.hit_rate(Pu, R, train, test), L.hit_rate(Pi, R, train, test))
    checks.append((f"hit rate@5: random 0.271, popularity 0.747, user-based 0.621, item-based 0.678 (got {hr[0]:.3f}, {hr[1]:.3f}, {hr[2]:.3f}, {hr[3]:.3f})",
                   tuple(round(h, 3) for h in hr) == (0.271, 0.747, 0.621, 0.678)))
    checks.append(("every method beats random; popularity is a strong baseline on this exposure-biased test", min(hr[1:]) > hr[0] and hr[1] > hr[2]))
    return checks


# --- runner ---
if __name__ == "__main__":
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parent / "lesson.py"
    spec = importlib.util.spec_from_file_location("lesson2", path)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    ok = True
    for label, passed in run_checks(lesson):
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= bool(passed)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
