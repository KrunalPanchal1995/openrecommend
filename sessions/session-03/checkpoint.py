"""Checkpoint for Session 3. Prints PASS or FAIL."""
import numpy as np


def run_checks(L):
    """L is the lesson module (or notebook namespace). Returns a list of (label, passed)."""
    checks = []

    # --- hand calculation of one ridge step (the heart of ALS)
    Q = np.array([[1.0], [2.0]])
    p = L.fold_in(np.array([3.0 + 1.0, 3.0 + 2.0]), np.array([0, 1]), 3.0, Q, lam=1.0)
    checks.append((f"worked example: ridge step p = (1*1 + 2*2) / (1 + 4 + 1) = 0.8333 (got {p[0]:.4f})", abs(p[0] - 5 / 6) < 1e-9))
    p0 = L.fold_in(np.array([4.0, 5.0]), np.array([0, 1]), 3.0, Q, lam=0.0)
    checks.append(("with lambda = 0 the fit is exact: p = 1", abs(p0[0] - 1.0) < 1e-9))
    checks.append(("edge: a user with no ratings gets the zero vector", bool(np.all(L.fold_in(np.array([]), np.array([], int), 3.0, Q, 1.0) == 0))))

    R, train, val, test, cold = L.build_world()
    checks.append(("world: 240 users x 100 items; 40 new users are removed from training", R.shape == (240, 100) and not train[cold].any()))
    checks.append((f"split sizes train 2143, validation 484, test 427 (got {train.sum()}, {val.sum()}, {test.sum()})", (int(train.sum()), int(val.sum()), int(test.sum())) == (2143, 484, 427)))
    checks.append(("splits do not overlap", not ((train & val) | (train & test) | (val & test)).any()))

    # --- ALS: objective falls, errors fall
    objs = [L.objective(R, train, *L.als(R, train, 3, 5.0, i)[:3], 5.0) for i in (1, 2, 3, 5, 8)]
    checks.append(("the regularised objective never rises over iterations", all(b <= a + 1e-6 for a, b in zip(objs, objs[1:]))))
    mu, P, Q, trc, vac = L.als(R, train, 3, 5.0, 15, watch=val)
    checks.append((f"k=3, lambda=5: train RMSE falls 1.066 -> 0.745 (got {trc[0]:.3f} -> {trc[-1]:.3f})", round(trc[0], 3) == 1.066 and round(trc[-1], 3) == 0.745 and trc[-1] < trc[0]))
    checks.append((f"k=3, lambda=5: validation RMSE falls 1.155 -> 0.969 (got {vac[0]:.3f} -> {vac[-1]:.3f})", round(vac[0], 3) == 1.155 and round(vac[-1], 3) == 0.969))

    p2 = L.fold_in(np.array([3.0, 5.0]), np.array([0, 1]), 3.0, Q_hand := np.array([[1.0], [2.0]]), lam=1.0)
    p2z = L.fold_in(np.array([3.0, 5.0]), np.array([0, 1]), 3.0, Q_hand, lam=0.0)
    checks.append(("exercise B: p = 0.667 (lambda 1) and 0.8 (lambda 0); predictions 4.333 and 4.6",
                   abs(p2[0] - 2 / 3) < 1e-9 and abs(p2z[0] - 0.8) < 1e-9 and abs(3 + 2 * p2[0] - 4.3333) < 1e-3 and abs(3 + 2 * p2z[0] - 4.6) < 1e-9))
    checks.append(("worked example predictions: 3 + 0.8333*2 = 4.667 and 3 + 1*2 = 5",
                   abs(3 + 2 * 5 / 6 - 4.6667) < 1e-3 and abs(3 + 2 * p0[0] - 5.0) < 1e-9))
    checks.append(("about 15% of the 200 non-new users' cells are observed",
                   round(int((train | val | test).sum()) / (200 * 100), 2) == 0.15))

    # --- choose k and lambda on validation, report once on test
    ks = [1, 2, 3, 4, 6, 8, 12, 20]
    lams = [0.1, 1.0, 5.0, 20.0]
    V, T = L.sweep(R, train, val, ks, lams)
    a, b = np.unravel_index(np.argmin(V), V.shape)
    checks.append((f"validation minimum 0.758 is at k=4, lambda=1 (got {V[a, b]:.3f} at k={ks[a]}, lambda={lams[b]})", (ks[a], lams[b], round(V[a, b], 3)) == (4, 1.0, 0.758)))
    checks.append(("the minimum is interior: both k=1 and k=20 are worse (at lambda 1)", V[0, 1] > V[a, b] and V[-1, 1] > V[a, b]))
    checks.append((f"overfitting: k=20, lambda=0.1 has train 0.016 but validation 1.103 (got {T[-1, 0]:.3f}, {V[-1, 0]:.3f})", round(T[-1, 0], 3) == 0.016 and round(V[-1, 0], 3) == 1.103))
    V_app = [[1.224, 1.177, 1.136, 1.208], [1.115, 1.017, 1.028, 1.208], [0.906, 0.871, 0.969, 1.208], [0.999, 0.758, 0.949, 1.208],
             [1.456, 0.891, 0.958, 1.208], [1.411, 0.892, 0.959, 1.208], [1.29, 0.932, 0.961, 1.208], [1.103, 0.908, 0.961, 1.208]]
    T_app = [[0.919, 0.924, 0.986, 1.167], [0.71, 0.707, 0.851, 1.167], [0.469, 0.507, 0.745, 1.167], [0.372, 0.376, 0.702, 1.167],
             [0.241, 0.29, 0.694, 1.167], [0.112, 0.239, 0.694, 1.167], [0.033, 0.191, 0.692, 1.167], [0.016, 0.17, 0.691, 1.167]]
    checks.append(("the numbers inside the Session 3 app equal the full sweep", np.allclose(np.round(V, 3), V_app) and np.allclose(np.round(T, 3), T_app)))
    checks.append(("lambda=20 collapses the model: validation 1.208 for every k", bool(np.all(np.round(V[:, 3], 3) == 1.208))))
    checks.append(("at k=20 raising lambda from 0.1 to 1 lowers validation from 1.103 to 0.908", round(V[-1, 0], 3) == 1.103 and round(V[-1, 1], 3) == 0.908))
    checks.append(("training error at lambda=0.1 falls as k grows", bool(np.all(np.diff(T[:, 0]) < 0))))
    checks.append(("regularisation helps at large k: lambda=1 beats lambda=0.1 at k=20", V[-1, 1] < V[-1, 0]))
    mu, P, Q, _, _ = L.als(R, train, ks[a], lams[b])
    pred = L.predict_from(mu, P, Q)
    cnt = train.sum(axis=0)
    base = np.tile(np.where(cnt > 0, (R * train).sum(0) / np.maximum(cnt, 1), mu), (R.shape[0], 1))
    t_als, t_base = L.rmse(pred, R, test), L.rmse(base, R, test)
    checks.append((f"test RMSE of the chosen model 0.773 beats the item-mean baseline 1.041 (got {t_als:.3f}, {t_base:.3f})", round(t_als, 3) == 0.773 and round(t_base, 3) == 1.041))

    # --- failure case: leakage
    mu2, P2, Q2, _, _ = L.als(R, train | test, 20, 0.1)
    mu3, P3, Q3, _, _ = L.als(R, train, 20, 0.1)
    leak, honest = L.rmse(L.predict_from(mu2, P2, Q2), R, test), L.rmse(L.predict_from(mu3, P3, Q3), R, test)
    checks.append((f"leakage: training on test ratings fakes a test RMSE of 0.017; honest value is 1.049 (got {leak:.3f}, {honest:.3f})", round(leak, 3) == 0.017 and round(honest, 3) == 1.049))

    # --- cold start (precision at 3)
    ms = [0, 1, 2, 3, 5, 10]
    ah, ph = L.cold_start_curve(R, train, cold, mu, Q, lams[b], ms)
    checks.append((f"cold start, 0 ratings: ALS falls back to popularity, both 0.74 (got {ah[0]:.2f}, {ph[0]:.2f})", round(ah[0], 2) == 0.74 and ah[0] == ph[0]))
    checks.append((f"cold start, 5 ratings: ALS 0.93 versus popularity 0.73 (got {ah[4]:.2f}, {ph[4]:.2f})", round(ah[4], 2) == 0.93 and round(ph[4], 2) == 0.73))
    checks.append((f"cold start, 10 ratings: ALS 0.94 versus popularity 0.75 (got {ah[5]:.2f}, {ph[5]:.2f})", round(ah[5], 2) == 0.94 and round(ph[5], 2) == 0.75))
    checks.append(("with a single rating ALS (0.72) is not yet better than popularity (0.74)", round(ah[1], 2) == 0.72 and round(ph[1], 2) == 0.74))
    liked = float((R[cold] >= 4).mean())
    checks.append((f"popularity beats picking at random for new users (0.74 against a base rate of 0.43, got {liked:.2f})", round(liked, 2) == 0.43))
    order = L.popularity_order(R, train)
    checks.append((f"cold-start curves: ALS {[round(x, 2) for x in ah]}, popularity {[round(x, 2) for x in ph]}",
                   [round(x, 2) for x in ah] == [0.74, 0.72, 0.89, 0.85, 0.93, 0.94] and [round(x, 2) for x in ph] == [0.74, 0.74, 0.75, 0.75, 0.73, 0.75]))
    first = next(i for i, (x, y) in enumerate(zip(ah, ph)) if x - y >= 0.10)
    checks.append(("mini-project: the smallest m where ALS beats popularity by 0.10 is m = 2", ms[first] == 2 and ah[4] - ph[4] >= 0.15))
    checks.append(("popularity order puts the best-rated well-known item first", cnt[order[0]] >= 5))
    rec = L.recommend_new_user([], [], mu, Q, order, 1.0, n=3)
    checks.append(("recommend_new_user with no ratings returns the 3 most popular items", rec == [int(i) for i in order[:3]]))
    items = np.array([0, 1, 2])
    rec2 = L.recommend_new_user(items, R[cold[0], items], mu, Q, order, 1.0, n=3)
    checks.append(("recommendations never repeat an item the user already rated", not set(rec2) & {0, 1, 2} and len(rec2) == 3))

    # --- popularity bias and diversity
    users = np.arange(len(cold), R.shape[0])
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    div = L.diversity_study(R, train, pred, users, alphas)
    cov = [d[0] for d in div]
    share = [d[1] for d in div]
    prec = [d[2] for d in div]
    checks.append((f"coverage falls from 0.67 to 0.13 as popularity weight goes 0 -> 1 (got {cov[0]:.2f}, {cov[-1]:.2f})", round(cov[0], 2) == 0.67 and round(cov[-1], 2) == 0.13 and bool(np.all(np.diff(cov) < 0))))
    checks.append((f"top-10 share rises from 0.30 to 0.99 (got {share[0]:.2f}, {share[-1]:.2f})", round(share[0], 2) == 0.30 and round(share[-1], 2) == 0.99 and bool(np.all(np.diff(share) > 0))))
    checks.append((f"precision at 5 is 0.95 for personal scores and 0.75 for popularity only (got {prec[0]:.2f}, {prec[-1]:.2f})", round(prec[0], 2) == 0.95 and round(prec[-1], 2) == 0.75))
    checks.append((f"blend a=0.25 keeps precision above 0.96 with coverage 0.46 (got {prec[1]:.3f}, {cov[1]:.2f})", prec[1] > 0.96 and round(cov[1], 2) == 0.46))
    lists = L.top_n_lists(np.arange(10.0)[None, :].repeat(2, 0), np.zeros((2, 10), bool), 3)
    checks.append(("hand check: coverage of two identical lists [9, 8, 7] over 10 items is 3/10", abs(L.coverage(lists, 10) - 0.3) < 1e-12))
    return checks


# --- runner ---
if __name__ == "__main__":
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parent / "lesson.py"
    spec = importlib.util.spec_from_file_location("lesson3", path)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    ok = True
    for label, passed in run_checks(lesson):
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= bool(passed)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
