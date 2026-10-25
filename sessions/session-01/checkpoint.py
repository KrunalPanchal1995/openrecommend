"""Checkpoint for Session 1. Prints PASS or FAIL."""
import numpy as np


def run_checks(L):
    """L is the lesson module (or notebook namespace). Returns a list of (label, passed)."""
    checks = []
    T = L.RATINGS
    checks.append(("table is 5 people x 6 films", T.shape == (5, 6)))
    checks.append(("exactly one rating is missing (Ana, Night Chase)", int(np.isnan(T).sum()) == 1 and bool(np.isnan(T[0, 5]))))
    checks.append(("all ratings are whole numbers from 1 to 5", bool(np.all((T[~np.isnan(T)] >= 1) & (T[~np.isnan(T)] <= 5)))))
    checks.append(("worked example: gap Ana-Eli is (0+1+1+1+0)/5 = 0.6", abs(L.taste_gap(T, 0, 4) - 0.6) < 1e-9))
    checks.append(("gap Ana-Ben is 1.0 (hand calculation)", abs(L.taste_gap(T, 0, 1) - 1.0) < 1e-9))
    checks.append(("gap Ana-Chen is 15/5 = 3.0", abs(L.taste_gap(T, 0, 2) - 3.0) < 1e-9))
    checks.append(("gap is symmetric", abs(L.taste_gap(T, 1, 3) - L.taste_gap(T, 3, 1)) < 1e-12))
    checks.append(("gap of a person with themselves is 0", L.taste_gap(T, 2, 2) == 0.0))
    g = L.all_gaps(T, 0)
    checks.append(("Eli is the closest person to Ana", int(np.argmin(g)) == 4))
    checks.append(("worked example: 1 neighbour predicts Ana -> Night Chase = 4", L.predict(T, 0, 5, k=1) == 4.0))
    checks.append(("2 neighbours predict (4 + 5) / 2 = 4.5", L.predict(T, 0, 5, k=2) == 4.5))
    checks.append(("recommendation for Ana is Night Chase", L.recommend(T, 0)[0] == 5))
    checks.append(("exercise answer: gap Ben-Eli on all 6 films is 3/6 = 0.5", abs(L.taste_gap(T, 1, 4) - 0.5) < 1e-9))
    checks.append(("using all 4 people gives (5+1+2+4)/4 = 3.0", L.predict(T, 0, 5, k=4) == 3.0))
    ana, eli = T[0, :5], T[4, :5]
    checks.append(("snippet: Pearson of Ana and Eli on 5 films is 0.888", round(float(np.corrcoef(ana, eli)[0, 1]), 3) == 0.888))
    # edge: two people with no film in common
    E = np.array([[5, np.nan], [np.nan, 4]])
    checks.append(("edge: no shared films gives an infinite gap", L.taste_gap(E, 0, 1) == float("inf")))
    checks.append(("edge: nobody to learn from gives NaN, not a crash", bool(np.isnan(L.predict(E, 0, 1)))))
    return checks


# --- runner ---
if __name__ == "__main__":
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parent / "lesson.py"
    spec = importlib.util.spec_from_file_location("lesson1", path)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    ok = True
    for label, passed in run_checks(lesson):
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= bool(passed)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
