# Activity sheet: openrecommend, Session 3: Hidden tastes, cold start and shrinking variety

Name: ____________________   Date: ____________

**Objectives**

- Fit a matrix factorisation by alternating least squares and watch the training error fall.
- Choose the rank k and the regularisation lambda on validation data, and report once on test data.
- Recognise overfitting at large k and data leakage, and measure how much each distorts the error.
- Measure the cold-start problem and how popularity fallback and fold-in compare, and measure how popularity weighting shrinks the variety of the recommendations.

## Before you start: predict
In your own words, what do you think will happen when you change the main slider in the live app?

______________________________________________________________

## Worked example (fill in as you go)
One ridge step by hand. Two items have one-number vectors q = 1 and q = 2. A user (with mu = 3) rated them 4 and 5, so the centred ratings are 1 and 2. Find the user's vector p for lambda = 1 and for lambda = 0, and the predicted rating for an item with q = 2.

1. ______________________________  (hint: For one hidden factor, p = (sum of q * centred rating) / (sum of q squ...)
2. ______________________________  (hint: Numerator: 1*1 + 2*2 = 5. Sum of q squared: 1 + 4 = 5....)
3. ______________________________  (hint: With lambda = 1: p = 5 / (5 + 1) = 0.8333. With lambda = 0: p = 5 / 5 ...)
4. ______________________________  (hint: Prediction for an item with q = 2: with lambda = 1 it is 3 + 0.8333*2 ...)

Final answer: ____________________

## Questions
**A. Conceptual.** Why can the training error go to zero while the model gets worse on new ratings? What is the role of lambda?

Answer: ______________________________________________


**B. Numerical.** Redo the worked example with the user's ratings 3 and 5 (centred ratings 0 and 2, mu = 3). Find p for lambda = 1 and lambda = 0 and the predicted rating for q = 2.

Answer: ______________________________________________


**C. Graph and simulation interpretation.** In the grid of validation errors, where is the minimum, and what happens in the column lambda = 20 for every k?

Answer: ______________________________________________


**D. Design and optimisation challenge.** Design: the onboarding screen of a new app can ask a new user for m ratings (0, 1, 2, 3, 5 or 10). Using precision at 3 for ALS fold-in (0.74, 0.72, 0.89, 0.85, 0.93, 0.94) and for popularity (0.74, 0.74, 0.75, 0.75, 0.73, 0.75), choose m and justify it. What would you do before the user has rated anything?

Answer: ______________________________________________

## Self-check
- [ ] I can explain the key idea to a friend
- [ ] My checkpoint printed PASS
- [ ] I tried one change of my own and wrote what happened: ____________________
