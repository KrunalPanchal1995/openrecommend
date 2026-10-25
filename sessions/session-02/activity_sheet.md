# Activity sheet: openrecommend, Session 2: Measure similarity, predict, and test honestly

Name: ____________________   Date: ____________

**Objectives**

- Compute cosine and Pearson similarity between two users on their co-rated items.
- Predict a missing rating as the user's own average plus a similarity-weighted average of neighbours' surprises.
- Explain the difference between user-based and item-based prediction.
- Evaluate a recommender by hiding known ratings and reporting RMSE and top-N hit rate.

## Before you start: predict
In your own words, what do you think will happen when you change the main slider in the live app?

______________________________________________________________

## Worked example (fill in as you go)
User A rated four items 5, 3, 4, 4 and has not rated item 5. User B rated them 3, 1, 2, 2 and gave item 5 a 3. Find cosine and Pearson between A and B on the 4 shared items, then predict A's rating for item 5 using B alone.

1. ______________________________  (hint: Dot product: 5*3 + 3*1 + 4*2 + 4*2 = 15 + 3 + 8 + 8 = 34. Lengths: sqr...)
2. ______________________________  (hint: Pearson: A's mean is 4, so A - 4 = (1, -1, 0, 0). B's mean on the shar...)
3. ______________________________  (hint: B's mean over all 5 of its ratings is (3 + 1 + 2 + 2 + 3) / 5 = 2.2, s...)
4. ______________________________  (hint: With one neighbour of similarity 1 the weighted average of surprises i...)

Final answer: ____________________

## Questions
**A. Conceptual.** Why does Pearson similarity separate users better than cosine when all ratings are between 1 and 5? What does a negative Pearson mean?

Answer: ______________________________________________


**B. Numerical.** Two users share 3 items. User x rated them 5, 3, 1 and user y rated them 4, 4, 1. Compute the cosine and the Pearson similarity (no unit).

Answer: ______________________________________________


**C. Graph and simulation interpretation.** In the right panel, the error falls from 1.229 at k = 1. Roughly where is the best k, and why does the error rise again for very large k?

Answer: ______________________________________________


**D. Design and optimisation challenge.** Design: the checkpoint reports hit rate at 5 of 0.271 for random, 0.747 for popularity, 0.621 for user-based and 0.678 for item-based, but the RMSE of user-based is much better than popularity-like baselines. Which method would you ship for a 'top 5 picks' row, and what would you change in the test before trusting the answer?

Answer: ______________________________________________

## Self-check
- [ ] I can explain the key idea to a friend
- [ ] My checkpoint printed PASS
- [ ] I tried one change of my own and wrote what happened: ____________________
