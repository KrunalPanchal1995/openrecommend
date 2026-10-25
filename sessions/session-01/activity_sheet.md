# Activity sheet: openrecommend, Session 1: People who agree keep agreeing

Name: ____________________   Date: ____________

**Objectives**

- Read a ratings table that has missing cells.
- Compute the average rating gap between two people on the films both have seen.
- Predict a missing rating from the nearest person, or from the k nearest people.
- Explain why 'people who liked the same things like the same new things' is a guess, not a fact.

## Before you start: predict
In your own words, what do you think will happen when you change the main slider in the live app?

______________________________________________________________

## Worked example (fill in as you go)
Ana has not seen Night Chase. Use the table to guess her rating with 1 nearest person and with 2 nearest people.

1. ______________________________  (hint: Compare Ana with Eli on the 5 films both rated: the gaps are 0, 1, 1, ...)
2. ______________________________  (hint: In the same way Ben has gap 5 / 5 = 1.0, and Chen and Dia each have ga...)
3. ______________________________  (hint: The nearest person is Eli (0.6). Eli rated Night Chase 4, so the guess...)
4. ______________________________  (hint: The two nearest people are Eli (0.6) and Ben (1.0). Ben rated Night Ch...)

Final answer: ____________________

## Questions
**A. Conceptual.** Why not simply recommend the film with the highest average rating to everybody? Name one situation where the neighbour method gives a better answer.

Answer: ______________________________________________


**B. Numerical.** Both Ben and Eli rated all 6 films: Ben gave 4, 5, 2, 1, 5, 5 and Eli gave 5, 5, 2, 1, 4, 4. Find the average rating gap between them and say whether they are similar. Give the unit.

Answer: ______________________________________________


**C. Graph and simulation interpretation.** In the bar chart of gaps to Ana, which two people are closest, and what does the guess become if we use all four people instead of the 2 nearest?

Answer: ______________________________________________


**D. Design and optimisation challenge.** Design: you may use k = 1, 2 or 4 neighbours to guess Ana's rating for Night Chase. Compare the three guesses and choose one. State the trade-off.

Answer: ______________________________________________

## Self-check
- [ ] I can explain the key idea to a friend
- [ ] My checkpoint printed PASS
- [ ] I tried one change of my own and wrote what happened: ____________________
