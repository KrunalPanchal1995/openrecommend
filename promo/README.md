# What you are building toward

A map of 150 synthetic users (dots) and 40 items (squares) takes shape as the model learns hidden tastes sweep by sweep, while the held-out error falls from about 0.84 to about 0.64 stars. Then a brand-new user rates six items one at a time: a star moves across the map and the top-3 recommendation list (arrows and bars) changes after every rating. Your goal over the coming sessions: build that yourself, with bias terms and a map you can read.

![Target: a learned taste map, a recommendation list that updates, and held-out error falling over ALS sweeps](media/target.gif)

*Target: a learned taste map, a recommendation list that updates, and held-out error falling over ALS sweeps*

![Still frame of the Target: the new user has rated 4 items](media/target_still.png)

*Still frame of the Target: the new user has rated 4 items*

These results come from a private reference build. The code is not published; the roadmap shows the path so you can build your own.

Live links:

- [Math Visual Studio (Studio root)](https://krunalpanchal1995.github.io/studio/)
