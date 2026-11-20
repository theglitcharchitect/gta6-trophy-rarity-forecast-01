# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/theglitcharchitect/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/theglitcharchitect/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for how many GTA VI players will hold each PlayStation 5 trophy a week after launch, entered in make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty. `main.py` (Python 3.13, standard library only, no network) reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): for each of a case's trophies, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate on Exophase at the case's target, in percent.

## The model

Facts, from the bounty's own data: Exophase prints each trophy's earn rate as a percent of the players it tracks for the game, and it refreshes those counts on its own schedule. Between refreshes the page is frozen; at a refresh the player count jumps and rates step, usually down, because the new players have earned almost nothing yet. The answer at any target is the page as of the last refresh before it.

The program works from those facts:

1. The history is compressed to its distinct states: consecutive reads with the same player count and rates are one observation.
2. Each rate is rebuilt as a ratio at the target. Earners (rate × players tracked) and the player count are forecast separately, because they move for different reasons: earners grow as players progress, the player count grows as the tracker adds profiles, and a refresh moves the second without moving the first.
3. Growth is a straight line in log time, fitted over the second half of the observed span. The first states are a pre-release tester pool (638 players, most trophies at the same handful-of-people rate), and a slope fitted through them describes a regime that is already over. Each trophy's earners slope is shrunk toward its tier's mean slope, weighted by how large its own counts are.
4. Projections are anchored at the last observed state: the page supplies the level, the fit supplies only the trend.
5. Quantiles are stated on the logit scale, where the score lives. The spread widens with the square root of the horizon, carries a constant term for refresh timing, widens again for rare trophies (1% to 2% costs like 33% to 50% there), and widens further when the page has sat frozen, because an unseen refresh is then the whole story.

Guesses, named as such: a history with a single distinct state has no measurable growth, so the program uses a refresh prior there, a fourfold player jump discounted by the earners growth expected over the frozen stretch. That prior is calibrated on the one refresh the practice cases straddle (638 to 3,868 players on launch night) and is the model's weakest part; the final case, with about two days of refreshes in its history, never uses it.

Failure modes considered: a trophy list that changes at the target (the program forecasts exactly the trophies its case lists, no more), a target that lands just before a refresh rather than just after one (the refresh-timing term in the spread), and small-count noise on rare trophies (the tier shrinkage and the wider spread).

Validation: the practice cases (p00, p01) produce valid, deterministic output, byte-identical across runs, in about 0.05 s per case against the 8 s cap. A synthetic case built to the same schema, twelve refresh states over two days with saturating player and earners growth, also produces valid output with medians that track the constructed rates.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=f5nczt9z&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
