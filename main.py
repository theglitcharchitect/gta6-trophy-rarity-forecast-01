"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01. The program is
run once per case (python3.13, standard library only, no network): it
reads every <id>.in in the input directory and writes <id>.out to the
output directory, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each
trophy's earn rate, in percent, at the case's target.

Model, in one paragraph per stage:

1.  A case history is a sequence of reads of one Exophase page. The page
    only changes when Exophase refreshes its counts, so the history is
    compressed to its distinct states first: consecutive reads with the
    same players_tracked and the same rates are one observation, stamped
    with the first read that showed it.

2.  Every rate is a ratio, earners / players_tracked, and the two sides
    move for different reasons, so they are forecast separately and the
    rate is rebuilt as a ratio at the target. Earners only grow (nobody
    un-earns a trophy), and players_tracked only grows (the tracker only
    adds profiles), but both grow in steps at refreshes and both slow as
    the launch surge passes.

3.  Growth is fitted in log time: log(count) against log(elapsed + TAU).
    That family rises fast in the first hours and flattens over days,
    which is the shape a launch week takes, and a straight line in those
    coordinates is all the data can support: the final case holds about
    two days of refreshes, and a richer curve would be fitted to noise.
    Each trophy's earners slope is shrunk toward its tier's mean slope,
    because a rare trophy's own series is a handful of small counts.

4.  The median lands on a refreshed page, not on a smooth path: the
    answer is the page as of Exophase's last refresh before the target,
    and every observed state is itself a refreshed page, so a fit through
    the states is already a fit through refreshed states.

5.  Spread is stated on the logit scale, where the score lives. It
    widens with the square root of the horizon, carries a constant term
    for refresh timing (when the last refresh before the target fell,
    and how large it was, are both unknown), widens again for rare
    trophies (a move from 1% to 2% costs like 33% to 50%), and widens
    further when the history shows a long frozen stretch, because a
    refresh the program has not seen is then the whole story.

Fallbacks keep every case valid: a history with a single distinct state
has no measurable growth, so players are projected with a refresh prior
(a frozen launch-night page broke to about six times the players at its
first refresh; the prior uses a fourfold jump, discounted by the earners
growth expected over the frozen stretch) and earners with a pooled
log-time slope; a trophy with no usable rate gets a small prior rate.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Log-time offset, in hours. One value for players and earners so the two
# fits share a clock and their ratio stays interpretable.
TAU_HOURS = 36.0

# A rate is clamped inside these bounds before any logit is taken, and a
# forecast median is clamped to them as well: the observed page prints
# two decimals, so 0.01 and 99.99 are its own extremes, and a median
# stated as a flat 100 would claim a certainty no week-one pool has.
RATE_LO = 0.01
RATE_HI = 99.99
MEDIAN_HI = 96.5

# Spread, in logit units: SIGMA_PER_ROOT_DAY * sqrt(days) for the trend
# part, plus SIGMA_REFRESH for the refresh-timing part.
SIGMA_PER_ROOT_DAY = 0.50
SIGMA_REFRESH = 0.35


def parse_time(text):
    """ISO 8601 UTC text (with or without fractional seconds) to epoch."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def rounded4(value):
    """Round to four decimals, half away from zero, deterministically."""
    scaled = math.floor(abs(value) * 10000.0 + 0.5) / 10000.0
    result = scaled if value >= 0 else -scaled
    return int(result) if float(result).is_integer() else result


def distinct_states(history):
    """Compress reads to the moments the page actually changed.

    Returns a list of (epoch_seconds, players_tracked, rates) with the
    timestamp of the first read that showed each state.
    """
    states = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if states and states[-1][1] == players and states[-1][2] == rates:
            continue
        states.append((parse_time(read["at"]), players, list(rates)))
    return states


def fit_log_time(times_h, values_log, tau_h):
    """Least squares line of values_log against log(times_h + tau_h).

    Returns (intercept, slope). With fewer than two points, or no time
    spread, the slope is 0 and the intercept is the last value.
    """
    if not values_log:
        return 0.0, 0.0
    if len(values_log) < 2:
        return values_log[-1], 0.0
    xs = [math.log(t + tau_h) for t in times_h]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(values_log) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return values_log[-1], 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values_log))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def blended_slope(times_h, values_log, tau_h):
    """A growth slope for projection: the log-time regression over the
    second half of the observed span, falling back to the full window
    when the second half holds fewer than two states. The first states
    of a launch are a pre-release tester pool orders of magnitude
    smaller than the launch pool (638 tracked players, most trophies
    sitting at the same handful-of-people rate), and growth decelerates
    from there, so a slope fitted through those states describes a
    regime that is already over and badly overstates the coming days.
    """
    if len(values_log) < 2:
        return 0.0
    midpoint = 0.5 * (times_h[0] + times_h[-1])
    recent_pairs = [(t, v) for (t, v) in zip(times_h, values_log) if t >= midpoint]
    if len(recent_pairs) < 2:
        recent_pairs = list(zip(times_h, values_log))
    _intercept, slope = fit_log_time([t for (t, _v) in recent_pairs],
                                     [v for (_t, v) in recent_pairs], tau_h)
    return slope


def tier_of(trophy):
    return trophy.get("tier") or "bronze"


def forecast_players(states, start_epoch, target_h):
    """Project players_tracked at the target hour.

    With two or more distinct player counts, a log-time fit. With one,
    the refresh prior: a page frozen since the first read is expected to
    break to a multiple of its count by the target.
    """
    points = [(t, p) for (t, p, _r) in states if p]
    last_players = points[-1][1] if points else 0
    if len(points) >= 2:
        times_h = [(t - start_epoch) / 3600.0 for (t, _p) in points]
        logs = [math.log(p) for (_t, p) in points]
        slope = min(1.5, max(0.0, blended_slope(times_h, logs, TAU_HOURS)))
        # Anchor the projection at the last observed count, not at the
        # fitted line's level: the line's slope carries the trend, the
        # last state carries the level, and mixing the fitted level back
        # in would let early launch-night counts move the target.
        last_h = times_h[-1]
        growth = ((target_h + TAU_HOURS) / (last_h + TAU_HOURS)) ** slope
        return max(float(last_players), float(last_players) * growth), True
    if last_players:
        return float(last_players) * 4.0, False
    return 0.0, False


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    states = distinct_states(history)
    start_epoch = states[0][0]
    checkpoint_epoch = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = (checkpoint_epoch - start_epoch) / 3600.0 + case["horizon_minutes"] / 60.0

    players_target, players_fitted = forecast_players(states, start_epoch, target_h)
    last_state = states[-1]
    last_players = last_state[1] or 0
    if players_target <= 0:
        players_target = float(last_players) if last_players else 1.0

    # Earners per trophy at each distinct state, and each tier's pooled
    # log-time slope over the states where the tier has earners at all.
    state_times_h = [(t - start_epoch) / 3600.0 for (t, _p, _r) in states]
    earners_series = []
    for idx in range(len(trophies)):
        series = []
        for (_t, players, rates) in states:
            rate = rates[idx] if idx < len(rates) else None
            if rate is None or not players:
                series.append(None)
            else:
                series.append(rate / 100.0 * players)
        earners_series.append(series)

    tier_slopes = {}
    tier_counts = {}
    raw_slopes = []
    for idx, trophy in enumerate(trophies):
        usable = [(h, e) for (h, e) in zip(state_times_h, earners_series[idx]) if e is not None]
        if len(usable) >= 2 and usable[-1][0] > usable[0][0]:
            slope = blended_slope([h for (h, _e) in usable],
                                  [math.log(e + 0.5) for (_h, e) in usable], TAU_HOURS)
            slope = min(1.2, max(0.0, slope))
        else:
            slope = None
        raw_slopes.append(slope)
        if slope is not None:
            tier = tier_of(trophy)
            tier_slopes[tier] = tier_slopes.get(tier, 0.0) + slope
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
    tier_mean = {t: tier_slopes[t] / tier_counts[t] for t in tier_slopes}
    all_slopes = [s for s in raw_slopes if s is not None]
    global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.35

    # How long the page sat frozen at its last state before the
    # checkpoint: a long freeze means the next refresh carries hours of
    # unseen earning and a large intake of new players at once.
    if len(states) >= 2:
        frozen_h = (checkpoint_epoch - states[-1][0]) / 3600.0
        gaps = [(states[i + 1][0] - states[i][0]) / 3600.0 for i in range(len(states) - 1)]
        typical_gap_h = sorted(gaps)[len(gaps) // 2]
    else:
        frozen_h = (checkpoint_epoch - states[-1][0]) / 3600.0
        typical_gap_h = 0.0
    freeze_extra = 0.0
    if typical_gap_h > 0 and frozen_h > 2.0 * typical_gap_h:
        freeze_extra = 0.45
    elif len(states) == 1 and frozen_h >= 1.0:
        freeze_extra = 0.85

    rows = []
    for idx, trophy in enumerate(trophies):
        series = earners_series[idx]
        usable = [(h, e) for (h, e) in zip(state_times_h, series) if e is not None]
        last_rate = None
        for (_t, _p, rates) in reversed(states):
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        if usable:
            last_earners = usable[-1][1]
            tier = tier_of(trophy)
            pooled = tier_mean.get(tier, global_mean)
            own = raw_slopes[idx]
            if own is None:
                slope = pooled
            else:
                # Shrink the trophy's own slope toward its tier's: the
                # fewer and smaller its counts, the less its own series
                # can be trusted. Weight by log earners, capped.
                weight = min(0.85, max(0.15, math.log10(last_earners + 10.0) / 4.0))
                slope = weight * own + (1.0 - weight) * pooled
            # Same anchoring as the players projection: grow the last
            # observed earners count at the shrunk slope, so the level
            # is the page's own and only the trend is modelled.
            growth = ((target_h + TAU_HOURS) / (usable[-1][0] + TAU_HOURS)) ** slope
            earners_target = (last_earners + 0.5) * growth - 0.5
            earners_target = max(last_earners, earners_target)
            earners_target = min(earners_target, players_target)
        elif last_rate is not None and last_players:
            earners_target = last_rate / 100.0 * players_target
        else:
            earners_target = 0.005 * players_target

        if players_fitted:
            median = 100.0 * earners_target / players_target
        elif last_rate is not None:
            # Single-state history: the ratio of the two priors. Earners
            # grow over the frozen stretch plus the horizon at the pooled
            # slope while the player count jumps, so a frozen page's
            # rates are diluted at the refresh it has not shown yet.
            median = 100.0 * earners_target / players_target if players_target else last_rate
        else:
            median = 0.5
        if last_rate is not None and last_rate >= 99.9:
            # A trophy essentially every tracked player holds: its
            # earners track the player count itself, so the ratio model
            # already keeps it high; this floor only guards the case
            # where a noisy slope would drag it away from that fact.
            median = min(MEDIAN_HI, max(median, 85.0))
        median = min(MEDIAN_HI, max(0.02, median))

        sigma = math.sqrt((SIGMA_PER_ROOT_DAY * math.sqrt(horizon_days)) ** 2
                          + SIGMA_REFRESH ** 2 + freeze_extra ** 2)
        if median < 2.0:
            sigma *= 1.35
        elif median < 5.0:
            sigma *= 1.20
        if not players_fitted:
            sigma = max(sigma, 1.05)

        centre = logit_of(median)
        quantiles = []
        for z in Z_SCORES:
            value = rate_of(centre + z * sigma)
            quantiles.append(rounded4(min(99.99, max(0.01, value))))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
