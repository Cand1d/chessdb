"""Does the channel still describe the asset?

The strategy page answers "what would the rules have done". These functions
answer the prior question -- whether the trend the rules are measured against
is still standing -- because a price far below trend is simultaneously the
strongest buy signal and the strongest evidence that the trend is gone. The
price alone cannot separate the two. Two things can:

  * how long the price has stayed outside the band, against how long the
    fitted mean reversion permits (a direct falsification test), and
  * whether the channel, frozen each 31 December, actually held over the
    following year -- out of sample, the way it is really used.

Standard library only.
"""

import math
import random

SECONDS_PER_YEAR = 365.25 * 86400
MC_STEPS = 1_000_000  # weekly steps drawn when profiling excursion lengths
MC_PATHS = 20_000     # paths drawn for the forward hold probability
SEED = 20260101       # fixed so a rebuild does not reshuffle the verdict

# Bins for "what happened after the price sat here", in sigma.
BINS = [(-99, -3), (-3, -2), (-2, -1.5), (-1.5, -1), (-1, 0), (0, 1), (1, 2), (2, 99)]

HOLD_BAND = 2.0       # "inside the channel" means within this many sigma
WEAK_RECORD = 0.6     # held in fewer than this share of years -> caution


def _trend(channel, t, freeze_ts):
    return channel["anchor"] * (1 + channel["cagr"]) ** ((t - freeze_ts) / SECONDS_PER_YEAR)


def signal_series(channel, series, freeze_ts):
    """z in units of sigma for every point, against one frozen channel."""
    return [
        (t, p, math.log(p / _trend(channel, t, freeze_ts)) / channel["sigma"])
        for t, p in series
    ]


MIN_EXCURSIONS = 500  # deep bands are rare; keep drawing until the denominator holds
MAX_MC_STEPS = 8_000_000


def _excursion_lengths(channel, threshold, steps=MC_STEPS):
    """Simulate the fitted AR(1) and record how long each trip beyond
    `threshold` sigma lasts.

    A -3 sigma band is visited rarely, so a fixed budget can leave a
    handful of excursions to divide by -- and "0 of 271" is a far weaker
    claim than it looks. Keep drawing until the denominator is worth
    quoting, or the cap is reached.
    """
    phi, sigma, innov = channel["phi"], channel["sigma"], channel["innovations"]
    m = len(innov)
    rnd = random.Random(SEED).random
    runs, z, cur, longest, drawn = [], 0.0, 0, 0, 0
    while True:
        for _ in range(steps):
            z = phi * z + innov[int(rnd() * m)]
            if z / sigma < threshold:
                cur += 1
                if cur > longest:
                    longest = cur
            elif cur:
                runs.append(cur)
                cur = 0
        drawn += steps
        if len(runs) >= MIN_EXCURSIONS or drawn >= MAX_MC_STEPS:
            return runs, longest


def duration_test(channel, series, freeze_ts):
    """How long has the price been outside the band, and does the fitted
    mean reversion allow a stay that long?"""
    zs = signal_series(channel, series, freeze_ts)

    def current_run(threshold):
        n = 0
        for _, _, z in reversed(zs):
            if z < threshold:
                n += 1
            else:
                break
        return n

    runs = {t: current_run(t) for t in (-1.0, -2.0, -3.0)}
    # Test the deepest band the price is currently sitting below.
    threshold = -3.0 if runs[-3.0] else -2.0 if runs[-2.0] else -1.0
    weeks = runs[threshold]
    if weeks == 0:
        return {
            "runs": {str(k): v for k, v in runs.items()},
            "threshold": None, "weeks": 0,
        }
    sim, longest = _excursion_lengths(channel, threshold)
    at_least = sum(1 for r in sim if r >= weeks)
    sim.sort()
    return {
        "runs": {str(k): v for k, v in runs.items()},
        "threshold": threshold,
        "weeks": weeks,
        "simCount": len(sim),
        "simMedian": sim[len(sim) // 2] if sim else 0,
        "simP90": sim[int(0.9 * (len(sim) - 1))] if sim else 0,
        "simLongest": longest,
        "atLeast": at_least,
        "pValue": at_least / len(sim) if sim else None,
    }


def hold_probability(channel, z_now_sigma, weeks=52, paths=MC_PATHS):
    """Chance the price stays inside +/- HOLD_BAND sigma for `weeks`,
    starting from where it is today."""
    phi, sigma, innov = channel["phi"], channel["sigma"], channel["innovations"]
    m = len(innov)
    rnd = random.Random(SEED + weeks).random
    ok = 0
    for _ in range(paths):
        z = z_now_sigma * sigma
        for _ in range(weeks):
            z = phi * z + innov[int(rnd() * m)]
            if abs(z) / sigma > HOLD_BAND:
                break
        else:
            ok += 1
    return ok / paths


def rollover_record(series, freeze_ts, fit_channel, window_years,
                    min_span=5.0, first_year=2016):
    """Freeze the channel each 31 December and judge it on the following
    year -- strictly out of sample, exactly how the page rolls it."""
    import datetime as dt

    def year_end(y):
        return dt.datetime(y, 12, 31, tzinfo=dt.timezone.utc).timestamp()

    this_year = dt.datetime.fromtimestamp(freeze_ts, dt.timezone.utc).year + 1
    rows = []
    for year in range(first_year, this_year + 1):
        fz = year_end(year - 1)
        try:
            channel = fit_channel(series, fz, window_years)
        except Exception:
            continue
        if (fz - channel["fitStart"]) / SECONDS_PER_YEAR < min_span:
            continue
        nxt = [(t, p) for t, p in series if fz < t <= year_end(year)]
        if len(nxt) < 20:
            continue
        zs = [math.log(p / _trend(channel, t, fz)) / channel["sigma"] for t, p in nxt]
        rows.append({
            "year": year,
            "weeks": len(zs),
            "in1": sum(1 for z in zs if abs(z) <= 1) / len(zs),
            "in2": sum(1 for z in zs if abs(z) <= HOLD_BAND) / len(zs),
            "low": min(zs),
            "high": max(zs),
            "held": all(abs(z) <= HOLD_BAND for z in zs),
            "partial": year >= this_year,
        })
    done = [r for r in rows if not r["partial"]]
    weeks = sum(r["weeks"] for r in done)
    return {
        "rows": rows,
        "years": len(done),
        "held": sum(1 for r in done if r["held"]),
        "cover1": sum(r["in1"] * r["weeks"] for r in done) / weeks if weeks else None,
        "cover2": sum(r["in2"] * r["weeks"] for r in done) / weeks if weeks else None,
    }


def conditional_outcomes(series, freeze_ts, fit_channel, window_years,
                         min_span=5.0, horizon=52):
    """What the following year looked like, given where the price sat --
    measured against the channel frozen before it happened."""
    import datetime as dt

    def year_end(y):
        return dt.datetime(y, 12, 31, tzinfo=dt.timezone.utc).timestamp()

    this_year = dt.datetime.fromtimestamp(freeze_ts, dt.timezone.utc).year + 1
    marks = []
    for year in range(2016, this_year + 1):
        fz = year_end(year - 1)
        try:
            channel = fit_channel(series, fz, window_years)
        except Exception:
            continue
        if (fz - channel["fitStart"]) / SECONDS_PER_YEAR < min_span:
            continue
        for t, p in series:
            if fz < t <= year_end(year):
                marks.append((t, p, math.log(p / _trend(channel, t, fz)) / channel["sigma"]))
    marks.sort()
    if len(marks) < horizon + 20:
        return []

    obs = []
    for i, (t, p, z) in enumerate(marks):
        j = i + horizon
        if j >= len(marks):
            break
        if marks[j][0] - t > 400 * 86400:  # a gap in the series, not a year
            continue
        forward = marks[j][1] / p - 1
        recovered = max(m[2] for m in marks[i + 1:j + 1]) > -0.5
        obs.append((z, forward, recovered))

    out = []
    for lo, hi in BINS:
        sel = [o for o in obs if lo <= o[0] < hi]
        if not sel:
            continue
        # Weeks overlap heavily; episodes are the honest sample size.
        episodes, inside = 0, False
        for z, _, _ in obs:
            now_in = lo <= z < hi
            if now_in and not inside:
                episodes += 1
            inside = now_in
        rets = sorted(o[1] for o in sel)
        out.append({
            "lo": lo, "hi": hi,
            "weeks": len(sel),
            "episodes": episodes,
            "median": rets[len(rets) // 2],
            "up": sum(1 for o in sel if o[1] > 0) / len(sel),
            "recovered": sum(1 for o in sel if o[2]) / len(sel),
        })
    return out


def verdict(duration, record, span_years, snapshot=()):
    """Three states, because the distinction that matters is not
    'good/bad' but 'the model is standing' / 'it is not'."""
    p = duration.get("pValue")
    weeks = duration.get("weeks", 0)
    if p is not None and weeks and p < 0.01:
        thr = duration["threshold"]
        return {
            "level": "falsified",
            "headline": "The channel no longer describes this asset",
            "reasons": [
                f"the price has held below {thr:g}σ for {weeks} weeks, and the "
                f"fitted mean reversion produced a stay that long "
                f"{duration['atLeast']} times in {duration['simCount']:,} "
                f"simulated excursions",
                "whatever happens next, this channel is not the reason to expect it",
            ] + list(snapshot),
        }
    reasons = list(snapshot)
    if record["years"] >= 4 and record["held"] / record["years"] < WEAK_RECORD:
        reasons.append(
            f"frozen each January, the channel held for the whole of only "
            f"{record['held']} of the last {record['years']} years"
        )
    if record["cover2"] is not None and record["cover2"] < 0.85:
        reasons.append(
            f"the price sat inside ±2σ {record['cover2'] * 100:.0f}% of "
            f"the time, against the 95% the band claims"
        )
    if span_years < 5.0:
        reasons.append(
            f"only {span_years:.1f} years of history stand behind the fit"
        )
    if reasons:
        return {
            "level": "caution",
            "headline": "The channel is standing, but it has a weak record here",
            "reasons": reasons,
        }
    return {
        "level": "intact",
        "headline": "The channel is standing",
        "reasons": [
            f"the price is inside the band, and frozen each January the channel "
            f"held all year in {record['held']} of the last {record['years']}"
        ],
    }
