"""Build the GMI Compounding Machine dashboard (trading.html).

Fetches weekly price history for each asset, fits a frozen log-linear trend
channel, calibrates a mean-reverting model of the residuals, and bakes the
whole payload into a self-contained HTML page. All of the strategy maths and
charting happens client-side so the sliders recompute instantly.

Standard library only -- no third-party dependencies.
"""

import datetime as dt
import json
import math
import re
import time
import urllib.request

# --- CONFIG -------------------------------------------------------------
FIT_WINDOW_YEARS = 8  # regression lookback, ending at the freeze date
HORIZON_END = dt.date(2031, 1, 1)  # how far the channel is drawn forward
TEMPLATE = "gmi_template.html"
OUTPUT = "trading.html"
SECONDS_PER_YEAR = 365.25 * 86400
USER_AGENT = "Mozilla/5.0 (compatible; gmi-dashboard/1.0)"


def freeze_date(today):
    """The regression is frozen at the last day of the previous year and
    rolls forward each January."""
    return dt.date(today.year - 1, 12, 31)


def _get_json(url, attempts=3):
    """These endpoints are rate-limited and occasionally flaky from CI
    runners, so back off and retry before giving up on an asset."""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as res:
                return json.load(res)
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = 2 ** attempt
            print(f"    {type(exc).__name__}: {exc} -- retrying in {wait}s")
            time.sleep(wait)


def fetch_kraken_weekly(pair):
    """Weekly closes from Kraken (~13 years of history, no API key)."""
    data = _get_json(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=10080")
    if data.get("error"):
        raise RuntimeError(f"Kraken error for {pair}: {data['error']}")
    rows = next(iter(data["result"].values()))
    return [(int(r[0]), float(r[4])) for r in rows if float(r[4]) > 0]


def fetch_yahoo_weekly(symbol, years=20):
    """Weekly adjusted closes from Yahoo Finance."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={years}y&interval=1wk"
    )
    result = _get_json(url)["chart"]["result"][0]
    closes = result["indicators"]["adjclose"][0]["adjclose"]
    return [
        (int(t), float(p))
        for t, p in zip(result["timestamp"], closes)
        if p and p > 0
    ]


def fit_channel(series, freeze_ts, window_years):
    """OLS of ln(price) on time over the fit window, plus an AR(1) fit of the
    residuals so the forward simulation reverts at the observed speed."""
    lo = freeze_ts - window_years * SECONDS_PER_YEAR
    points = [(t, p) for t, p in series if lo <= t <= freeze_ts]
    if len(points) < 52:
        raise RuntimeError(f"only {len(points)} points inside the fit window")

    xs = [(t - freeze_ts) / SECONDS_PER_YEAR for t, _ in points]
    ys = [math.log(p) for _, p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x

    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sigma = math.sqrt(sum(r * r for r in resid) / (n - 2))

    # AR(1) on the residuals -> weekly persistence of the deviation.
    phi = sum(resid[i] * resid[i + 1] for i in range(len(resid) - 1)) / sum(
        r * r for r in resid
    )
    phi = min(max(phi, 0.0), 0.999)
    # Innovations drive the forward paths; bootstrapping them (rather than
    # drawing Gaussians) keeps the fat tails and skew of the real series. They
    # are centred first: the AR(1) turns a residual mean of e into a stationary
    # mean of e/(1-phi), so a rounding-level bias here becomes a large drift
    # away from the very trend the process is supposed to revert to.
    innovations = [resid[i + 1] - phi * resid[i] for i in range(len(resid) - 1)]
    drift = sum(innovations) / len(innovations)
    innovations = [v - drift for v in innovations]

    half_life = math.log(2) / (-math.log(phi) * 52.0) if 0 < phi < 1 else None

    # An AR(1) driven by these shocks settles at a spread of
    # sd(e)/sqrt(1-phi^2), which the small-sample bias in phi puts a little
    # under the sigma the channel is drawn with. Rescale the shocks -- keeping
    # their shape, so the fat tails survive -- so the forward paths fill
    # exactly the band the page claims for them.
    var_e = sum(v * v for v in innovations) / len(innovations)
    scale = sigma * math.sqrt(1 - phi * phi) / math.sqrt(var_e)
    innovations = [v * scale for v in innovations]

    return {
        "anchor": math.exp(intercept),  # fitted price at the freeze date
        "cagr": math.exp(slope) - 1.0,
        "sigma": sigma,  # standard deviation in log space
        "phi": phi,
        "halfLifeYears": half_life,
        "innovations": [round(v, 6) for v in innovations],
        "fitPoints": n,
        "fitStart": points[0][0],
    }


def build_asset(key, name, blurb, series, freeze_ts, price_decimals):
    channel = fit_channel(series, freeze_ts, FIT_WINDOW_YEARS)
    history = [[t, round(p, price_decimals)] for t, p in series]
    last_t, last_p = series[-1]
    trend_now = channel["anchor"] * (1 + channel["cagr"]) ** (
        (last_t - freeze_ts) / SECONDS_PER_YEAR
    )
    # Recomputed from the rounded values actually shipped to the browser.
    shocks = channel["innovations"]
    var_e = sum(v * v for v in shocks) / len(shocks)
    spread = math.sqrt(var_e / (1 - channel["phi"] ** 2))
    drift_ratio = spread / channel["sigma"]
    print(
        f"  {key}: {len(history)} weeks, CAGR {channel['cagr'] * 100:.1f}%, "
        f"1sigma +{(math.exp(channel['sigma']) - 1) * 100:.0f}%, "
        f"last {last_p:,.2f} vs trend {trend_now:,.2f} "
        f"({math.log(last_p / trend_now) / channel['sigma']:+.2f} sigma)"
    )
    print(
        f"     AR(1) phi={channel['phi']:.4f}, half-life "
        f"{channel['halfLifeYears']:.2f}y, simulated spread "
        f"{drift_ratio:.2f}x the fitted sigma"
    )
    if not 0.8 <= drift_ratio <= 1.25:
        raise RuntimeError(
            f"{key}: forward model spread is {drift_ratio:.2f}x the fitted sigma"
        )
    return {"key": key, "name": name, "blurb": blurb, "history": history, **channel}


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    freeze = freeze_date(today)
    freeze_ts = dt.datetime(
        freeze.year, freeze.month, freeze.day, tzinfo=dt.timezone.utc
    ).timestamp()
    print(f"Building {OUTPUT} -- channel frozen {freeze}")

    # One unreachable feed should cost us that asset, not the whole page --
    # this runs unattended, and a stale trading.html beats a failed build.
    sources = [
        ("BTC", "Bitcoin", "Kraken XBT/USD weekly close",
         lambda: fetch_kraken_weekly("XBTUSD")),
        ("QQQ", "Nasdaq 100 ETF", "Yahoo Finance QQQ weekly adjusted close",
         lambda: fetch_yahoo_weekly("QQQ")),
    ]
    assets, failed = [], []
    for key, name, blurb, fetch in sources:
        try:
            assets.append(build_asset(key, name, blurb, fetch(), freeze_ts, 2))
        except Exception as exc:
            failed.append(f"{key} ({type(exc).__name__}: {exc})")
            print(f"  {key}: SKIPPED -- {exc}")
    if not assets:
        raise RuntimeError("no asset could be built: " + "; ".join(failed))
    if failed:
        print(f"WARNING: built without {', '.join(failed)}")

    payload = {
        "builtAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "freeze": freeze.isoformat(),
        "nextRoll": f"Jan {freeze.year + 2}",
        "channelVersion": f"v{freeze.year + 1}",
        "fitWindowYears": FIT_WINDOW_YEARS,
        "horizonEnd": HORIZON_END.isoformat(),
        "assets": assets,
    }

    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    if "__GMI_DATA__" not in template:
        raise RuntimeError(f"{TEMPLATE} is missing the __GMI_DATA__ placeholder")

    # json.dumps output is inserted inside a <script> block: neutralise any
    # sequence that could close it early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__GMI_DATA__", blob)
    html = re.sub(r"__BUILT_AT__", payload["builtAt"], html)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUTPUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
