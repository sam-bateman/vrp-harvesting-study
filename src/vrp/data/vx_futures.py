"""VIX futures (VX) continuous front/second-month series.

Sources CBOE CFE per-contract daily settlements from the CBOE CDN, then
stitches them into a continuous front/second-month series. The roll
schedule follows CBOE's SOQ rule: expiration is the Wednesday 30 days
prior to the 3rd Friday of the month FOLLOWING the contract's designated
expiry month. If that Wednesday, or the 3rd Friday it is anchored to, is
an exchange holiday, settlement moves to the business day immediately
preceding the Wednesday (CFE rule; e.g. Good Friday Aprils 2014/2019/2022
and Juneteenth 2024).

Besides the market-designated front/second settles (used for signals),
the continuous frame carries splice-free *held-contract* daily returns
(``held_front_ret``/``held_second_ret``): each day's return is computed
within a single contract, and the held pair rolls ``roll_days_before_expiry``
trading days before front expiry. Diffing the spliced ``front_settle``
column across a roll books the calendar spread as phantom PnL — use the
held returns for any PnL computation.

Data source (confirmed working as of 2025):
    https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/
    VX/VX_{YYYY-MM-DD}.csv
    where {YYYY-MM-DD} is the expiry date of the specific contract.
    Coverage: 2013-present (pre-2013 contracts return HTTP 403).

Fallback — local override:
    Set the VX_CSV_OVERRIDE env var to a directory containing one CSV per
    contract, named  VX_{YYYY-MM-DD}.csv (same naming as the CBOE CDN)
    with columns:
        Trade Date, Futures, Open, High, Low, Close, Settle, Change,
        Total Volume, EFP, Open Interest
"""
from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

from . import cache

# v2: holiday-adjusted expiries, Settle=0 -> Close fallback, held-contract
# return columns. Bumped so stale v1 parquet is never reused.
_KEY_PREFIX = "vx_futures_v2"

_CONTRACT_CODE_MAP = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

_MONTH_NAME_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

_CBOE_BASE = (
    "https://cdn.cboe.com/data/us/futures/market_statistics/"
    "historical_data/VX/VX_{date}.csv"
)


# ---------------------------------------------------------------------------
# Roll calendar — pure logic, no network
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.weekday()) % 7  # Mon=0 … Fri=4
    first_friday = first + pd.Timedelta(days=offset)
    return first_friday + pd.Timedelta(days=14)


class _ExchangeHolidays(AbstractHolidayCalendar):
    """US equity/options exchange holidays relevant to VX settlement dates.

    Not a full NYSE calendar (no Sat/Sun observance edge cases for New
    Year's on a Saturday, etc.) — only what is needed to place the
    Wednesday settlement / 3rd-Friday anchor correctly.
    """
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def _holiday_set(around: pd.Timestamp) -> set:
    cal = _ExchangeHolidays()
    return set(cal.holidays(around - pd.Timedelta(days=400),
                            around + pd.Timedelta(days=400)))


def _prev_business_day(d: pd.Timestamp, holidays: set) -> pd.Timestamp:
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5 or d in holidays:
        d = d - pd.Timedelta(days=1)
    return d


def vx_expiration(contract_year: int, contract_month: int) -> pd.Timestamp:
    """Final settlement date for a VX contract designated (year, month).

    CFE rule: the Wednesday 30 days prior to the 3rd Friday of the MONTH
    AFTER the contract's designated month. If that Wednesday or that 3rd
    Friday is an exchange holiday, settlement is the business day
    immediately preceding the Wednesday. (Good Friday moves the April
    3rd Friday in 2014/2019/2022; Juneteenth 2024 lands on the Wednesday
    itself.)
    """
    next_month = (
        pd.Timestamp(year=contract_year, month=contract_month, day=1)
        + pd.offsets.MonthBegin(1)
    )
    tf = _third_friday(next_month.year, next_month.month)
    wed = (tf - pd.Timedelta(days=30)).normalize()
    holidays = _holiday_set(wed)
    if tf.normalize() in holidays or wed in holidays:
        return _prev_business_day(wed, holidays)
    return wed


def build_roll_calendar(start: str, end: str) -> pd.DataFrame:
    """Per business day in [start, end]: front and second contract expiries.

    Returns a DataFrame indexed by business date with columns
    ``front_expiry`` and ``second_expiry``.
    """
    idx = pd.bdate_range(start, end)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    # Generate enough expiration dates to cover the range.
    expirations = []
    y, m = s.year, s.month
    while True:
        exp = vx_expiration(y, m)
        expirations.append(exp)
        if exp > e + pd.Timedelta(days=90):
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    expirations = sorted(set(expirations))

    front_exp = []
    second_exp = []
    for d in idx:
        future = [x for x in expirations if x > d][:2]
        if len(future) < 2:
            front_exp.append(pd.NaT)
            second_exp.append(pd.NaT)
        else:
            front_exp.append(future[0])
            second_exp.append(future[1])

    return pd.DataFrame(
        {"front_expiry": front_exp, "second_expiry": second_exp},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Data fetch — CBOE per-contract CSVs keyed by expiry date
# ---------------------------------------------------------------------------

def _fetch_contract(expiry: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Download the per-contract CSV for the given expiry date.

    Returns None if the contract is unavailable (HTTP 403 / network error).
    """
    date_str = expiry.strftime("%Y-%m-%d")

    override = os.environ.get("VX_CSV_OVERRIDE")
    if override:
        path = Path(override) / f"VX_{date_str}.csv"
        if path.exists():
            return pd.read_csv(path)

    url = _CBOE_BASE.format(date=date_str)
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.text) > 50:
            return pd.read_csv(StringIO(r.text))
    except Exception:
        pass
    return None


def _parse_futures_column(futures_str: str) -> tuple[int, int]:
    """Parse the Futures column from a CBOE per-contract CSV.

    CBOE format: 'F (Jan 2020)'  →  (2020, 1)
    Also handles legacy formats:
        'VX/H20', 'VX H20', 'VXH20', 'H20'  →  (2020, 3)
    """
    s = futures_str.strip()

    # Primary CBOE CDN format: 'F (Jan 2020)' or 'M (Jun 2025)'
    m = re.match(r"[A-Z]\s+\((\w{3})\s+(\d{4})\)", s)
    if m:
        month_name, year = m.group(1), int(m.group(2))
        return year, _MONTH_NAME_MAP[month_name]

    # Legacy format: 'VX/H20', 'VX H20', 'VXH20', 'H20'
    inv = {v: k for k, v in _CONTRACT_CODE_MAP.items()}
    code = (
        s.upper()
        .replace("VX/", "")
        .replace("VX ", "")
        .replace("VX", "")
        .strip()
    )
    letter = code[0]
    year_suffix = code[1:]
    if len(year_suffix) == 2:
        year = 2000 + int(year_suffix)
    elif len(year_suffix) == 4:
        year = int(year_suffix)
    else:
        raise ValueError(f"Cannot parse contract year from '{futures_str}'")
    return year, inv[letter]


# ---------------------------------------------------------------------------
# Continuous series builder
# ---------------------------------------------------------------------------

def build_continuous_from_panel(panel: pd.DataFrame,
                                roll_days_before_expiry: int,
                                holidays: set) -> pd.DataFrame:
    """Build the continuous frame from a (trade_date x expiry) settle panel.

    Pure function of its inputs — see ``load_vx_continuous`` for the
    column contract. Split out so the roll/splice logic is unit-testable
    with synthetic panels.
    """
    ret_panel = panel.pct_change()
    holidays_np = pd.DatetimeIndex(sorted(holidays)).values.astype(
        "datetime64[D]"
    )
    expiries = list(panel.columns)
    dates = panel.index

    # Trading days strictly between date d and expiry e.
    tdte = np.busday_count(
        dates.values.astype("datetime64[D]")[:, None] + np.timedelta64(1, "D"),
        pd.DatetimeIndex(expiries).values.astype("datetime64[D]")[None, :],
        holidays=holidays_np,
    )

    def _held_pair(i: int) -> Optional[tuple]:
        cands = [k for k, e in enumerate(expiries)
                 if e > dates[i] and tdte[i, k] > roll_days_before_expiry
                 and not pd.isna(panel.iloc[i, k])]
        if len(cands) < 2:
            return None
        return cands[0], cands[1]

    def _market_pair(i: int) -> Optional[tuple]:
        cands = [k for k, e in enumerate(expiries)
                 if e > dates[i] and not pd.isna(panel.iloc[i, k])]
        if len(cands) < 2:
            return None
        return cands[0], cands[1]

    out_rows = []
    prev_held = None
    for i in range(len(dates)):
        market = _market_pair(i)
        held = _held_pair(i)
        if market is None or held is None:
            prev_held = held
            continue
        mf, ms = market
        # Return on day i belongs to the pair held at the previous close.
        hf_ret = hs_ret = float("nan")
        if prev_held is not None:
            pf, ps = prev_held
            hf_ret = float(ret_panel.iloc[i, pf])
            hs_ret = float(ret_panel.iloc[i, ps])
        out_rows.append({
            "date": dates[i],
            "front_settle": float(panel.iloc[i, mf]),
            "second_settle": float(panel.iloc[i, ms]),
            "front_expiry": expiries[mf],
            "second_expiry": expiries[ms],
            "days_to_front_expiry": (expiries[mf] - dates[i]).days,
            "held_front_ret": hf_ret,
            "held_second_ret": hs_ret,
            "held_front_expiry": expiries[held[0]],
            "is_roll_day": prev_held is not None and held != prev_held,
        })
        prev_held = held

    return pd.DataFrame(out_rows).set_index("date").sort_index()


def load_vx_continuous(
    start: str = "2013-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    roll_days_before_expiry: int = 5,
) -> pd.DataFrame:
    """Continuous front/second VX settlement series.

    Returns a DataFrame indexed by trade date with columns:
        front_settle, second_settle, front_expiry, second_expiry,
        days_to_front_expiry
            — market designation: "front" is the nearest listed expiry
              strictly after the trade date. Use for signals (term-
              structure inversion, sanity checks). ``days_to_front_expiry``
              is in calendar days. Diffing these spliced series books the
              calendar spread as phantom PnL at each roll — never use
              them for PnL.
        held_front_ret, held_second_ret, held_front_expiry, is_roll_day
            — held designation: the pair a trader holds after rolling
              ``roll_days_before_expiry`` TRADING days before front
              expiry. Returns are computed within a single contract
              (splice-free) and attributed to the pair held at the
              previous close. ``is_roll_day`` marks the day the pair
              changes (transaction costs apply there).

    Data coverage: 2013-present (CBOE CDN availability). For pre-2013
    data, set VX_CSV_OVERRIDE to a directory of per-contract CSVs.
    Early-2013 files carry Settle=0 with the real mark in Close; Close
    is used as the fallback mark on such rows.

    Parameters
    ----------
    start:
        First date to include (inclusive).
    end:
        Last date to include (inclusive). Defaults to today.
    use_cache:
        Read/write a local parquet cache keyed by (start, end, roll).
    roll_days_before_expiry:
        Trading days before front expiry at which the held pair rolls.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    key = f"{_KEY_PREFIX}_continuous__{start}__{end}__roll{roll_days_before_expiry}"

    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached

    # Build the list of contracts we need to cover [start, end].
    # We need contracts whose data window overlaps [start, end].
    # Add a two-month buffer on each side to ensure front/second coverage.
    s = pd.Timestamp(start) - pd.DateOffset(months=2)
    e = pd.Timestamp(end) + pd.DateOffset(months=2)

    y, m = s.year, s.month
    contract_expiries: list[pd.Timestamp] = []
    while True:
        exp = vx_expiration(y, m)
        if exp > e:
            break
        contract_expiries.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1

    frames = []
    missing = []
    for exp in contract_expiries:
        df = _fetch_contract(exp)
        if df is not None:
            frames.append(df)
        else:
            missing.append(exp.date())

    if missing:
        in_window = [m for m in missing
                     if pd.Timestamp(start) <= pd.Timestamp(m)]
        print(
            f"VX: {len(missing)} contract(s) unavailable "
            f"(first: {missing[0]}, last: {missing[-1]})"
        )
        if in_window:
            raise RuntimeError(
                f"VX contracts inside the requested window are missing: "
                f"{in_window}. The stitched series would silently promote "
                f"the next contract to front. Fix the fetch (check "
                f"vx_expiration holiday handling) or supply the files via "
                f"VX_CSV_OVERRIDE."
            )

    if not frames:
        raise RuntimeError(
            "No VX contract data fetched. Set VX_CSV_OVERRIDE to a directory "
            "with VX_{YYYY-MM-DD}.csv files and retry."
        )

    raw = pd.concat(frames, ignore_index=True)
    # Normalise column names
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    date_col = "trade_date" if "trade_date" in raw.columns else "date"
    raw["trade_date"] = pd.to_datetime(raw[date_col])

    # Settle column: 'settle' is standard in CBOE per-contract files
    settle_col = next(
        (c for c in ("settle", "settlement_price", "settle_price", "close")
         if c in raw.columns),
        None,
    )
    if settle_col is None:
        raise RuntimeError(
            f"No settle column found. Available columns: {raw.columns.tolist()}"
        )

    close_col = "close" if "close" in raw.columns else None
    keep = ["trade_date", "futures", settle_col] + (
        [close_col] if close_col and close_col != settle_col else []
    )
    raw = raw[keep].dropna(subset=["trade_date", "futures"])
    raw = raw.rename(columns={settle_col: "settle"})
    raw["settle"] = pd.to_numeric(raw["settle"], errors="coerce")
    # Early-2013 CDN files publish Settle=0 with the real mark in Close.
    if close_col and close_col in raw.columns:
        close_vals = pd.to_numeric(raw[close_col], errors="coerce")
        bad = ~(raw["settle"] > 0)
        raw.loc[bad, "settle"] = close_vals[bad]
    raw = raw.dropna(subset=["settle"])
    raw = raw[raw["settle"] > 0]

    raw["expiry"] = raw["futures"].map(
        lambda c: vx_expiration(*_parse_futures_column(c))
    )

    # date x expiry settle panel; per-contract returns are splice-free.
    panel = raw.pivot_table(index="trade_date", columns="expiry",
                            values="settle", aggfunc="last").sort_index()

    holidays = (_holiday_set(pd.Timestamp(start)) |
                _holiday_set(pd.Timestamp(end)))
    out = build_continuous_from_panel(panel, roll_days_before_expiry,
                                      holidays)
    out = out.loc[start:end]

    n_nan = int(out["held_front_ret"].isna().sum())
    if n_nan > 1:  # first row has no prior close by construction
        print(f"VX: {n_nan} day(s) with missing held-contract returns")

    cache.save(key, out)
    return out
