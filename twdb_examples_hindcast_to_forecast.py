# %% Configuration — run this cell first

import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


# ENTER YOUR API CODE HERE
api_key = "ENTER CODE HERE"

# TWDB — all endpoints (discovery, historical, and forecast timeseries)
TWDB_BASE_URL = "https://twdb.dri.edu"
TWDB_HEADERS = {"api-key": api_key}

# seconds to wait before every request (avoids 429s proactively)
REQUEST_DELAY = 1.0

def _get(url: str, headers: dict, params: dict = None, retries: int = 5) -> requests.Response:
    """GET with a proactive delay and exponential backoff on 429 Too Many Requests."""
    time.sleep(REQUEST_DELAY)
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = 2 ** attempt
        print(f"Rate limited — retrying in {
              wait}s (attempt {attempt + 1}/{retries})")
        time.sleep(wait)
    r.raise_for_status()
    return r


def _clamp_dates(start_date: str, end_date: str, available: list[str]) -> tuple[str, str]:
    """
    Clamp start/end to the available date range, printing a warning if adjusted.
    Dates must be strings in 'YYYY-MM-DD' format.
    """
    first, last = available[0], available[-1]
    clamped_start = max(start_date, first)
    clamped_end = min(end_date,   last)
    if clamped_start != start_date:
        print(f"Warning: start_date {
              start_date!r} is before available range. Clamped to {clamped_start!r}.")
    if clamped_end != end_date:
        print(f"Warning: end_date {
              end_date!r} is after available range. Clamped to {clamped_end!r}.")
    if clamped_start > clamped_end:
        raise ValueError(
            f"Requested date range [{start_date}, {
                end_date}] does not overlap "
            f"with available range [{first}, {last}]."
        )
    return clamped_start, clamped_end


def get_historical(
    reservoir_names: list[str] | str,
    datasets_variables: dict[str, list[str] | str],
    start_date: str,
    end_date: str,
    units: str = "metric",
) -> pd.DataFrame:
    """
    Fetch historical daily timeseries for one or more reservoirs.

    Dates are automatically clamped to the available historical range with a warning.
    One API call is made per (dataset, variable) pair; results are merged on
    (start_date, RES_NAME).

    Parameters
    ----------
    reservoir_names : str or list[str]
        e.g. "CANYON LAKE" or ["CADDO LAKE", "CANYON LAKE"]
    datasets_variables : dict
        {dataset: variable_or_list_of_variables}, e.g.
            {"nete-volume-calcs": ["NetE", "E_volume"], "rtma": "pr"}
    start_date, end_date : str  -- "YYYY-MM-DD"
    units : str                 -- "metric" (default) or "english"

    Returns
    -------
    pd.DataFrame with columns: start_date, RES_NAME, <variable columns ...>
    """
    if isinstance(reservoir_names, str):
        reservoir_names = [reservoir_names]

    available = _get(f"{TWDB_BASE_URL}/info/list_dates",
                     TWDB_HEADERS).json()["dates"]
    start_date, end_date = _clamp_dates(start_date, end_date, available)

    merge_keys = ["date", "RES_NAME"]
    frames = []
    for dataset, variables in datasets_variables.items():
        if isinstance(variables, str):
            variables = [variables]
        for variable in variables:
            r = _get(
                f"{TWDB_BASE_URL}/timeseries/daily/reservoirs/daterange",
                TWDB_HEADERS,
                params={
                    "RES_NAMES":     ",".join(reservoir_names),
                    "datasets":      dataset,
                    "variables":     variable,
                    "start_date":    start_date,
                    "end_date":      end_date,
                    "units":         units,
                    "output_format": "json",
                },
            )
            df = pd.DataFrame(r.json())

            df["date"] = pd.to_datetime(df["date"])
            drop = [
                c for c in df.columns if c not in merge_keys and c in frames[0].columns] if frames else []
            frames.append(df.drop(columns=drop))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=merge_keys, how="outer")

    return merged.sort_values(merge_keys).reset_index(drop=True)


def get_forecast(
    reservoir_names: list[str] | str,
    datasets_variables: dict[str, list[str] | str],
    start_date: str,
    end_date: str,
    units: str = "metric",
) -> pd.DataFrame:
    """
    Fetch forecast daily timeseries for one or more reservoirs.

    Dates are automatically clamped to the available forecast range with a warning.

    Parameters
    ----------
    reservoir_names : str or list[str]
        e.g. "CANYON LAKE" or ["CADDO LAKE", "CANYON LAKE"]
    datasets_variables : dict
        {dataset: variable_or_list_of_variables}, e.g.
            {"cfs_gridmet": ["pr", "tmmx_c"], "nete-volume-calcs": "NetE"}
    start_date, end_date : str  -- "YYYY-MM-DD"
    units : str                 -- "metric" (default) or "english"

    Returns
    -------
    pd.DataFrame with columns: date, RES_NAME, <variable + ensemble columns ...>
    """
    if isinstance(reservoir_names, str):
        reservoir_names = [reservoir_names]

    available = _get(f"{TWDB_BASE_URL}/info/list_forecast_dates",
                     TWDB_HEADERS).json()["dates"]
    start_date, end_date = _clamp_dates(start_date, end_date, available)

    all_datasets, all_variables = [], []
    for dataset, variables in datasets_variables.items():
        if isinstance(variables, str):
            variables = [variables]
        all_datasets.append(dataset)
        all_variables.extend(variables)

    r = _get(
        f"{TWDB_BASE_URL}/timeseries/forecast/daily/reservoirs/daterange",
        TWDB_HEADERS,
        params={
            "RES_NAMES":      ",".join(reservoir_names),
            "datasets":       ",".join(all_datasets),
            "variables":      ",".join(all_variables),
            "start_date":     start_date,
            "end_date":       end_date,
            "units":          units,
            "output_format":  "json",
        },
    )
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "RES_NAME"]).reset_index(drop=True)


# %% Plot historical + forecast for a single reservoir
# --- inputs ---
today = datetime.date.today()
reservoir_names = ["CANYON LAKE"]
hist_datasets_variables = {"lem": ["E_hs"]}
fcast_datasets_variables = {"lem": ["E_hs"]}

# plot available historical estimates
hist_start =(today + datetime.timedelta(days=-14)).strftime("%Y-%m-%d")
hist_end = (today + datetime.timedelta(days=-4)).strftime("%Y-%m-%d")

#get forecast data plus plast forecast data to include 3 days between hist and current fcast
fcast_start = (today + datetime.timedelta(days=-3)).strftime("%Y-%m-%d")
fcast_end = (today + datetime.timedelta(days=27)).strftime("%Y-%m-%d")

# --- usage ---
hist = get_historical(
    reservoir_names, hist_datasets_variables,  hist_start,  hist_end)
fcast = get_forecast(
    reservoir_names, fcast_datasets_variables, fcast_start, fcast_end)

fig, ax = plt.subplots(figsize=(12, 6))

plt.plot(hist["date"],  hist["E_hs (mm)"], linewidth=2.5,
         c="black",   label="E_hs Historical")
plt.plot(fcast["date"],       fcast["E_hs ensemble_mean (mm)"], linewidth=2.5,
         c="Violet", label="E_hs Forecast mean")

#Connect historical and forecast time series
plt.plot([hist["date"].iloc[-1], fcast["date"].iloc[0]], [hist["E_hs (mm)"].iloc[-1], fcast["E_hs ensemble_mean (mm)"].iloc[0]], color='black')

# Shaded Area: 10th - 90th (Lighter)
plt.fill_between(fcast.date, fcast["E_hs ensemble_p10 (mm)"], fcast["E_hs ensemble_p90 (mm)"], color='blue', alpha=0.2, label='10th-90th Percentile')


# Shaded Area: 25th - 75th (Darker)
plt.fill_between(fcast.date, fcast["E_hs ensemble_p25 (mm)"], fcast["E_hs ensemble_p75 (mm)"], color='blue', alpha=0.4, label='25th-75th Percentile')

# Use vertical lines to mark prior forecast estimates 
plt.axvline(x=today, color='k', linestyle='--')
plt.axvline(x=(today + datetime.timedelta(days=-3)), color='k', linestyle='--', label='Prior Forecast Estimates')


ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
fig.autofmt_xdate(rotation=45) 

plt.title("Historical & Forecast Evaporation Comparison CANYON LAKE")
plt.ylabel("mm/day")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
