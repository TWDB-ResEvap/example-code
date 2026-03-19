# %% Configuration — run this cell first

import time
import requests
import pandas as pd
import matplotlib.pyplot as plt

# TWDB — all endpoints (discovery, historical, and forecast timeseries)
TWDB_BASE_URL = "https://twdb.dri.edu"
TWDB_HEADERS = {"api-key": "ADD KEY HERE"}

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


def _post(url: str, headers: dict, params: dict = None, retries: int = 5) -> requests.Response:
    """POST with a proactive delay and exponential backoff on 429 Too Many Requests."""
    time.sleep(REQUEST_DELAY)
    for attempt in range(retries):
        r = requests.post(url, headers=headers, params=params)
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


# %% Example 1 — Discover available dates, datasets and variables


def get_dates() -> dict[str, list[str]]:
    """Return available date ranges (historical and forecast) from TWDB."""
    hist = _get(f"{TWDB_BASE_URL}/info/list_dates",          TWDB_HEADERS)
    fcast = _get(f"{TWDB_BASE_URL}/info/list_forecast_dates", TWDB_HEADERS)
    return {"historical": hist.json()["dates"], "forecast": fcast.json()["dates"]}


def get_datasets() -> list[str]:
    """Return the list of dataset names available in the TWDB API."""
    return _get(f"{TWDB_BASE_URL}/info/list_datasets", TWDB_HEADERS).json()["datasets"]


def get_variables(dataset: str) -> list[str]:
    """Return all variable names available in *dataset*."""
    return [v["api_name"] for v in _get(
        f"{TWDB_BASE_URL}/info/list_variables", TWDB_HEADERS, params={"dataset": dataset}
    ).json()["variables"]]


def get_all_variables() -> dict[str, list[str]]:
    """Return {dataset: [variables]} for every dataset in the TWDB API."""
    return {ds: get_variables(ds) for ds in get_datasets()}


# --- usage ---

dates = get_dates()
print("Historical range:", dates["historical"]
      [0], "->", dates["historical"][-1])
print("Forecast range:  ", dates["forecast"][0],   "->", dates["forecast"][-1])

print("\nTWDB datasets/variables:")
for ds, variables in get_all_variables().items():
    print(f"  {ds}: {variables}")


# %% Example 2 — Reservoir names and metadata


def get_reservoir_names() -> list[str]:
    """Return all available reservoir names."""
    return _post(f"{TWDB_BASE_URL}/info/list_RES_NAMES", TWDB_HEADERS).json()["RES_NAME"]


def get_reservoir_metadata(reservoir_names: list[str] | str) -> pd.DataFrame:
    """Return metadata (lat, lon, elevation, ...) for one or more reservoirs."""
    if isinstance(reservoir_names, str):
        reservoir_names = [reservoir_names]
    return pd.DataFrame(_get(
        f"{TWDB_BASE_URL}/metadata/reservoirs",
        TWDB_HEADERS,
        params={"RES_NAME": ",".join(
            reservoir_names), "output_format": "json"},
    ).json())


# --- usage ---

reservoir_names = ["LAKE ALICE", "LAKE ESTES"]

reservoirs = get_reservoir_names()
print(f"Total reservoirs: {len(reservoirs)}")

meta = get_reservoir_metadata(reservoir_names)
print(meta[["RES_NAME", "LAT", "LON", "ELEV_M"]])


# %% Example 3 — Forecast: multiple reservoirs, multiple datasets


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


# --- inputs ---

reservoir_names = ["CADDO LAKE", "CANYON LAKE"]
datasets_variables = {"nete-volume-calcs": ["NetE", "E_volume"]}
start_date = "2026-03-17"
end_date = "2026-03-26"

# --- usage ---

fcast = get_forecast(reservoir_names, datasets_variables, start_date, end_date)

plt.figure(figsize=(12, 4))
for res in fcast["RES_NAME"].unique():
    sub = fcast[fcast["RES_NAME"] == res]
    plt.plot(sub["date"], sub["NetE ensemble_mean (mm)"], label=res)
plt.title("Forecast Net Evaporation — Multiple Reservoirs")
plt.xlabel("Date")
plt.ylabel("mm/day")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Example 4 — Forecast: single reservoir with ensemble uncertainty bands


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


# --- inputs ---

reservoir_names = ["CANYON LAKE"]
datasets_variables = {"nete-volume-calcs": ["NetE"]}
start_date = "2026-03-17"
end_date = "2026-04-12"

# --- usage ---

fcast = get_forecast(reservoir_names, datasets_variables, start_date, end_date)

plt.figure(figsize=(10, 5))
plt.plot(fcast["date"], fcast["NetE ensemble_mean (mm)"],
         c="red", label="Forecast mean")
plt.plot(fcast["date"], fcast["NetE ensemble_p10 (mm)"],  c="red", ls="--")
plt.plot(fcast["date"], fcast["NetE ensemble_p90 (mm)"],
         c="red", ls="--", label="p10-p90 range")
plt.title("Forecast Evaporation — CANYON LAKE")
plt.ylabel("mm/day")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Example 5 — Historical + forecast for a single reservoir


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


# --- inputs ---

reservoir_names = ["CANYON LAKE"]
hist_datasets_variables = {"nete-volume-calcs": ["NetE"]}
fcast_datasets_variables = {"nete-volume-calcs": ["NetE"]}
hist_start = "2026-03-01"
hist_end = "2026-03-16"
fcast_start = "2026-03-17"
fcast_end = "2026-04-12"

# --- usage ---

hist = get_historical(
    reservoir_names, hist_datasets_variables,  hist_start,  hist_end)
fcast = get_forecast(
    reservoir_names, fcast_datasets_variables, fcast_start, fcast_end)

plt.figure(figsize=(12, 5))
plt.plot(hist["date"],  hist["NetE (mm)"],
         c="k",   label="Historical")
plt.plot(fcast["date"],       fcast["NetE ensemble_mean (mm)"],
         c="red", label="Forecast mean")
plt.plot(fcast["date"],       fcast["NetE ensemble_p10 (mm)"],
         c="red", ls="--")
plt.plot(fcast["date"],       fcast["NetE ensemble_p90 (mm)"],
         c="red", ls="--", label="p10/p90")
plt.axvline(fcast["date"].iloc[0], color="gray",
            ls=":", lw=1, label="Forecast start")
plt.title("Historical & Forecast Net Evaporation — CANYON LAKE")
plt.ylabel("mm/day")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Example 6 — comparison with and without heat storage in historical + forecast for a single reservoir


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


# --- inputs ---

reservoir_names = ["CANYON LAKE"]
hist_datasets_variables = {"lem": ["E_hs", "E_nhs"]}
fcast_datasets_variables = {"lem": ["E_hs", "E_nhs"]}
hist_start = "2026-02-01"
hist_end = "2026-03-16"
fcast_start = "2026-03-17"
fcast_end = "2026-04-12"

# --- usage ---

hist = get_historical(
    reservoir_names, hist_datasets_variables,  hist_start,  hist_end)
fcast = get_forecast(
    reservoir_names, fcast_datasets_variables, fcast_start, fcast_end)

plt.figure(figsize=(12, 5))
plt.plot(hist["date"],  hist["E_nhs (mm)"],
         c="k",   label="E_nhs Historical")
plt.plot(fcast["date"],       fcast["E_nhs ensemble_mean (mm)"],
         c="red", label="E_nhs Forecast mean")
plt.plot(fcast["date"],       fcast["E_nhs ensemble_p10 (mm)"],
         c="red", ls="--")
plt.plot(fcast["date"],       fcast["E_nhs ensemble_p90 (mm)"],
         c="red", ls="--", label="E_nhs p10-p90 range")

plt.plot(hist["date"],  hist["E_hs (mm)"],
         c="CornflowerBlue",   label="E_hs Historical")
plt.plot(fcast["date"],       fcast["E_hs ensemble_mean (mm)"],
         c="Violet", label="E_hs Forecast mean")
plt.plot(fcast["date"],       fcast["E_hs ensemble_p10 (mm)"],
         c="Violet", ls="--")
plt.plot(fcast["date"],       fcast["E_hs ensemble_p90 (mm)"],
         c="Violet", ls="--", label="E_hs p10-p90 range")

plt.title("Historical & Forecast Evaporation Comparison — CANYON LAKE")
plt.ylabel("mm/day")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
