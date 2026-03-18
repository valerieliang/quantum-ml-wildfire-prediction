"""
Step 2 - Feature Engineering

Builds a leakage-safe zip x year feature matrix from monthly weather data,
fire-event data, and optional census features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _split_weather_and_fire(
    df: pd.DataFrame,
    fire_df: pd.DataFrame | None,
    fire_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fire_df is not None:
        return df.copy(), fire_df.copy()
    if fire_id_col not in df.columns:
        raise ValueError(
            f"fire_df was not provided and '{fire_id_col}' is not in df; "
            "cannot split weather vs fire rows."
        )
    weather = df[df[fire_id_col].isna()].copy()
    fires = df[df[fire_id_col].notna()].copy()
    return weather, fires


def _build_weather_features(
    weather_df: pd.DataFrame,
    zip_col: str,
    year_month_col: str,
    tmax_col: str,
    prcp_col: str,
) -> pd.DataFrame:
    _require_columns(
        weather_df, [zip_col, year_month_col, tmax_col, prcp_col], "weather_df"
    )

    weather = weather_df[[zip_col, year_month_col, tmax_col, prcp_col]].copy()
    weather = weather[weather[zip_col].notna()].copy()
    weather[zip_col] = weather[zip_col].astype(int)
    weather["date"] = pd.to_datetime(
        weather[year_month_col].astype(str) + "-01", errors="coerce"
    )
    weather = weather[weather["date"].notna()].copy()
    weather[tmax_col] = pd.to_numeric(weather[tmax_col], errors="coerce")
    weather[prcp_col] = pd.to_numeric(weather[prcp_col], errors="coerce")

    # Deduplicate join-expanded rows at zip x month and keep one monthly signal.
    weather = (
        weather.groupby([zip_col, "date"], as_index=False)
        .agg(
            avg_tmax_c=(tmax_col, "mean"),
            tot_prcp_mm=(prcp_col, "mean"),
        )
        .sort_values([zip_col, "date"])
    )

    weather["year"] = weather["date"].dt.year.astype(int)
    weather["month"] = weather["date"].dt.month.astype(int)

    # Strict anti-leakage: shift by 1 month before rolling/cumulative ops.
    by_zip_tmax = weather.groupby(zip_col)["avg_tmax_c"]
    by_zip_prcp = weather.groupby(zip_col)["tot_prcp_mm"]

    weather["rolling_12m_avg_tmax"] = by_zip_tmax.transform(
        lambda s: s.shift(1).rolling(window=12, min_periods=12).mean()
    )
    weather["prcp_12m_prior"] = by_zip_prcp.transform(
        lambda s: s.shift(1).rolling(window=12, min_periods=12).sum()
    )

    # Fallback for early months with short history (still prior-only via shift).
    weather["tmax_expanding_prior"] = by_zip_tmax.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    weather["prcp_expanding_prior"] = by_zip_prcp.transform(
        lambda s: s.shift(1).expanding(min_periods=1).sum()
    )
    weather["rolling_12m_avg_tmax"] = weather["rolling_12m_avg_tmax"].fillna(
        weather["tmax_expanding_prior"]
    )
    weather["prcp_12m_prior"] = weather["prcp_12m_prior"].fillna(
        weather["prcp_expanding_prior"]
    )

    # Yearly features are anchored at January of each year.
    year_start = weather[weather["month"] == 1].copy()
    if year_start.empty:
        raise ValueError("No January rows found; cannot create leakage-safe yearly rows.")
    year_start = year_start.sort_values([zip_col, "year"])

    # Precipitation deficit:
    # deficit(Y) = max(0, climatology_prior(Y) - prior_12m_precip(Y))
    # where climatology_prior uses only years <= Y-2 (shift before expanding mean).
    year_start["prcp_climatology_prior"] = year_start.groupby(zip_col)[
        "prcp_12m_prior"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    year_start["precip_deficit"] = (
        year_start["prcp_climatology_prior"] - year_start["prcp_12m_prior"]
    ).clip(lower=0)
    year_start["precip_deficit"] = year_start["precip_deficit"].fillna(0.0)

    # Drought proxy: hotter + drier => larger ratio.
    year_start["drought_proxy_ratio"] = year_start["rolling_12m_avg_tmax"] / (
        year_start["prcp_12m_prior"].clip(lower=0) + 1.0
    )

    return year_start[
        [zip_col, "year", "rolling_12m_avg_tmax", "precip_deficit", "drought_proxy_ratio"]
    ].copy()


def _build_fire_features(
    fire_df: pd.DataFrame,
    base_index: pd.DataFrame,
    zip_col: str,
    fire_year_col: str,
    acres_col: str,
    training_window: tuple[int, int] | None,
) -> pd.DataFrame:
    year_col = fire_year_col if fire_year_col in fire_df.columns else "Year"
    _require_columns(fire_df, [zip_col, year_col, acres_col], "fire_df")

    fires = fire_df[[zip_col, year_col, acres_col]].copy()
    fires = fires[fires[zip_col].notna() & fires[year_col].notna()].copy()
    fires[zip_col] = fires[zip_col].astype(int)
    fires["year"] = pd.to_numeric(fires[year_col], errors="coerce").astype("Int64")
    fires = fires[fires["year"].notna()].copy()
    fires["year"] = fires["year"].astype(int)
    fires[acres_col] = pd.to_numeric(fires[acres_col], errors="coerce").fillna(0.0)

    annual = (
        fires.groupby([zip_col, "year"], as_index=False)
        .agg(annual_fire_count=("year", "size"), annual_acres_burned=(acres_col, "sum"))
    )

    fire_year = base_index[[zip_col, "year"]].drop_duplicates().copy()
    fire_year = fire_year.merge(annual, on=[zip_col, "year"], how="left")
    fire_year["annual_fire_count"] = fire_year["annual_fire_count"].fillna(0.0)
    fire_year["annual_acres_burned"] = fire_year["annual_acres_burned"].fillna(0.0)
    fire_year = fire_year.sort_values([zip_col, "year"])

    # Strict anti-leakage for fire history: shift after cumulative sum.
    fire_year["cumulative_acres_burned_prior"] = fire_year.groupby(zip_col)[
        "annual_acres_burned"
    ].transform(lambda s: s.cumsum().shift(1).fillna(0.0))

    if training_window is None:
        fire_year["fire_frequency_train_window"] = fire_year.groupby(zip_col)[
            "annual_fire_count"
        ].transform(lambda s: s.cumsum().shift(1).fillna(0.0))
    else:
        start_year, end_year = training_window
        in_window = fire_year["year"].between(start_year, end_year)
        fire_year["window_fire_count"] = np.where(
            in_window, fire_year["annual_fire_count"], 0.0
        )
        fire_year["fire_frequency_train_window"] = fire_year.groupby(zip_col)[
            "window_fire_count"
        ].transform(lambda s: s.cumsum().shift(1).fillna(0.0))
        fire_year = fire_year.drop(columns=["window_fire_count"])

    return fire_year[
        [zip_col, "year", "cumulative_acres_burned_prior", "fire_frequency_train_window"]
    ].copy()


def _merge_census_features(
    feature_df: pd.DataFrame,
    census_df: pd.DataFrame | None,
    zip_col: str,
    census_zip_col: str,
    census_year_col: str | None,
    census_feature_cols: Sequence[str] | None,
) -> pd.DataFrame:
    if census_df is None:
        return feature_df

    if census_zip_col not in census_df.columns:
        raise ValueError(f"census_df is missing '{census_zip_col}'")

    census = census_df.copy()
    census["zip_key"] = pd.to_numeric(census[census_zip_col], errors="coerce").astype("Int64")
    census = census[census["zip_key"].notna()].copy()
    census["zip_key"] = census["zip_key"].astype(int)

    if census_feature_cols is None:
        preferred = ["population_density", "distance_to_wildland"]
        census_feature_cols = [col for col in preferred if col in census.columns]
        if not census_feature_cols:
            census_feature_cols = [
                col
                for col in census.columns
                if col not in {census_zip_col, "zip_key", census_year_col}
            ]

    missing = [col for col in census_feature_cols if col not in census.columns]
    if missing:
        raise ValueError(f"census_df is missing requested census columns: {missing}")

    cols = ["zip_key"] + list(census_feature_cols)
    if census_year_col is not None:
        if census_year_col not in census.columns:
            raise ValueError(f"census_df is missing '{census_year_col}'")
        # Lag by one year so features for Y only see census info up to Y-1.
        census = census[cols + [census_year_col]].copy()
        census["year"] = pd.to_numeric(census[census_year_col], errors="coerce").astype("Int64")
        census = census[census["year"].notna()].copy()
        census["year"] = census["year"].astype(int) + 1
        merge_keys_left = [zip_col, "year"]
        merge_keys_right = ["zip_key", "year"]
    else:
        census = census[cols].copy()
        merge_keys_left = [zip_col]
        merge_keys_right = ["zip_key"]

    merged = feature_df.merge(
        census,
        left_on=merge_keys_left,
        right_on=merge_keys_right,
        how="left",
    ).drop(columns=["zip_key"])

    for col in census_feature_cols:
        if col not in merged.columns:
            continue
        if pd.api.types.is_numeric_dtype(merged[col]):
            merged[col] = merged[col].fillna(merged[col].median())
        else:
            mode = merged[col].mode(dropna=True)
            merged[col] = merged[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    return merged


def build_feature_matrix(
    df: pd.DataFrame,
    fire_df: pd.DataFrame | None = None,
    census_df: pd.DataFrame | None = None,
    *,
    zip_col: str = "zip",
    year_month_col: str = "year_month",
    tmax_col: str = "avg_tmax_c",
    prcp_col: str = "tot_prcp_mm",
    acres_col: str = "GIS_ACRES",
    fire_year_col: str = "year",
    fire_id_col: str = "OBJECTID",
    training_window: tuple[int, int] | None = None,
    census_zip_col: str = "zip",
    census_year_col: str | None = None,
    census_feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build leakage-safe zip x year features for Step 2 only.

    Output columns include:
      - rolling_12m_avg_tmax
      - precip_deficit
      - cumulative_acres_burned_prior
      - fire_frequency_train_window
      - drought_proxy_ratio
      - optional census columns (if provided)
    """

    weather_df, fire_events = _split_weather_and_fire(df, fire_df, fire_id_col)

    weather_features = _build_weather_features(
        weather_df=weather_df,
        zip_col=zip_col,
        year_month_col=year_month_col,
        tmax_col=tmax_col,
        prcp_col=prcp_col,
    )

    fire_features = _build_fire_features(
        fire_df=fire_events,
        base_index=weather_features[[zip_col, "year"]],
        zip_col=zip_col,
        fire_year_col=fire_year_col,
        acres_col=acres_col,
        training_window=training_window,
    )

    feature_df = weather_features.merge(fire_features, on=[zip_col, "year"], how="left")

    core_cols = [
        "rolling_12m_avg_tmax",
        "precip_deficit",
        "cumulative_acres_burned_prior",
        "fire_frequency_train_window",
        "drought_proxy_ratio",
    ]
    for col in core_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)

    feature_df = _merge_census_features(
        feature_df=feature_df,
        census_df=census_df,
        zip_col=zip_col,
        census_zip_col=census_zip_col,
        census_year_col=census_year_col,
        census_feature_cols=census_feature_cols,
    )

    feature_df = (
        feature_df.sort_values([zip_col, "year"]).drop_duplicates([zip_col, "year"]).reset_index(drop=True)
    )
    return feature_df


def sanity_check_feature_matrix(
    feature_df: pd.DataFrame,
    fire_df: pd.DataFrame,
    *,
    zip_col: str = "zip",
    year_col: str = "year",
    fire_year_col: str = "year",
    acres_col: str = "GIS_ACRES",
) -> dict[str, bool]:
    """
    Basic Step 2 checks:
      1) no duplicate zip x year rows
      2) no NaNs in core engineered features
      3) no fire leakage in cumulative_acres_burned_prior
    """
    checks: dict[str, bool] = {}

    checks["unique_zip_year"] = not feature_df.duplicated([zip_col, year_col]).any()

    core_cols = [
        "rolling_12m_avg_tmax",
        "precip_deficit",
        "cumulative_acres_burned_prior",
        "fire_frequency_train_window",
        "drought_proxy_ratio",
    ]
    checks["no_nans_core_features"] = bool(
        feature_df[core_cols].isna().sum().sum() == 0
        if set(core_cols).issubset(feature_df.columns)
        else False
    )

    fire_year = fire_year_col if fire_year_col in fire_df.columns else "Year"
    _require_columns(fire_df, [zip_col, fire_year, acres_col], "fire_df")
    fire = fire_df[[zip_col, fire_year, acres_col]].copy()
    fire = fire[fire[zip_col].notna() & fire[fire_year].notna()].copy()
    fire[zip_col] = fire[zip_col].astype(int)
    fire["year"] = pd.to_numeric(fire[fire_year], errors="coerce").astype("Int64")
    fire = fire[fire["year"].notna()].copy()
    fire["year"] = fire["year"].astype(int)
    fire[acres_col] = pd.to_numeric(fire[acres_col], errors="coerce").fillna(0.0)

    annual_sparse = (
        fire.groupby([zip_col, "year"], as_index=False)
        .agg(annual_acres_burned=(acres_col, "sum"))
    )

    # Rebuild on the same dense zip-year grid as feature_df to validate leakage correctly.
    annual = (
        feature_df[[zip_col, year_col]]
        .rename(columns={year_col: "year"})
        .drop_duplicates()
        .merge(annual_sparse, on=[zip_col, "year"], how="left")
    )
    annual["annual_acres_burned"] = annual["annual_acres_burned"].fillna(0.0)
    annual = annual.sort_values([zip_col, "year"])
    annual["expected_cum_prior"] = annual.groupby(zip_col)["annual_acres_burned"].transform(
        lambda s: s.cumsum().shift(1).fillna(0.0)
    )

    merged = feature_df[[zip_col, year_col, "cumulative_acres_burned_prior"]].merge(
        annual[[zip_col, "year", "expected_cum_prior"]],
        left_on=[zip_col, year_col],
        right_on=[zip_col, "year"],
        how="left",
    )
    merged["expected_cum_prior"] = merged["expected_cum_prior"].fillna(0.0)
    checks["no_fire_leakage_cumulative"] = np.allclose(
        merged["cumulative_acres_burned_prior"].to_numpy(),
        merged["expected_cum_prior"].to_numpy(),
        equal_nan=True,
    )

    return checks


if __name__ == "__main__":
    # Example local run for this repo (Step 2 only):
    # python feature_engineering.py
    raw = pd.read_csv("data/wildfire_weather.csv", low_memory=False)
    fires = pd.read_csv("data/fires_geocoded.csv", low_memory=False)

    matrix = build_feature_matrix(
        df=raw,
        fire_df=fires,
        training_window=(2018, 2022),
    )
    print("Feature matrix shape:", matrix.shape)
    print("Columns:", matrix.columns.tolist())

    checks = sanity_check_feature_matrix(matrix, fires)
    print("Sanity checks:", checks)
    matrix.to_csv("feature_matrix.csv")

