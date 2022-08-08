"""Pandas assignment questions.
"""
from pathlib import Path

import pandas as pd

from utils import FrBusinessCalendar

FrHolidays = FrBusinessCalendar()


DATA_DIR = Path('data')


def load_data():

    df = pd.read_parquet(DATA_DIR / 'bike-counter-data.parquet')
    weather = pd.read_csv(DATA_DIR / "external_data.csv")
    weather['date'] = pd.to_datetime(weather['date'])

    return df, weather


def q0_how_many_records_per_site(df):
    """Return a Series that contains the number of available
    data points (records) for each of site_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the bike counters.

    Returns
    -------
    df_counts_per_site : pd.Series
        The number of records per site. The index of the Series
        should be the side_id and the Series should be sorted
        by the count value in descending order (from the most
        frequent to the least frequent site in the dataframe).
    """
    return df['bike_count'] * 0


def q1_resample_3h(df):
    """For each bike counter resample the data with 3H time delta between rows.
    You will take the mean value within 3H time windows for each counter_id
    and then put back the data in its original form (same colums) as before
    but now the date columns is based on 3H intervals between values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the bike counters.

    Returns
    -------
    df_3h : pd.DataFrame
        DataFrame containing all counter_id but now with
    """
    df_3h = df
    return df_3h


def q2_merge_weather(df, weather):
    """Merge weather from the nearest previous date into the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the bike counters.
    weather : pd.DataFrame
        DataFrame containing external data with meteorological conditions.

    Returns
    -------
    merged_df : pd.DataFrame
        dataframe with columns 't' and 'rr1' merged from weather.
    """
    merged_df = df
    return merged_df


def q3_days_to_holiday(df):
    """Return a Series containing the number of days to the closest holiday.

    Return the number of days (integer) to get the relative distance to
    the nearest holiday. If you are one day after the holidays you should
    have 1. And if you are 2 days before it should be -2. On holidays it
    should contain a 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a `date` column

    Returns
    -------
    to_holiday : pd.Series(int)
        the number of days to get the relative distance to the nearest holiday.
    """

    to_holiday = pd.Timestamp.now() - df['date']
    return to_holiday


if __name__ == "__main__":

    df, weather = load_data()

    print(q1_resample_3h(df))
    # # Merge external weather data into the bike counter DataFrame.
    # df = q2_merge_weather(df, weather)

    # # Add a column to holidays
    # df['to_holiday'] = q3_days_to_holiday(df)
