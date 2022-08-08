import pandas as pd

from pandas_questions import load_data
from pandas_questions import q0_how_many_records_per_site
from pandas_questions import q1_resample_3h
from pandas_questions import q2_merge_weather
from pandas_questions import q3_days_to_holiday


def test_how_many_records_per_site():
    df, _ = load_data()
    counts = q0_how_many_records_per_site(df)
    assert counts.size == 70
    assert counts.sum() == len(df)
    assert all(counts.values[-5:] == [9548, 9389, 9129, 7372, 6154])


def test_resample_3h():
    df, _ = load_data()
    df_3h = q1_resample_3h(df)
    assert all(df_3h.columns == df.columns)
    assert all(df_3h['date'].dt.hour.values % 3 == 0)
    assert len(df_3h) == 313371


def test_merge_weather():
    df, weather = load_data()

    cols = ['t', 'rr1']
    small_weather = weather.sort_values('date').iloc[:2]

    date_limit = small_weather['date'].iloc[-1]  # noqa: F841
    small_df = df.query("date <= @date_limit & site_id == 100003096")
    small_merged = q2_merge_weather(small_df, weather)

    merged_cols = list(df.columns) + cols
    assert set(merged_cols) == set(small_merged.columns)

    first = small_merged.query('date < @date_limit')
    assert all(first[cols] == small_weather.iloc[0][cols])
    second = small_merged.query('date == @date_limit')
    assert all(second[cols] == small_weather.iloc[1][cols])


def test_days_to_holidays():

    df, _ = load_data()

    labor_day = pd.Timestamp(year=2021, month=5, day=1)

    small_df = df.query("site_id == 100003096")
    small_df = small_df[abs(small_df['date'] - labor_day).dt.days < 8]

    small_df['to_holiday'] = q3_days_to_holiday(small_df)

    res = (
        small_df.groupby(small_df['date'].dt.floor('D'))['to_holiday'].first()
    )
    assert res['2021-04-30'] == pd.Timedelta('-1D')
    assert res['2021-05-01'] == pd.Timedelta('0D')
    assert res['2021-05-02'] == pd.Timedelta('1D')
    assert res['2021-05-04'] == pd.Timedelta('3D')
    assert res['2021-05-05'] == pd.Timedelta('-3D')
    assert res['2021-05-08'] == pd.Timedelta('0D')

    # Test that if some holidays are not in the date range, they
    # are still considered for the computation
    small_df = df.query("site_id == 100003096").set_index('date')
    small_df = small_df['2021-05-02':'2021-08-01 23:59'].reset_index()
    small_df['to_holiday'] = q3_days_to_holiday(small_df)

    res = (
        small_df.groupby(small_df['date'].dt.floor('D'))['to_holiday'].first()
    )
    assert res['2021-05-02'] == pd.Timedelta('1D')
    assert res['2021-08-01'] == pd.Timedelta('-14D')
