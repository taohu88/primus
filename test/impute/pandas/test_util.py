import numpy as np
import pandas as pd
from primus.impute.pandas import empty_to_none, strs_to_none


def test_empty_to_none():
    df = pd.DataFrame({'A': ['', "Hello", "world"]})
    print("\n")
    print(df)
    df2 = empty_to_none(df)
    print(df2)
    expected = pd.DataFrame({'A': [np.nan, "Hello", "world"]})
    assert(df2.equals(expected))


def test_str_series_to_none():
    s = pd.Series(['', "Hello", "world"])
    print("\n")
    print(s)
    s2 = strs_to_none(s, ['', 'hello'])
    expected = pd.Series([np.nan, np.nan, "world"])
    assert(s2.equals(expected))


def test_strs_to_none():
    df = pd.DataFrame({'A': ['', "Hello", "world", "ok"],
                       'B': ['--', 'n/a', 'NA', "win"]})
    print("\n", df)
    df2 = strs_to_none(df, ['', '--', 'n/a', 'na'])
    print(df2)
    expected = pd.DataFrame({'A': [np.nan, "Hello", "world", "ok"],
                       'B': [np.nan]*3 + ["win"]})
    assert(df2.equals(expected))
