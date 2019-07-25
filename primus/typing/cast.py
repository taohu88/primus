import math
import pandas as pd
from pandas.api.types import is_scalar


def maybe_to_dtype(ds: pd.Series, check_type_func, to_type_func, max_error_rate=0.01) -> pd.Series:
    # nothing to do
    if check_type_func(ds.dtype):
        return ds

    sz = ds.size
    scalar_cnt = ds.apply(is_scalar).sum()
    # It is not scalar element, nothing to do
    if scalar_cnt != sz:
        return ds

    null_cnt = ds.isnull().sum()
    # It is all null, nothing to do
    if null_cnt == sz:
        return ds

    rds = to_type_func(ds, error='coerce')
    new_null_cnt = rds.isnull().sum()
    # Convert successfully
    if new_null_cnt == null_cnt:
        return rds

    old_cnt = sz - null_cnt
    new_cnt = sz - new_null_cnt

    error_cnt = old_cnt - new_cnt
    max_errors = math.floor(max_error_rate * old_cnt)
    if error_cnt <= max_errors:
        return rds
    return ds


def maybe_to_datetime(ds: pd.Series, max_error_rate=0.01) -> pd.Series:
    return maybe_to_dtype(ds,
                          pd.api.types.is_datetime64_any_dtype,
                          pd.to_datetime,
                          max_error_rate=max_error_rate)


def maybe_to_numeric(ds: pd.Series, max_error_rate=0.01) -> pd.Series:
    return maybe_to_dtype(ds,
                          pd.api.types.is_numeric_dtype,
                          pd.to_numeric,
                          max_error_rate=max_error_rate)


def maybe_to_categorical(ds: pd.Series, max_uniq_rate=0.2, max_uniq=500) -> pd.Series:
    # Nothing to do
    if pd.api.types.is_categorical_dtype(ds.dtype):
        return ds
    # fail fast
    if not pd.api.types.is_string_dtype(ds.dtype):
        return ds

    sz = ds.size
    uniq_cnt = ds.nunique(dropna=False)
    max_uniq = math.ceil(min(max_uniq_rate*sz, max_uniq))
    if uniq_cnt <= max_uniq:
        rds = ds.factorize()
        return rds
    return ds

