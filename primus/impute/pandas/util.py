import re
import pandas as pd
import numpy as np
from typing import List, Union


def empty_to_none(X: pd.DataFrame) -> pd.DataFrame:
    X.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return X


def strs_to_nonev_0(X: Union[pd.Series, pd.DataFrame], strs: List[str]) -> Union[pd.Series, pd.DataFrame]:
    p_v_dict = {re.compile('^' + n + '$', re.IGNORECASE): '' for n in strs}
    def sub_object(o):
        if pd.isnull(o):
            return o
        text = str(o)
        for p, v in p_v_dict.items():
            text = p.sub(v, text)
            if not text:
                return np.nan
        return text

    if isinstance(X, pd.Series):
        return X.apply(sub_object)

    return X.applymap(sub_object)


def strs_to_none(X: Union[pd.Series, pd.DataFrame], strs: List[str]) -> Union[pd.Series, pd.DataFrame]:
    p_v_dict = {'(?i)^'+n+'$': np.nan for n in strs}
    X.replace(p_v_dict, inplace=True, regex=True)
    return X
