import pandas as pd
from primus.category import LeaveOneOutEncoder



def test_leave_one_out_values():
    df = pd.DataFrame({
        'color': ["a", "a", "a", "b", "b", "b"],
        'outcome': [1, 0, 0, 1, 0, 1]})

    X = df.drop('outcome', axis=1)
    y = df.drop('color', axis=1)

    ce_leave = LeaveOneOutEncoder(cols=['color'])
    obtained = ce_leave.fit_transform(X, y['outcome'])
    print("\nobstained\n", obtained)

    assert([0.0, 0.5, 0.5, 0.5, 1.0, 0.5], list(obtained['color']))