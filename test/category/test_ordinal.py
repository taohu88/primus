import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def test_pandas_categorical():
    X = pd.DataFrame({
        'Str': ['d', 'c', 'c', 'a'],
        'Categorical': pd.Categorical(list('bbea'), categories=['e', 'a', 'b'], ordered=True)
    })
    print("\n", X.head())

    oe = OrdinalEncoder()
    out = oe.fit_transform(X)
    print(out)

    print(oe.categories_)

    # Note skilearn OrderinalEncoder honor natural order and start from 0
    assert(2, out['Str'][0])
    assert(1, out['Str'][1])
    assert(1, out['Str'][2])
    assert(0, out['Str'][3])

    assert(1, out['Categorical'][0])
    assert(1, out['Categorical'][1])
    assert(2, out['Categorical'][2])
    assert(0, out['Categorical'][3])

