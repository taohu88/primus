import pandas as pd
from primus.category import OccOrdEncorder


def test_pandas_categorical():
    X = pd.DataFrame({
        'Str': ['d', 'c', 'c', 'a'],
        'Categorical': pd.Categorical(list('bbea'), categories=['e', 'a', 'b'], ordered=True)
    })
    print("\n", X.head())

    oe = OccOrdEncorder()
    out = oe.fit_transform(X)
    print(out)

    print(oe.category_mapping)

    assert(3, out['Categorical'][0])
    assert(3, out['Categorical'][1])
    assert(1, out['Categorical'][2])
    assert(2, out['Categorical'][3])

