import pandas as pd
from primus.category import OneHotEncoder


def test_fit_transform_HaveHandleUnknownValueAndUnseenValues_ExpectAllZeroes():
    train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
    test = pd.DataFrame({'city': ['Chicago', 'Detroit']})
    expected_result = pd.DataFrame({'city_1': [1, 0],
                                    'city_2': [0, 0]},
                                   columns=['city_1', 'city_2'])

    print("\ntrain\n", train)
    print("test\n", test)
    print("expected \n", expected_result)
    enc = OneHotEncoder(handle_unknown='value')
    result = enc.fit(train).transform(test)
    print("result\n", result)

    pd.testing.assert_frame_equal(expected_result, result)