import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from category_encoders import OrdinalEncoder

def test_column_transformer():
    titanic = pd.read_csv('../datasets/titanic3.csv')
    # there is still a small problem with using the OneHotEncoder and missing values,
    # so for now I am going to assume there are no missing values by dropping them
    titanic2 = titanic.dropna(subset=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

    target = titanic2.survived.values
    features = titanic2[['pclass', 'sex', 'age', 'fare', 'embarked']]
    print(features.head())

    preprocess = make_column_transformer(
        #(StandardScaler(), ['age', 'fare']),
        (OrdinalEncoder(), ['sex']),
        #(OneHotEncoder(), ['pclass', 'embarked'])
    )

    w = preprocess.fit_transform(features)[:5]
    print(w)
