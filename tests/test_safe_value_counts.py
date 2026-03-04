import pandas as pd
from src.utils import safe_value_counts


def test_safe_value_counts_basic():
    df = pd.DataFrame({'Churn': ['Yes','No','Yes']})
    vc = safe_value_counts(df, 'Churn')
    assert vc['Yes'] == 2 and vc['No'] == 1


def test_safe_value_counts_duplicate_cols():
    # create DataFrame with duplicate column names
    df = pd.DataFrame([[1,'Yes'], [2,'No']], columns=['id','Churn'])
    # artificially add another 'Churn' column
    df['Churn'] = ['Yes','No']
    # create duplicate by concatenating
    df2 = pd.concat([df, df[['Churn']]], axis=1)
    # ensure duplicate labels
    df2.columns = ['id','Churn','Churn']
    vc = safe_value_counts(df2, 'Churn')
    assert vc['Yes'] == 2 and vc['No'] == 2


def test_safe_value_counts_missing():
    df = pd.DataFrame({'a':[1,2]})
    try:
        safe_value_counts(df, 'Churn')
        assert False
    except KeyError:
        assert True