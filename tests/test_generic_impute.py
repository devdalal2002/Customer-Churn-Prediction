import pandas as pd
from src.data_preprocessing import generic_impute
from src.feature_engineering import build_preprocessor_from_config


def test_generic_impute_median_and_mode():
    df = pd.DataFrame({
        'A': [1.0, None, 3.0],
        'B': [None, None, 2.0],
        'C': ['x', None, 'x']
    })
    cfg = {'imputation': {'default_numeric': 'median', 'default_categorical': 'mode'}}
    out = generic_impute(df, cfg)
    assert out['A'].tolist() == [1.0, 2.0, 3.0]
    assert out['B'].tolist() == [2.0, 2.0, 2.0]
    assert out['C'].tolist() == ['x', 'x', 'x']


def test_generic_impute_telco_totalcharges_strategy():
    df = pd.DataFrame({
        'customerID': ['1'],
        'tenure': [0],
        'MonthlyCharges': [50.0],
        'TotalCharges': [None],
    })
    cfg = {'imputation': {'column_imputations': {'TotalCharges': 'telco_total_charges'}}}
    out = generic_impute(df, cfg)
    assert 'TotalCharges_imputed' in out.columns
    assert out.loc[0, 'TotalCharges'] == 0.0
    assert bool(out.loc[0, 'TotalCharges_imputed']) is True


def test_build_preprocessor_from_config():
    df = pd.DataFrame({'num1': [1, 2], 'num2': [3, 4], 'cat1': ['a', 'b']})
    cfg = {'features': {'numeric': ['num1'], 'categorical': ['cat1']}}
    pre = build_preprocessor_from_config(cfg, df)
    # ColumnTransformer stores the column names in transformers_[0][2] etc after fitting in sklearn >=1.0,
    # but we can check the named transformers exist
    assert hasattr(pre, 'transformers')
    names = [t[0] for t in pre.transformers]
    assert 'num' in names and 'cat' in names