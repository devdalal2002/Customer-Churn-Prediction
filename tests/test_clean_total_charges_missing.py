import pandas as pd
from src.data_preprocessing import clean_total_charges, generic_impute


def test_clean_total_charges_absent_column():
    # If TotalCharges is not present, the function should be a no-op
    df = pd.DataFrame({'a': [1,2], 'b':[3,4]})
    out = clean_total_charges(df)
    assert 'TotalCharges' not in out.columns


def test_clean_total_charges_missing_tenure_monthly():
    df = pd.DataFrame({'TotalCharges': [None, '100']})
    out = clean_total_charges(df)
    assert 'TotalCharges' in out.columns
    assert float(out.loc[0, 'TotalCharges']) == 0.0
    assert bool(out.loc[0, 'TotalCharges_imputed']) is True


def test_generic_impute_skips_telco_when_no_column():
    df = pd.DataFrame({'a':[1, None]})
    cfg = {'imputation': {'column_imputations': {'TotalCharges': 'telco_total_charges'}}}
    out = generic_impute(df, cfg)
    # no changes to unrelated columns besides imputation defaults
    assert 'a' in out.columns
    assert out['a'].isna().sum() == 0  # median imputation should fill the None