import pandas as pd
from src.data_preprocessing import clean_total_charges


def test_clean_total_charges_tenure_zero():
    df = pd.DataFrame({
        'customerID': ['1'],
        'tenure': [0],
        'MonthlyCharges': [50.0],
        'TotalCharges': [None],
    })
    df2 = clean_total_charges(df)
    assert 'TotalCharges_imputed' in df2.columns
    assert df2.loc[0, 'TotalCharges'] == 0.0
    assert bool(df2.loc[0, 'TotalCharges_imputed']) is True


def test_clean_total_charges_monthly_multiplied():
    df = pd.DataFrame({
        'customerID': ['2'],
        'tenure': [10],
        'MonthlyCharges': [20.0],
        'TotalCharges': [None],
    })
    df2 = clean_total_charges(df)
    assert df2.loc[0, 'TotalCharges'] == 200.0
    assert bool(df2.loc[0, 'TotalCharges_imputed']) is True
