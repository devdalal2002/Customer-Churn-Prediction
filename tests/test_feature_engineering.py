import pandas as pd
from src.feature_engineering import create_features


def test_create_features_adds_columns():
    df = pd.DataFrame({
        'customerID': ['1', '2'],
        'tenure': [0, 12],
        'MonthlyCharges': [10.0, 20.0],
        'TotalCharges': [0.0, 240.0],
    })
    out = create_features(df)
    assert 'TenureGroup' in out.columns
    assert 'TotalChargesPerMonth' in out.columns
    # TotalChargesPerMonth should handle tenure 0 by replacing divide by zero with 0's (or similar)
    assert out.loc[0, 'TotalChargesPerMonth'] == 0.0
    assert out.loc[1, 'TotalChargesPerMonth'] == 20.0
