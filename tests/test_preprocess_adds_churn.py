import pandas as pd
from src.data_preprocessing import preprocess_dataframe


def test_preprocess_adds_churn():
    df = pd.DataFrame({'customerID': [1,2,3], 'featureA':[1,2,3]})
    cfg = {'target_column': 'Churn'}
    out = preprocess_dataframe(df, cfg=cfg)
    assert 'Churn' in out.columns
    assert out['Churn'].isna().all()
