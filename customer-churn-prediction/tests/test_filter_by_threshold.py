import pandas as pd
from src.utils import filter_by_threshold


def test_filter_by_threshold_basic():
    df = pd.DataFrame({'customerID': ['1','2','3'], 'churn_proba': [0.1, 0.6, 0.8]})
    out = filter_by_threshold(df, 'churn_proba', 0.5)
    assert len(out) == 2
    assert out['customerID'].tolist() == ['2','3']


def test_filter_by_threshold_missing_col():
    df = pd.DataFrame({'a':[1,2]})
    try:
        filter_by_threshold(df, 'churn_proba', 0.5)
        assert False, "Expected KeyError"
    except KeyError:
        assert True