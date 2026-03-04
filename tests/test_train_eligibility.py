import pandas as pd
from src.utils import can_train_on_dataframe


def test_can_train_missing_target():
    df = pd.DataFrame({'a':[1,2,3]})
    ok, msg = can_train_on_dataframe(df, 'Churn', min_rows=1, min_class_count=1)
    assert not ok and 'not found' in msg


def test_can_train_insufficient_rows():
    df = pd.DataFrame({'Churn': ['Yes']*50 + ['No']*50})
    ok, msg = can_train_on_dataframe(df, 'Churn', min_rows=200, min_class_count=1)
    assert not ok and 'Insufficient rows' in msg


def test_can_train_success():
    df = pd.DataFrame({'Churn': ['Yes']*150 + ['No']*150})
    ok, msg = can_train_on_dataframe(df, 'Churn', min_rows=100, min_class_count=10)
    assert ok and msg == 'OK'