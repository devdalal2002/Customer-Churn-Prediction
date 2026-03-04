from src.utils import infer_column_mapping


def test_infer_basic():
    cols = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    out = infer_column_mapping(cols)
    assert out['id'] == 'customerID'
    assert out['target'] == 'Churn'


def test_infer_variants():
    cols = ['id', 'months', 'monthly_charge', 'is_churned']
    out = infer_column_mapping(cols)
    assert out['id'] == 'id'
    assert out['target'] == 'is_churned'


def test_infer_none():
    cols = ['a','b','c']
    out = infer_column_mapping(cols)
    assert out['id'] is None and out['target'] is None