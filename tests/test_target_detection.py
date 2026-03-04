import pandas as pd
from src.utils import detect_target_candidates


def test_detect_target_exact_name():
    df = pd.DataFrame({'customerID': [1,2,3], 'Churn': ['Yes','No','No']})
    cands = detect_target_candidates(df, id_column='customerID')
    assert cands and cands[0][0].lower() == 'churn' and cands[0][1] >= 0.9


def test_detect_binary_numeric():
    df = pd.DataFrame({'id':[1,2,3,4], 'label':[1,0,1,0]})
    cands = detect_target_candidates(df, id_column='id')
    assert any(c[0]=='label' and c[1]>=0.95 for c in cands)


def test_detect_low_cardinality():
    df = pd.DataFrame({'A':[1,2,1,2,1], 'B':['x','x','y','x','y']})
    cands = detect_target_candidates(df)
    # Both A and B are low cardinality; scores should be >= 0.6
    assert any(c[0]=='A' and c[1]>=0.6 for c in cands)
    assert any(c[0]=='B' and c[1]>=0.6 for c in cands)
