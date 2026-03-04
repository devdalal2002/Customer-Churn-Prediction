import pytest
from src.utils import map_requested_to_actual


def test_map_requested_to_actual_case_and_substring():
    actual = ['customerID','tenure_months','monthly_charges','TotalCharges','Churn']
    requested = ['tenure','MonthlyCharges','Gender','TotalCharges']
    mapped, missing, mapping = map_requested_to_actual(requested, actual)
    assert 'tenure_months' in mapped
    assert 'monthly_charges' in mapped
    assert 'TotalCharges' in mapped
    assert 'Gender' in missing
    assert mapping['Gender'] is None
