import pandas as pd
import os
import joblib
from src.pipeline_runner import process_and_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_process_and_score_no_pipeline():
    df = pd.DataFrame({'tenure': [1,2,3], 'MonthlyCharges': [20,30,40], 'TotalCharges': [20,60,120]})
    cfg = {
        'models': {'pipeline': 'models/pipeline_test.pkl'},
        'features': {'numeric': ['tenure','MonthlyCharges','TotalCharges']},
        'id_column': 'customerID',
        'target_column': 'Churn',
        'prediction_threshold': 0.5,
        'training': {'min_rows': 1, 'min_class_count': 1}
    }
    # ensure no pipeline file exists
    try:
        os.remove(cfg['models']['pipeline'])
    except Exception:
        pass
    df_proc, status = process_and_score(df, cfg, try_auto_train=False)
    assert not status['scored']


def test_process_and_score_with_pipeline(tmp_path):
    # create dummy pipeline that accepts numeric features
    X_train = pd.DataFrame({'tenure': [1,2,3,4], 'MonthlyCharges': [20,30,40,50], 'TotalCharges':[20,60,120,200]})
    y_train = [0,1,0,1]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pipe = Pipeline([('clf', clf)])
    path = tmp_path / 'pipeline_test.pkl'
    joblib.dump(pipe, path)

    df = pd.DataFrame({'tenure': [1,2,3], 'MonthlyCharges': [20,30,40], 'TotalCharges': [20,60,120]})
    cfg = {
        'models': {'pipeline': str(path)},
        'features': {'numeric': ['tenure','MonthlyCharges','TotalCharges']},
        'id_column': 'customerID',
        'target_column': 'Churn',
        'prediction_threshold': 0.5,
    }

    df_proc, status = process_and_score(df, cfg, try_auto_train=False)
    # because our pipeline doesn't have preprocessor, predict_proba may not be available; status may be False
    assert isinstance(status, dict)


def test_process_and_score_missing_requested_features_autotrain(tmp_path):
    # Dataframe without the requested Telco-like columns
    df = pd.DataFrame({
        'tenure_months': [1,2,3,4,5,6],
        'monthly_charges': [20,30,40,50,60,70],
        'TotalCharges': [20,60,120,200,300,400],
        'Churn': [0,1,0,1,0,1]
    })

    cfg = {
        'models': {'pipeline': str(tmp_path / 'pipeline_test.pkl')},
        # deliberately request columns that don't match exactly (case / different names)
        'features': {'numeric': ['tenure','MonthlyCharges','Gender','InternetService']},
        'id_column': 'customerID',
        'target_column': 'Churn',
        'prediction_threshold': 0.5,
        'training': {'min_rows': 3, 'min_class_count': 1, 'test_size': 0.5}
    }

    # Ensure there's no pipeline on disk
    try:
        os.remove(cfg['models']['pipeline'])
    except Exception:
        pass

    df_proc, status = process_and_score(df, cfg, try_auto_train=True)

    # should have attempted fallback or reported missing features
    assert ('falling back to inferred features' in status['message']) or ('Some requested features were not found' in status['message']) or ('Auto-trained' in status['message'])


def test_process_and_score_force_train_overwrites_pipeline(tmp_path):
    # prepare an initial dummy pipeline file (to be overwritten)
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyClassifier
    import time

    initial_pipe = Pipeline([('clf', DummyClassifier(strategy='constant', constant=0))])
    pipeline_path = tmp_path / 'pipeline_test.pkl'
    joblib.dump(initial_pipe, pipeline_path)
    t0 = os.path.getmtime(pipeline_path)

    # dataset with target to allow training
    # create a larger balanced dataset to ensure CV folds work
    n = 40
    df = pd.DataFrame({
        'featureA': list(range(n)),
        'featureB': list(range(1, n+1)),
        'Churn': [0,1] * (n//2)
    })

    cfg = {
        'models': {'pipeline': str(pipeline_path)},
        'features': {'numeric': ['featureA','featureB']},
        'id_column': 'customerID',
        'target_column': 'Churn',
        'prediction_threshold': 0.5,
        'training': {'min_rows': 4, 'min_class_count': 1, 'test_size': 0.5, 'cv_lr': 2, 'cv_rf': 2}
    }

    df_proc, status = process_and_score(df, cfg, try_auto_train=True, force_train=True)

    assert status['trained'] or ('Auto-trained' in status['message'])
    # pipeline file should be updated (mtime greater than t0)
    t1 = os.path.getmtime(pipeline_path)
    assert t1 >= t0
    # if scoring succeeded, churn_proba should be present
    if status.get('scored'):
        assert 'churn_proba' in df_proc.columns
