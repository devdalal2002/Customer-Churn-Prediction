"""
Explainability helpers using SHAP.

Functions:
- explain_model_shap: compute SHAP values and save summary / bar plots
"""
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
except Exception:
    shap = None


def explain_model_shap(model, X, feature_names=None, out_dir: str = 'reports/figures'):
    """Compute SHAP values for `model` given feature matrix X and save summary plots.

    Returns: shap_values object when computed, else None.
    """
    os.makedirs(out_dir, exist_ok=True)
    if shap is None:
        print('shap library not available. Install with: pip install shap')
        return None

    # handle pipelines (preprocessor + clf) by extracting the final estimator and transformed X
    try:
        final_clf = model
        X_to_explain = X
        feature_names_out = feature_names
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(model, Pipeline):
                pre = model.named_steps.get('preprocessor')
                final_clf = model.named_steps.get('clf')
                if pre is not None:
                    X_to_explain = pre.transform(X)
                    try:
                        feature_names_out = pre.get_feature_names_out()
                    except Exception:
                        # fallback to provided raw feature names
                        feature_names_out = feature_names
        except Exception:
            pass

        if hasattr(shap, 'TreeExplainer') and (hasattr(final_clf, 'feature_importances_') or final_clf.__class__.__name__.lower().startswith('xg')):
            explainer = shap.TreeExplainer(final_clf)
        else:
            explainer = shap.Explainer(final_clf.predict, X_to_explain)
        shap_values = explainer(X_to_explain)
    except Exception as e:
        print('SHAP explain error:', e)
        return None

    # summary plot
    try:
        shap.summary_plot(shap_values, X_to_explain, feature_names_out if feature_names_out is not None else None, show=False)
        plt.savefig(os.path.join(out_dir, 'shap_summary.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print('Failed to save shap summary plot:', e)

    # bar plot of mean absolute SHAP values
    try:
        vals = np.abs(shap_values.values).mean(0)
        names = feature_names_out if feature_names_out is not None else [f'f{i}' for i in range(len(vals))]
        idx = np.argsort(vals)[-10:][::-1]
        plt.figure(figsize=(8,6))
        plt.barh([names[i] for i in idx][::-1], vals[idx][::-1])
        plt.title('Top 10 mean(|SHAP value|)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_top10.png'))
        plt.close()
    except Exception as e:
        print('Failed to save shap bar plot:', e)

    return shap_values
