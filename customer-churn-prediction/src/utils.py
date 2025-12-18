import yaml
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file and return as a dict.

    Raises FileNotFoundError if the path does not exist.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_yaml(path: str, content: Dict[str, Any]) -> None:
    """Save config dict to YAML file.
    Creates directories if needed.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(content, f)


def filter_by_threshold(df, prob_col: str = 'churn_proba', threshold: float = 0.5):
    """Return rows where probability column >= threshold.

    Parameters
    - df: pd.DataFrame with probability column
    - prob_col: name of probability column
    - threshold: cutoff in [0.0, 1.0]

    Returns a DataFrame filtered by the threshold. If the probability column
    is not found, raises KeyError.
    """
    if prob_col not in df.columns:
        raise KeyError(f"Probability column '{prob_col}' not found in DataFrame")
    return df[df[prob_col] >= threshold].copy()
