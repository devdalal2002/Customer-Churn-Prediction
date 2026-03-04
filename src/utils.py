import yaml
import pandas as pd
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


def infer_column_mapping(uploaded_cols, id_candidates=('customerid', 'id', 'custid'), target_candidates=('churn', 'churned', 'is_churn')):
    """Infer best guesses for id and target columns from uploaded column names.

    Returns a dict with keys 'id' and 'target' mapping to suggested column name or None.
    Strategy: exact match (case-insensitive), substring match, candidate synonyms.
    """
    uploaded_lower = {c.lower(): c for c in uploaded_cols}

    def find_best(candidates):
        # exact match
        for cand in candidates:
            if cand.lower() in uploaded_lower:
                return uploaded_lower[cand.lower()]
        # substring match (require meaningful length to avoid accidental single-letter matches)
        for name in uploaded_cols:
            low = name.lower()
            for cand in candidates:
                if (len(cand) >= 3 and cand.lower() in low) or (len(low) >= 3 and low in cand.lower()):
                    return name
        return None

    id_match = find_best(id_candidates)
    target_match = find_best(target_candidates)
    return {'id': id_match, 'target': target_match}


def can_train_on_dataframe(df, target_col: str, min_rows: int = 200, min_class_count: int = 10):
    """Quick checks whether a dataframe is a reasonable candidate for training.

    Checks:
    - target_col exists
    - number of rows >= min_rows
    - each class in target has at least min_class_count samples

    Returns (ok: bool, message: str)
    """
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in uploaded data"
    if len(df) < min_rows:
        return False, f"Insufficient rows: {len(df)} (minimum {min_rows} required)"
    # obtain class counts — map Yes/No if object dtype
    y = df[target_col]
    # handle duplicate columns case where y may be a DataFrame
    if isinstance(y, (list, tuple)):
        # unexpected, coerce
        y = pd.Series(y)
    if isinstance(y, pd.DataFrame):
        # choose the first column if duplicates exist
        y = y.iloc[:, 0]
    if y.dtype == object:
        y_mapped = y.map({'No': 0, 'Yes': 1})
    else:
        y_mapped = y
    counts = y_mapped.value_counts(dropna=True).to_dict()
    # ensure at least two classes and counts
    if len(counts) < 2:
        return False, f"Target must have at least 2 classes, found {len(counts)}"
    if any(v < min_class_count for v in counts.values()):
        return False, f"One or more classes have fewer than {min_class_count} examples: {counts}"
    return True, "OK"


def safe_value_counts(df, col, dropna=False):
    """Return value_counts for a column robustly.

    Handles duplicate column labels by selecting the first matching column.
    Raises KeyError if the column is not present.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    ser = df[col]
    # if duplicate columns result in a DataFrame being returned, aggregate across columns
    if isinstance(ser, pd.DataFrame):
        if ser.shape[1] > 1:
            # flatten all duplicate columns into a single Series to count values across them
            flat = pd.concat([ser.iloc[:, i] for i in range(ser.shape[1])], ignore_index=True)
            if dropna:
                flat = flat.dropna()
            return flat.value_counts()
        else:
            ser = ser.iloc[:, 0]
    return ser.value_counts(dropna=dropna)

def map_requested_to_actual(requested: list, actual_cols: list, case_insensitive: bool = True):
    """Map a list of requested feature names to the actual DataFrame columns.

    Returns (mapped_list, missing_list, mapping_dict).
    - mapped_list: list of actual columns present that match requested names (in order)
    - missing_list: requested names with no match
    - mapping_dict: dict requested_name -> actual_name or None
    Matching strategy:
    1. Exact match
    2. Case-insensitive match
    3. Substring match (if safe length)
    """
    mapped = []
    missing = []
    mapping = {}
    actual_lower = {c.lower(): c for c in actual_cols}

    def _normalize(s: str) -> str:
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    actual_norm = {_normalize(c): c for c in actual_cols}

    for req in requested:
        if req in actual_cols:
            mapping[req] = req
            mapped.append(req)
            continue
        if case_insensitive and req.lower() in actual_lower:
            mapping[req] = actual_lower[req.lower()]
            mapped.append(mapping[req])
            continue
        # normalized substring / exact match
        req_norm = _normalize(req)
        if req_norm in actual_norm:
            mapping[req] = actual_norm[req_norm]
            mapped.append(mapping[req])
            continue
        found = None
        for a in actual_cols:
            a_norm = _normalize(a)
            if len(req_norm) >= 3 and req_norm in a_norm:
                found = a
                break
            if len(a_norm) >= 3 and a_norm in req_norm:
                found = a
                break
        if found:
            mapping[req] = found
            mapped.append(found)
        else:
            mapping[req] = None
            missing.append(req)
    # deduplicate preserving order
    seen = set()
    mapped_dedup = []
    for m in mapped:
        if m not in seen:
            seen.add(m)
            mapped_dedup.append(m)
    return mapped_dedup, missing, mapping


def detect_target_candidates(df, id_column: str = 'customerID'):
    """Detect candidate target columns for binary classification.

    Returns a list of tuples (column_name, score, reason) sorted by score desc.
    Score is in [0,1] where higher is more confident.
    Heuristics applied (in order of priority):
      - exact name match (churn, target, label, y) -> high score
      - binary categorical (values like yes/no, true/false, 0/1) -> high score
      - numeric with exactly {0,1} values -> high score
      - low cardinality categorical (2-10 unique) -> medium score
      - columns with names containing 'churn' (case-insensitive) -> medium-high score

    This function is conservative: id_column is excluded.
    """
    candidates = []
    if id_column in df.columns:
        cols = [c for c in df.columns if c != id_column]
    else:
        cols = list(df.columns)

    def _norm_vals(s):
        try:
            vals = set(pd.Series(s).dropna().unique())
            return {str(v).strip().lower() for v in vals}
        except Exception:
            return set()

    strong_names = {'churn', 'target', 'label', 'y'}
    for c in cols:
        score = 0.0
        reasons = []
        name_lower = c.lower()
        if any(s == name_lower for s in strong_names):
            score += 0.9
            reasons.append('exact_name')
        if 'churn' in name_lower and score < 0.9:
            score += 0.6
            reasons.append('name_contains_churn')

        vals = _norm_vals(df[c])
        # binary yes/no/true/false/0/1
        if vals and vals <= {'yes', 'no'} or vals <= {'y', 'n'} or vals <= {'true', 'false'}:
            score = max(score, 0.95)
            reasons.append('binary_text')
        # numeric binary 0/1
        if vals and vals <= {'0', '1'}:
            score = max(score, 0.95)
            reasons.append('binary_numeric')

        # low cardinality
        try:
            n_unique = df[c].nunique(dropna=True)
        except Exception:
            n_unique = len(vals)
        if 2 <= n_unique <= 10 and score < 0.6:
            score = max(score, 0.6)
            reasons.append('low_cardinality')

        # normalize small strings
        if score > 0.0:
            candidates.append((c, float(score), reasons))

    # sort by score desc then by column name
    candidates_sorted = sorted(candidates, key=lambda x: (-x[1], x[0]))
    return candidates_sorted
