import os
import subprocess
from pathlib import Path


def test_export_for_powerbi_creates_files(tmp_path):
    # Run the script and check that it outputs a file in `data/processed/telco_powerbi.csv` or to the working dir
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'scripts' / 'export_for_powerbi.py'
    # run via python -u to ensure environment is correct
    res = subprocess.run(['python', str(script)], cwd=str(repo_root), capture_output=True)
    assert res.returncode == 0
    out_path = repo_root / 'data' / 'processed' / 'telco_powerbi.csv'
    assert out_path.exists()
    metrics_path = repo_root / 'reports' / 'figures' / 'powerbi_metrics.csv'
    assert metrics_path.exists()
