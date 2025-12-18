# Power BI Dashboard Integration

This directory contains instructions and helper files to visualize the Telco churn model outputs in Power BI Desktop or Power BI Service.

Steps to create the dashboard in Power BI Desktop

1. Run the export script to generate a scored dataset and evaluation artifacts

```pwsh
pip install -r requirements.txt
python scripts/export_for_powerbi.py
```

2. Open Power BI Desktop
3. Click `Get Data` → `Text/CSV` and select `data/processed/telco_powerbi.csv`
4. Use the built-in Power Query editor to apply column data types as needed
5. Recommended visuals
   - KPI card: overall churn rate (use `churn_pred` or aggregate `churn_proba`)
   - Donut chart: churn by `gender` or `Partner`
   - Bar chart: churn distribution by `Contract`, `PaymentMethod`, `InternetService`
   - Map (if you add `State` or `Zip`): location-based churn
   - ROC/PR images: `Report` → `Insert` → `Image` → pick `reports/figures/powerbi_roc_curve.png` and `powerbi_pr_curve.png`
6. Measures (DAX) examples

```
ChurnCount = COUNTROWS(FILTER('telco_powerbi', 'telco_powerbi'[churn_pred] = 1))
TotalCount = COUNTROWS('telco_powerbi')
ChurnRate = DIVIDE([ChurnCount], [TotalCount], 0)
AvgChurnProb = AVERAGE('telco_powerbi'[churn_proba])
```

7. For dynamic predictions (refresh), you can re-run the `scripts/export_for_powerbi.py` script and replace the `telco_powerbi.csv` dataset or set up a scheduled export to a shared location.

8. To use Python visuals in Power BI Desktop (optional):
   - File → Options and settings → Options → Python scripting → set Python home directory
   - Add a Python visual and paste a small script to plot features using seaborn/matplotlib

Power BI Service (Publish)
- After building the report in Power BI Desktop, publish to Power BI Service (requires Power BI account).
- For a scheduled refresh, store `telco_powerbi.csv` in a cloud location (OneDrive, SharePoint or a SQL database) and configure dataset refresh in Power BI Service.

Notes & Best Practices
- Use `churn_proba` for nuanced analytics and probability thresholds to flag 'at-risk' customers.
- If you want live-scoring rather than periodic CSV export, consider exposing a REST endpoint (Flask/FastAPI) and using Power BI's `Web` data source (Web.Contents POST) to score and return results; this is advanced.
