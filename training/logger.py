"""
Metric logging utilities for training.

Logs scalar metrics (loss, accuracy, lr) to:
  - In-memory buffer (for live streaming to frontend/Streamlit)
  - SQLite database (persistent, queryable by PowerBI / Tableau / pandas)
  - CSV export (for BI tools that prefer flat files)
"""
