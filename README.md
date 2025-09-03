# Time-to-Goal Fitness

Predicts days to reach target weight. Flask + SQLite. ML: Linear / Random Forest / XGBoost. Synthetic data + charts.

This is a small, safe demo for portfolio use. A deeper technical write-up and the full ML pipeline are available on request.

## Review in 60 seconds
- End-to-end DS: data → models → API → UI
- Clean structure: Flask blueprints, SQLAlchemy models, ML in `services/`
- Reproducible: synthetic data, 3-line run
- Practical ML: baseline + ensembles, simple time feature, clear outputs
- Synthetic holdout: RF/XGB R² ≈ 0.8; RMSE ≈ 3–3.5 days
 
## Quickstart
```bash
pip install -r requirements.txt
python generate_data.py
python app.py   # then open http://127.0.0.1:5001/

Access
Open http://127.0.0.1:5001/
Quick links: /, /weight_tracker, /weight_progress_chart, /population_charts, /plan, /ai_plan, /plans, /project_links

API examples

# add a goal
curl -X POST http://127.0.0.1:5001/goals \
  -H "Content-Type: application/json" \
  -d '{"content":"Learn Flask","completed":false}'

# add a weight entry (optional goal_weight for prediction)
curl -X POST http://127.0.0.1:5001/weight \
  -H "Content-Type: application/json" \
  -d '{"weight":78.5,"goal_weight":70}'

# get ensemble predictions for days to goal
curl -X POST http://127.0.0.1:5001/predict_days_to_goal \
  -H "Content-Type: application/json" \
  -d '{"goal_weight":70}'

Tech:
Backend: Flask, SQLAlchemy (SQLite)
Data/ML: NumPy, Pandas, scikit-learn, XGBoost 
Viz: Chart.js (frontend), Matplotlib/Seaborn 

Notes:
Educational/demo use; not medical advice.
Usage rights: No license granted. All rights reserved. Deeper technical notes available on request.
