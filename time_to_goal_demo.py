# time_to_goal_demo.py
"""
Portfolio Demo: Time-to-Goal Fitness
Predicts number of days to reach a target weight using a simple ML model.
Safe, lightweight demo version of the full Flask app.
"""

from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Synthetic Training Data (demo only) 
days = np.arange(0, 30).reshape(-1, 1)
weights = 90 - 0.3 * days.flatten() + np.random.normal(0, 0.2, size=len(days))
model = LinearRegression().fit(days, weights)


@app.route("/")
def index():
    return jsonify({
        "message": "Welcome to Time-to-Goal Fitness demo!",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        },
        "note": "Demo version. Full ML app available on request."
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "LinearRegression",
        "training_points": len(days)
    })


@app.route("/predict", methods=["POST"])
def predict_days():
    """
    Example request:
      curl -X POST http://127.0.0.1:5001/predict \
           -H "Content-Type: application/json" \
           -d '{"goal_weight": 75}'
    """
    data = request.get_json(silent=True) or {}
    if "goal_weight" not in data:
        return jsonify({"error": "Missing 'goal_weight'"}), 400

    goal_weight = float(data["goal_weight"])
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    if abs(slope) < 1e-9:
        return jsonify({"error": "Model slope invalid"}), 500

    pred_days = (goal_weight - intercept) / slope
    pred_days = float(np.clip(pred_days, 0.0, 365.0))

    return jsonify({
        "goal_weight": goal_weight,
        "predicted_days": round(pred_days, 1),
        "details": {
            "intercept": round(intercept, 2),
            "slope": round(slope, 2),
            "training_points": len(days)
        },
        "note": "Demo prediction only – full ML pipeline available on request."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
