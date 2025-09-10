Pharma Multi-Agent Dashboard

An AI-driven pharmaceutical supply chain dashboard integrating multi-model demand forecasting, machine health monitoring, and MILP-based production optimization.
Built with a multi-agent orchestration architecture and an interactive Streamlit UI.

🚀 Features

📊 Interactive Dashboard with Plotly charts and expandable sections

🤖 Multi-Agent System (Forecasting, Machine Monitoring, Supply Optimization)

🔮 Demand Forecasting (ARIMA, Prophet, XGBoost, optional LSTM) with uncertainty bands

⚙️ Machine Health Monitoring with anomaly detection (Isolation Forest)

📦 Supply Recommendations using MILP optimization (PuLP)

⚡ Scenario Simulation (demand scaling, machine failures, heuristic/MILP toggle)

💬 Chatbot for natural language queries on forecasts, risks, and production

🛠️ Experiment Tracking with optional MLflow integration

🔬 Tech Stack

Frontend: Streamlit, Plotly, Custom CSS

Backend: Python, Orchestrator pattern, Pub/Sub messaging

ML Models: ARIMA, Prophet, XGBoost, IsolationForest, LSTM (legacy)

Optimization: PuLP (MILP solver)

Data Handling: Pandas, NumPy

Experiment Tracking: MLflow (optional)

🎯 Example Use Cases

Pharma companies planning production schedules under demand uncertainty

Predicting machine failures to reduce downtime

Running what-if scenarios (e.g., demand surge, machine breakdowns)

Academic/Research use in AI-driven supply chain analytics

