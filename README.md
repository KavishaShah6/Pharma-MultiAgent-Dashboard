Pharma Multi-Agent Dashboard

An AI-driven pharmaceutical supply chain dashboard integrating multi-model demand forecasting, machine health monitoring, and MILP-based production optimization.
Built with a multi-agent orchestration architecture and an interactive Streamlit UI.

Features

Interactive dashboard with Plotly charts and expandable sections

Multi-agent system (Forecasting, Machine Monitoring, Supply Optimization)

Demand forecasting (ARIMA, Prophet, XGBoost, optional LSTM) with uncertainty bands

Machine health monitoring with anomaly detection (Isolation Forest)

Supply recommendations using MILP optimization (PuLP)

Scenario simulation (demand scaling, machine failures, heuristic/MILP toggle)

Chatbot for natural language queries on forecasts, risks, and production

Experiment tracking with optional MLflow integration

Tech Stack

Frontend: Streamlit, Plotly, Custom CSS

Backend: Python, Orchestrator pattern, Pub/Sub messaging

Machine Learning Models: ARIMA, Prophet, XGBoost, IsolationForest, LSTM (legacy)

Optimization: PuLP (MILP solver)

Data Handling: Pandas, NumPy

Experiment Tracking: MLflow (optional)

Example Use Cases

Supporting pharmaceutical companies in planning production schedules under demand uncertainty

Predicting machine failures to reduce downtime and improve reliability

Running scenario-based analyses (e.g., demand surge, machine breakdowns)

Academic and research applications in AI-driven supply chain analytics
