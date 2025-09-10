import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from models.forecasting import arima_forecast, rolling_cv_splits, rmse, mae, mape
from utils.experiment import start_run, log_params, log_metrics, log_artifact

try:
    from prophet import Prophet  # type: ignore
    _PROPHET_AVAILABLE = True
except Exception:
    _PROPHET_AVAILABLE = False

try:
    import xgboost as xgb  # type: ignore
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

class DemandForecastAgent:
    def __init__(self, lookback_window=30, forecast_horizon=7):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.drug_scalers = {}
        self.artifacts_root = 'models/saved_models/demand'
        
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        # Group by drug and create sequences
        drug_data = {}
        for drug in data['drug'].unique():
            drug_df = data[data['drug'] == drug].sort_values('date')
            drug_df['date'] = pd.to_datetime(drug_df['date'])
            drug_df = drug_df.set_index('date').resample('D').sum().fillna(0)
            drug_data[drug] = drug_df['quantity_sold'].values
            
        return drug_data
    
    def create_sequences(self, data, drug):
        """Create sequences for LSTM"""
        if drug not in self.drug_scalers:
            self.drug_scalers[drug] = MinMaxScaler()
            scaled_data = self.drug_scalers[drug].fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = self.drug_scalers[drug].transform(data.reshape(-1, 1))
            
        X, y = [], []
        for i in range(self.lookback_window, len(scaled_data) - self.forecast_horizon + 1):
            X.append(scaled_data[i-self.lookback_window:i])
            y.append(scaled_data[i:i+self.forecast_horizon])
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_window, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.forecast_horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, historical_data):
        """Train and select best model per drug using rolling CV and save artifacts"""
        print("Training Demand Forecast Agent (multi-model)...")
        os.makedirs(self.artifacts_root, exist_ok=True)

        drug_series_map = self.prepare_data(historical_data)

        for drug, series in drug_series_map.items():
            if len(series) < max(60, self.lookback_window + self.forecast_horizon):
                continue
            best = self._train_and_select_best_model_for_drug(drug, series)
            if best:
                self._save_best_model_metadata(drug, best)
        # Mark as trained so predict can use saved models
        self.is_trained = True
        # Also save legacy scalers for LSTM compatibility if needed
        self.save_model()

    def _train_and_select_best_model_for_drug(self, drug: str, series: np.ndarray) -> Optional[Dict[str, Any]]:
        horizon = self.forecast_horizon
        window = max(self.lookback_window * 2, 60)
        splits = rolling_cv_splits(len(series), window=window, horizon=horizon, step=horizon)
        candidates: Dict[str, Dict[str, Any]] = {}

        with start_run(run_name=f"Demand-{drug}", tags={"agent": "DemandForecastAgent"}):
            # ARIMA
            arima_metrics = self._evaluate_candidate(series, splits, model_type='arima', params={})
            candidates['arima'] = {"metrics": arima_metrics, "params": {}}
            log_params({"drug": drug, "model": "arima"})
            log_metrics({f"cv_{k}": v for k, v in arima_metrics.items()})

            # Prophet (if available)
            if _PROPHET_AVAILABLE:
                prophet_metrics = self._evaluate_candidate(series, splits, model_type='prophet', params={})
                candidates['prophet'] = {"metrics": prophet_metrics, "params": {}}
                log_params({"drug": drug, "model": "prophet"})
                log_metrics({f"cv_{k}": v for k, v in prophet_metrics.items()})

            # XGBoost (if available)
            if _XGB_AVAILABLE:
                xgb_params = {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05}
                xgb_metrics = self._evaluate_candidate(series, splits, model_type='xgboost', params=xgb_params)
                candidates['xgboost'] = {"metrics": xgb_metrics, "params": xgb_params}
                log_params({"drug": drug, "model": "xgboost", **xgb_params})
                log_metrics({f"cv_{k}": v for k, v in xgb_metrics.items()})

        # Select best by RMSE
        best_model_type = min(candidates.keys(), key=lambda k: candidates[k]["metrics"]["rmse"]) if candidates else None
        if not best_model_type:
            return None

        # Fit best model on full series and save
        artifact_dir = os.path.join(self.artifacts_root, drug)
        os.makedirs(artifact_dir, exist_ok=True)
        best_info = {"model_type": best_model_type, "params": candidates[best_model_type]["params"], "trained_at": datetime.now().isoformat()}

        if best_model_type == 'arima':
            # No need to fit resuable object beyond series; we will re-fit quickly if needed; still store last residual std
            fc = arima_forecast(series, horizon=self.forecast_horizon)
            res_std = float(np.std(series[-window:] - np.roll(series, 1)[-window:])) if len(series) > window + 1 else float(np.std(series))
            best_info["residual_std"] = res_std
        elif best_model_type == 'prophet' and _PROPHET_AVAILABLE:
            df = pd.DataFrame({"ds": pd.date_range(end=pd.Timestamp.today(), periods=len(series), freq='D'), "y": series})
            m = Prophet(interval_width=0.95)
            m.fit(df)
            joblib.dump(m, os.path.join(artifact_dir, 'model.pkl'))
        elif best_model_type == 'xgboost' and _XGB_AVAILABLE:
            X, y = self._make_xgb_supervised(series)
            dtrain = xgb.DMatrix(X, label=y)
            params = {"max_depth": best_info["params"].get("max_depth", 3), "eta": best_info["params"].get("learning_rate", 0.05), "objective": "reg:squarederror"}
            bst = xgb.train(params, dtrain, num_boost_round=best_info["params"].get("n_estimators", 200))
            joblib.dump(bst, os.path.join(artifact_dir, 'model.pkl'))
        return best_info

    def _evaluate_candidate(self, series: np.ndarray, splits, model_type: str, params: Dict[str, Any]) -> Dict[str, float]:
        results = []
        for tr, te in splits:
            y_train = series[tr]
            y_test = series[te]
            if model_type == 'arima':
                fc = arima_forecast(y_train, horizon=len(y_test))
                y_pred = np.asarray(fc["forecast"])[: len(y_test)]
            elif model_type == 'prophet' and _PROPHET_AVAILABLE:
                df = pd.DataFrame({"ds": pd.date_range(end=pd.Timestamp.today(), periods=len(y_train), freq='D'), "y": y_train})
                m = Prophet(interval_width=0.95)
                m.fit(df)
                future = pd.DataFrame({"ds": pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=len(y_test), freq='D')})
                fcst = m.predict(future)
                y_pred = fcst['yhat'].to_numpy()
            elif model_type == 'xgboost' and _XGB_AVAILABLE:
                Xtr, ytr = self._make_xgb_supervised(y_train)
                if len(ytr) == 0:
                    continue
                dtrain = xgb.DMatrix(Xtr, label=ytr)
                xgb_params = {"max_depth": params.get("max_depth", 3), "eta": params.get("learning_rate", 0.05), "objective": "reg:squarederror"}
                bst = xgb.train(xgb_params, dtrain, num_boost_round=params.get("n_estimators", 200))
                # roll-forward predict horizon
                y_pred = self._xgb_roll_predict(bst, y_train, horizon=len(y_test))
            else:
                # Unknown model
                continue
            results.append({
                "rmse": rmse(y_test, y_pred),
                "mae": mae(y_test, y_pred),
                "mape": mape(y_test, y_pred)
            })
        if not results:
            return {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf')}
        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
        return avg

    def _make_xgb_supervised(self, series: np.ndarray, max_lag: int = 14) -> (np.ndarray, np.ndarray):
        y = np.asarray(series, dtype=float)
        rows = []
        for t in range(max_lag, len(y)):
            feats = []
            for lag in range(1, max_lag + 1):
                feats.append(y[t - lag])
            window = y[t - 7:t]
            feats.extend([np.mean(window), np.std(window)])
            rows.append((feats, y[t]))
        if not rows:
            return np.empty((0, max_lag + 2)), np.empty((0,))
        X = np.array([r[0] for r in rows])
        y_out = np.array([r[1] for r in rows])
        return X, y_out

    def _xgb_roll_predict(self, booster, series: np.ndarray, horizon: int, max_lag: int = 14) -> np.ndarray:
        y_hist = list(series.astype(float))
        preds = []
        for _ in range(horizon):
            if len(y_hist) < max_lag:
                preds.append(float(np.mean(y_hist[-7:])))
                y_hist.append(preds[-1])
                continue
            feats = []
            for lag in range(1, max_lag + 1):
                feats.append(y_hist[-lag])
            window = y_hist[-7:]
            feats.extend([float(np.mean(window)), float(np.std(window))])
            dmat = xgb.DMatrix(np.array(feats, dtype=float).reshape(1, -1))
            pred = float(booster.predict(dmat)[0])
            preds.append(pred)
            y_hist.append(pred)
        return np.array(preds)
        
    def predict(self, recent_data, drug):
        """Generate demand forecast"""
        # Use best saved model per drug when available; otherwise ARIMA fallback
        best_meta = self._load_best_model_metadata(drug)
        hist_series = np.asarray(recent_data, dtype=float)
        dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(self.forecast_horizon)]

        if best_meta:
            model_type = best_meta.get("model_type")
            artifact_dir = os.path.join(self.artifacts_root, drug)
            if model_type == 'prophet' and _PROPHET_AVAILABLE:
                try:
                    m = joblib.load(os.path.join(artifact_dir, 'model.pkl'))
                    start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
                    future = pd.DataFrame({"ds": pd.date_range(start=start, periods=self.forecast_horizon, freq='D')})
                    fcst = m.predict(future)
                    forecast = fcst['yhat'].to_numpy()
                    lower = fcst['yhat_lower'].to_numpy()
                    upper = fcst['yhat_upper'].to_numpy()
                    return {
                        'forecast': forecast,
                        'confidence_lower': lower,
                        'confidence_upper': upper,
                        'drug': drug,
                        'forecast_dates': dates
                    }, self.get_reasoning(drug, forecast)
                except Exception:
                    pass
            if model_type == 'xgboost' and _XGB_AVAILABLE:
                try:
                    bst = joblib.load(os.path.join(artifact_dir, 'model.pkl'))
                    forecast = self._xgb_roll_predict(bst, hist_series, self.forecast_horizon)
                    std = float(np.std(hist_series[-30:])) if len(hist_series) >= 30 else float(np.std(hist_series))
                    lower = forecast - 1.96 * std
                    upper = forecast + 1.96 * std
                    return {
                        'forecast': forecast,
                        'confidence_lower': lower,
                        'confidence_upper': upper,
                        'drug': drug,
                        'forecast_dates': dates
                    }, self.get_reasoning(drug, forecast)
                except Exception:
                    pass
            if model_type == 'arima':
                fc = arima_forecast(hist_series, horizon=self.forecast_horizon)
                return {
                    'forecast': fc['forecast'],
                    'confidence_lower': fc['confidence_lower'],
                    'confidence_upper': fc['confidence_upper'],
                    'drug': drug,
                    'forecast_dates': dates
                }, self.get_reasoning(drug, fc['forecast'])

        # Fallback to ARIMA on recent history
        if len(hist_series) < max(14, self.lookback_window // 2):
            return None, "Insufficient data for forecasting"
        arima_out = arima_forecast(hist_series, horizon=self.forecast_horizon)
        return {
            'forecast': arima_out['forecast'],
            'confidence_lower': arima_out['confidence_lower'],
            'confidence_upper': arima_out['confidence_upper'],
            'drug': drug,
            'forecast_dates': dates
        }, self.get_reasoning(drug, arima_out['forecast'])
    
    def get_reasoning(self, drug, forecast):
        """Provide reasoning for the forecast"""
        avg_demand = np.mean(forecast)
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"
        
        reasoning = f"Forecast for {drug}: Average predicted demand is {avg_demand:.1f} units. "
        reasoning += f"Trend appears to be {trend} over the next {self.forecast_horizon} days. "
        reasoning += "Confidence intervals account for historical variability."
        
        return reasoning
    
    def save_model(self):
        """Save trained model"""
        os.makedirs('models/saved_models', exist_ok=True)
        if self.model:
            self.model.save('models/saved_models/demand_forecast_model.h5')
        joblib.dump(self.drug_scalers, 'models/saved_models/drug_scalers.pkl')
        
    def load_model(self):
        """Load trained model"""
        try:
            self.model = load_model('models/saved_models/demand_forecast_model.h5')
            self.drug_scalers = joblib.load('models/saved_models/drug_scalers.pkl')
            self.is_trained = True
        except:
            print("No saved model found. Please train first.")

    def _save_best_model_metadata(self, drug: str, meta: Dict[str, Any]):
        artifact_dir = os.path.join(self.artifacts_root, drug)
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, 'best_model.json'), 'w') as f:
            json.dump(meta, f)
        # try log
        try:
            log_artifact(os.path.join(artifact_dir, 'best_model.json'))
        except Exception:
            pass

    def _load_best_model_metadata(self, drug: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.artifacts_root, drug, 'best_model.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None