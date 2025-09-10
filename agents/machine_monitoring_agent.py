import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta

class MachineMonitoringAgent:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['temperature', 'vibration']
        self.thresholds = {
            'temperature': {'critical': 85, 'warning': 80},
            'vibration': {'critical': 0.8, 'warning': 0.6}
        }
        
    def preprocess_data(self, data):
        """Preprocess machine sensor data"""
        # Create features
        features = data[self.feature_columns].copy()
        
        # Add derived features
        features['temp_vibration_ratio'] = features['temperature'] / (features['vibration'] + 1e-6)
        features['temp_rolling_avg'] = features['temperature'].rolling(window=5).mean().fillna(features['temperature'])
        features['vibration_rolling_std'] = features['vibration'].rolling(window=5).std().fillna(0)
        
        return features
    
    def train(self, historical_data):
        """Train anomaly detection model"""
        print("Training Machine Monitoring Agent...")
        
        # Filter out error conditions for normal behavior training
        normal_data = historical_data[historical_data['error_code'] == 0]
        
        if len(normal_data) < 50:
            raise ValueError("Insufficient normal operation data for training")
        
        features = self.preprocess_data(normal_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train anomaly detector
        self.anomaly_detector.fit(scaled_features)
        
        self.is_trained = True
        self.save_model()
        
    def predict(self, current_data):
        """Predict machine health and potential failures"""
        if not self.is_trained:
            self.load_model()
            
        results = {}
        
        for machine_id in current_data['machine_id'].unique():
            machine_data = current_data[current_data['machine_id'] == machine_id].copy()
            
            if len(machine_data) == 0:
                continue
                
            features = self.preprocess_data(machine_data)
            scaled_features = self.scaler.transform(features)
            
            # Anomaly detection
            anomaly_scores = self.anomaly_detector.decision_function(scaled_features)
            is_anomaly = self.anomaly_detector.predict(scaled_features)
            
            # Calculate health score (0-100)
            health_score = np.clip(50 + np.mean(anomaly_scores) * 50, 0, 100)
            
            # Rule-based alerts
            latest_data = machine_data.iloc[-1]
            alerts = self.check_thresholds(latest_data)
            
            # Predict failure probability
            failure_prob = self.calculate_failure_probability(machine_data, anomaly_scores)
            
            results[machine_id] = {
                'health_score': health_score,
                'anomaly_detected': any(is_anomaly == -1),
                'alerts': alerts,
                'failure_probability': failure_prob,
                'next_maintenance': self.suggest_maintenance(health_score, failure_prob)
            }
            
        return results, self.get_reasoning(results)
    
    def check_thresholds(self, data):
        """Check sensor thresholds"""
        alerts = []
        
        for sensor, thresholds in self.thresholds.items():
            value = data[sensor]
            if value > thresholds['critical']:
                alerts.append(f"{sensor.upper()} CRITICAL: {value:.2f}")
            elif value > thresholds['warning']:
                alerts.append(f"{sensor.upper()} WARNING: {value:.2f}")
                
        if data['error_code'] != 0:
            alerts.append(f"ERROR CODE: {data['error_code']}")
            
        return alerts
    
    def calculate_failure_probability(self, machine_data, anomaly_scores):
        """Calculate probability of failure in next 7 days"""
        # Simple heuristic based on recent anomalies and trends
        recent_anomalies = np.sum(anomaly_scores < -0.1) / len(anomaly_scores)
        error_frequency = np.sum(machine_data['error_code'] != 0) / len(machine_data)
        
        # Temperature and vibration trends
        temp_trend = np.polyfit(range(len(machine_data)), machine_data['temperature'], 1)[0]
        vibration_trend = np.polyfit(range(len(machine_data)), machine_data['vibration'], 1)[0]
        
        failure_prob = min(0.9, recent_anomalies * 0.4 + error_frequency * 0.4 + 
                          max(0, temp_trend) * 0.1 + max(0, vibration_trend) * 0.1)
        
        return failure_prob
    
    def suggest_maintenance(self, health_score, failure_prob):
        """Suggest maintenance schedule"""
        if failure_prob > 0.7 or health_score < 30:
            return "IMMEDIATE"
        elif failure_prob > 0.4 or health_score < 60:
            return "WITHIN_WEEK"
        elif failure_prob > 0.2 or health_score < 80:
            return "WITHIN_MONTH"
        else:
            return "ROUTINE"
    
    def get_reasoning(self, results):
        """Provide reasoning for machine health assessments"""
        reasoning = []
        
        for machine_id, result in results.items():
            machine_reasoning = f"Machine {machine_id}: "
            machine_reasoning += f"Health Score: {result['health_score']:.1f}/100. "
            
            if result['anomaly_detected']:
                machine_reasoning += "Anomalous behavior detected. "
                
            if result['alerts']:
                machine_reasoning += f"Alerts: {', '.join(result['alerts'])}. "
                
            machine_reasoning += f"Failure probability: {result['failure_probability']:.2f}. "
            machine_reasoning += f"Maintenance: {result['next_maintenance']}"
            
            reasoning.append(machine_reasoning)
            
        return "\n".join(reasoning)
    
    def save_model(self):
        """Save trained model"""
        os.makedirs('models/saved_models', exist_ok=True)
        joblib.dump(self.anomaly_detector, 'models/saved_models/anomaly_detector.pkl')
        joblib.dump(self.scaler, 'models/saved_models/machine_scaler.pkl')
        
    def load_model(self):
        """Load trained model"""
        try:
            self.anomaly_detector = joblib.load('models/saved_models/anomaly_detector.pkl')
            self.scaler = joblib.load('models/saved_models/machine_scaler.pkl')
            self.is_trained = True
        except:
            print("No saved model found. Please train first.")