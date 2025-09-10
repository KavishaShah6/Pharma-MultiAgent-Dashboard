"""
Main orchestration script for the Multi-Agent Pharmaceutical Supply Chain System
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from agents import DemandForecastAgent, MachineMonitoringAgent, SupplyRecommendationAgent
from utils import generate_mock_data, MessageBroker

class MultiAgentOrchestrator:
    def __init__(self):
        self.demand_agent = DemandForecastAgent()
        self.machine_agent = MachineMonitoringAgent()
        self.supply_agent = SupplyRecommendationAgent()
        self.message_broker = MessageBroker()
        self.initialized = False
        
        # Subscribe to events
        self.setup_subscriptions()
        
    def setup_subscriptions(self):
        """Setup inter-agent communication"""
        self.message_broker.subscribe('machine_alert', self.handle_machine_alert)
        self.message_broker.subscribe('demand_change', self.handle_demand_change)
        
    def handle_machine_alert(self, message):
        """Handle machine alerts and trigger supply recommendations"""
        print(f"Alert received: {message['message']}")
        
    def handle_demand_change(self, message):
        """Handle demand forecast changes"""
        print(f"Demand change: {message['message']}")
        
    def initialize_system(self):
        """Initialize and train all agents"""
        if self.initialized:
            print("Multi-Agent System already initialized. Skipping re-initialization.")
            return
        print("Initializing Multi-Agent System...")
        
        # Generate mock data if it doesn't exist
        if not os.path.exists('data/historical_sales.csv'):
            generate_mock_data()
            
        # Load data
        sales_data = pd.read_csv('data/historical_sales.csv')
        machine_data = pd.read_csv('data/machine_logs.csv')
        inventory_data = pd.read_csv('data/inventory.csv')
        
        print("Loading agents or training if needed...")
        
        # Demand forecasting agent: load if saved, else train
        demand_model_path = 'models/saved_models/demand_forecast_model.h5'
        demand_scalers_path = 'models/saved_models/drug_scalers.pkl'
        try:
            if os.path.exists(demand_model_path) and os.path.exists(demand_scalers_path):
                self.demand_agent.load_model()
                print("Demand agent: loaded saved model.")
            else:
                raise FileNotFoundError
        except Exception:
            try:
                print("Demand agent: training model (no saved model found)...")
                self.demand_agent.train(sales_data)
            except Exception as e:
                print(f"Error training demand agent: {e}")
        
        # Machine monitoring agent: load if saved, else train
        anomaly_path = 'models/saved_models/anomaly_detector.pkl'
        scaler_path = 'models/saved_models/machine_scaler.pkl'
        try:
            if os.path.exists(anomaly_path) and os.path.exists(scaler_path):
                self.machine_agent.load_model()
                print("Machine agent: loaded saved model.")
            else:
                raise FileNotFoundError
        except Exception:
            try:
                print("Machine agent: training model (no saved model found)...")
                self.machine_agent.train(machine_data)
            except Exception as e:
                print(f"Error training machine agent: {e}")
            
        print("System initialization complete!")
        self.initialized = True
        
    def run_analysis(self, demand_scale: float = 1.0, failed_machines=None, use_milp: bool = True):
        """Run complete multi-agent analysis"""
        print("\nRunning Multi-Agent Analysis...")
        
        # Load current data
        sales_data = pd.read_csv('data/historical_sales.csv')
        machine_data = pd.read_csv('data/machine_logs.csv')
        inventory_data = pd.read_csv('data/inventory.csv')
        
        # Convert inventory to dict
        inventory_dict = dict(zip(inventory_data['drug'], inventory_data['current_stock']))
        
        # Get recent machine data (last 24 hours)
        machine_data['timestamp'] = pd.to_datetime(machine_data['timestamp'])
        recent_machine_data = machine_data[
            machine_data['timestamp'] >= (datetime.now() - timedelta(days=1))
        ]
        
        # 1. Machine Health Assessment
        print("\n1. Analyzing Machine Health...")
        machine_status, machine_reasoning = self.machine_agent.predict(recent_machine_data)
        
        # Publish machine alerts
        for machine_id, status in machine_status.items():
            if status['next_maintenance'] == 'IMMEDIATE':
                self.message_broker.publish(
                    'machine_alert',
                    f"Machine {machine_id} requires immediate maintenance!",
                    'MachineMonitoringAgent'
                )
        
        # 2. Demand Forecasting
        print("\n2. Generating Demand Forecasts...")
        demand_forecasts = {}
        
        for drug in inventory_dict.keys():
            drug_data = sales_data[sales_data['drug'] == drug].sort_values('date')
            if len(drug_data) >= 30:  # Need enough history
                recent_sales = drug_data.tail(30)['quantity_sold'].values
                forecast, reasoning = self.demand_agent.predict(recent_sales, drug)
                if forecast:
                    demand_forecasts[drug] = forecast
                    print(f"  {reasoning}")
                    
                    # Check for significant demand changes
                    avg_historical = np.mean(recent_sales)
                    avg_forecast = np.mean(forecast['forecast'])
                    if abs(avg_forecast - avg_historical) / avg_historical > 0.2:
                        self.message_broker.publish(
                            'demand_change',
                            f"Significant demand change detected for {drug}: {avg_forecast:.1f} vs {avg_historical:.1f}",
                            'DemandForecastAgent'
                        )
        
        # 3. Supply Chain Optimization
        print("\n3. Generating Supply Recommendations...")
        supply_recommendations, supply_reasoning = self.supply_agent.generate_recommendations(
            demand_forecasts, machine_status, inventory_dict,
            demand_scale=demand_scale, failed_machines=failed_machines or [], use_milp=use_milp
        )
        
        # Display results
        self.display_results(machine_status, machine_reasoning, 
                           demand_forecasts, supply_recommendations, supply_reasoning)
        
        return {
            'machine_status': machine_status,
            'demand_forecasts': demand_forecasts,
            'supply_recommendations': supply_recommendations,
            'messages': self.message_broker.get_messages()
        }
    
    def display_results(self, machine_status, machine_reasoning, 
                       demand_forecasts, supply_recommendations, supply_reasoning):
        """Display analysis results"""
        print("\n" + "="*60)
        print("MULTI-AGENT ANALYSIS RESULTS")
        print("="*60)
        
        # Machine Status Summary
        print("\nMACHINE HEALTH SUMMARY:")
        print("-" * 30)
        for machine_id, status in machine_status.items():
            print(f"Machine {machine_id}:")
            print(f"  Health Score: {status['health_score']:.1f}/100")
            print(f"  Maintenance: {status['next_maintenance']}")
            print(f"  Failure Risk: {status['failure_probability']:.2%}")
            if status['alerts']:
                print(f"  Alerts: {', '.join(status['alerts'])}")
            print()
        
        # Demand Forecast Summary
        print("DEMAND FORECAST SUMMARY:")
        print("-" * 30)
        for drug, forecast in demand_forecasts.items():
            if forecast:
                total_demand = np.sum(forecast['forecast'])
                print(f"{drug}: {total_demand:.0f} units (7-day forecast)")
        
        # Supply Recommendations Summary
        print("\nSUPPLY RECOMMENDATIONS:")
        print("-" * 30)
        high_priority_drugs = []
        for drug, rec in supply_recommendations.items():
            print(f"{drug}:")
            print(f"  Current Stock: {rec['current_stock']:.0f} units")
            print(f"  Recommended Production: {rec['recommended_production']:.0f} units")
            print(f"  Priority: {rec['priority']}")
            if rec['expected_stockout_risk'] > 0:
                print(f"  Stockout Risk: {rec['expected_stockout_risk']:.1%}")
            
            if rec['priority'] == 'HIGH':
                high_priority_drugs.append(drug)
            print()
        
        # Critical Alerts
        print("CRITICAL ALERTS:")
        print("-" * 30)
        critical_alerts = []
        
        # Machine alerts
        for machine_id, status in machine_status.items():
            if status['next_maintenance'] == 'IMMEDIATE':
                critical_alerts.append(f"🔴 Machine {machine_id} needs IMMEDIATE maintenance")
            elif status['failure_probability'] > 0.5:
                critical_alerts.append(f"🟡 Machine {machine_id} has high failure risk ({status['failure_probability']:.1%})")
        
        # Supply alerts
        for drug in high_priority_drugs:
            critical_alerts.append(f"🔴 HIGH PRIORITY production needed for {drug}")
        
        if critical_alerts:
            for alert in critical_alerts:
                print(f"  {alert}")
        else:
            print("  ✅ No critical alerts")
        
        print("\n" + "="*60)

def main():
    """Main execution function"""
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Initialize system (train agents)
    orchestrator.initialize_system()
    
    # Run analysis
    results = orchestrator.run_analysis()
    
    # Display message history
    messages = orchestrator.message_broker.get_messages()
    if messages:
        print("\nINTER-AGENT MESSAGES:")
        print("-" * 30)
        for msg in messages:
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
            print(f"[{timestamp}] {msg['sender']}: {msg['message']}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Multi-Agent Analysis Complete!")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()