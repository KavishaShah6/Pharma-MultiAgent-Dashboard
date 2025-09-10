import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_historical_sales(start_date, end_date, drugs, filename='data/historical_sales.csv'):
    """Generate synthetic historical sales data"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for drug in drugs:
        base_demand = np.random.randint(50, 200)
        trend = np.random.normal(0, 0.1, len(date_range))
        seasonality = np.sin(np.arange(len(date_range)) * 2 * np.pi / 30) * 20
        noise = np.random.normal(0, 10, len(date_range))
        
        for i, date in enumerate(date_range):
            quantity = max(0, int(base_demand + trend[i] * i + seasonality[i] + noise[i]))
            data.append({
                'drug': drug,
                'date': date.strftime('%Y-%m-%d'),
                'quantity_sold': quantity
            })
    
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv(filename, index=False)
    return df

def generate_machine_logs(start_date, end_date, machine_ids, filename='data/machine_logs.csv'):
    """Generate synthetic machine sensor logs"""
    timestamp_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    data = []
    for machine_id in machine_ids:
        base_temp = np.random.uniform(70, 75)
        base_vibration = np.random.uniform(0.3, 0.5)
        
        for timestamp in timestamp_range:
            # Add some realistic patterns
            temp_noise = np.random.normal(0, 2)
            vibration_noise = np.random.normal(0, 0.05)
            
            # Simulate occasional issues
            if np.random.random() < 0.02:  # 2% chance of issues
                temp_noise += np.random.uniform(5, 15)
                vibration_noise += np.random.uniform(0.1, 0.3)
                error_code = np.random.choice([1, 2, 3, 0], p=[0.3, 0.3, 0.3, 0.1])
            else:
                error_code = 0
                
            data.append({
                'machine_id': machine_id,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': max(0, base_temp + temp_noise),
                'vibration': max(0, base_vibration + vibration_noise),
                'error_code': error_code
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

def generate_inventory_data(drugs, filename='data/inventory.csv'):
    """Generate current inventory levels"""
    data = []
    for drug in drugs:
        data.append({
            'drug': drug,
            'current_stock': np.random.randint(100, 500),
            'reorder_point': np.random.randint(50, 150),
            'max_capacity': np.random.randint(1000, 2000)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

def generate_mock_data():
    """Generate all mock datasets"""
    print("Generating mock data...")
    
    # Define parameters
    drugs = ['Aspirin', 'Ibuprofen', 'Acetaminophen', 'Penicillin', 'Amoxicillin']
    machine_ids = ['M001', 'M002', 'M003', 'M004', 'M005']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate datasets
    sales_df = generate_historical_sales(start_date, end_date, drugs)
    machine_df = generate_machine_logs(start_date, end_date, machine_ids)
    inventory_df = generate_inventory_data(drugs)
    
    print(f"Generated {len(sales_df)} sales records")
    print(f"Generated {len(machine_df)} machine log records")
    print(f"Generated {len(inventory_df)} inventory records")
    
    return sales_df, machine_df, inventory_df

if __name__ == "__main__":
    generate_mock_data()