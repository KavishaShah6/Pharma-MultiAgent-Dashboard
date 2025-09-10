import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize, LpBinary, value
from datetime import datetime, timedelta

class SupplyRecommendationAgent:
    def __init__(self):
        self.safety_stock_days = 7
        self.max_production_capacity = 1000  # units per day per machine (baseline)
        self.shift_hours_per_day = 16  # 2 shifts of 8h
        self.days_planning_horizon = 7
        self.setup_time_hours = 1.0  # per machine per drug if produced
        self.big_m_per_drug = 1e6
        
    def generate_recommendations(self, demand_forecasts, machine_status, current_inventory,
                                 demand_scale: float = 1.0, failed_machines=None, use_milp: bool = True):
        """Generate supply allocation and production recommendations"""
        recommendations = {}
        reasoning = []
        if failed_machines is None:
            failed_machines = []
        
        for drug in demand_forecasts:
            forecast_data = demand_forecasts[drug]
            
            if forecast_data is None:
                continue
                
            total_demand = float(np.sum(forecast_data['forecast'])) * float(demand_scale)
            current_stock = current_inventory.get(drug, 0)
            
            # Calculate safety stock
            safety_stock = np.mean(forecast_data['forecast']) * self.safety_stock_days
            
            # Assess machine capacity
            available_capacity = self.calculate_available_capacity(machine_status, failed_machines)
            
            # Optimization
            if use_milp:
                production_plan = self.optimize_production_milp(
                    {drug: total_demand}, {drug: current_stock}, {drug: safety_stock}, available_capacity
                )
                production_plan = production_plan.get(drug, {
                    'production': 0, 'priority': 'LOW', 'stockout_risk': 0.0, 'machine_allocation': {}
                })
            else:
                production_plan = self.optimize_production(
                    drug, total_demand, current_stock, safety_stock, available_capacity
                )
            
            recommendations[drug] = {
                'current_stock': current_stock,
                'forecasted_demand': total_demand,
                'safety_stock': safety_stock,
                'recommended_production': production_plan['production'],
                'priority': production_plan['priority'],
                'expected_stockout_risk': production_plan['stockout_risk'],
                'machine_allocation': production_plan['machine_allocation']
            }
            
            reasoning.append(self.get_drug_reasoning(drug, recommendations[drug], forecast_data))
            
        return recommendations, "\n".join(reasoning)
    
    def calculate_available_capacity(self, machine_status, failed_machines=None):
        """Calculate available production capacity"""
        available_machines = []
        if failed_machines is None:
            failed_machines = []
        
        for machine_id, status in machine_status.items():
            if machine_id in failed_machines:
                # simulate failure: zero capacity
                continue
            if status['next_maintenance'] in ['ROUTINE', 'WITHIN_MONTH']:
                capacity_factor = status['health_score'] / 100
                available_machines.append({
                    'machine_id': machine_id,
                    'capacity': self.max_production_capacity * capacity_factor,
                    'reliability': 1 - status['failure_probability']
                })
            elif status['next_maintenance'] == 'WITHIN_WEEK':
                capacity_factor = status['health_score'] / 100 * 0.7  # Reduced capacity
                available_machines.append({
                    'machine_id': machine_id,
                    'capacity': self.max_production_capacity * capacity_factor,
                    'reliability': 1 - status['failure_probability']
                })
            # Machines needing immediate maintenance are excluded
                
        return available_machines
    
    def optimize_production(self, drug, demand, current_stock, safety_stock, available_capacity):
        """Optimize production using linear programming"""
        # Simple heuristic if no LP solver available
        total_needed = max(0, demand + safety_stock - current_stock)
        
        if total_needed == 0:
            return {
                'production': 0,
                'priority': 'LOW',
                'stockout_risk': 0.0,
                'machine_allocation': {}
            }
        
        # Sort machines by reliability
        available_capacity.sort(key=lambda x: x['reliability'], reverse=True)
        
        total_capacity = sum(m['capacity'] for m in available_capacity) * 7  # 7 days
        
        if total_capacity < total_needed:
            stockout_risk = 1 - (total_capacity / total_needed)
            priority = 'HIGH'
            production = total_capacity
        else:
            stockout_risk = 0.0
            if total_needed > total_capacity * 0.8:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            production = total_needed
            
        # Allocate to machines
        machine_allocation = {}
        remaining_production = production
        
        for machine in available_capacity:
            if remaining_production <= 0:
                break
                
            allocation = min(remaining_production, machine['capacity'] * 7)
            machine_allocation[machine['machine_id']] = allocation
            remaining_production -= allocation
            
        return {
            'production': production,
            'priority': priority,
            'stockout_risk': stockout_risk,
            'machine_allocation': machine_allocation
        }

    def optimize_production_milp(self, demand_by_drug, current_stock_by_drug, safety_stock_by_drug, available_capacity):
        """MILP: minimize shortages and production cost with capacity, setup-time, and shifts.

        demand_by_drug: dict drug -> total 7-day demand (scaled)
        current_stock_by_drug: dict drug -> units on hand
        safety_stock_by_drug: dict drug -> target safety stock units
        available_capacity: list of machines with fields machine_id, capacity (units/day), reliability
        """
        drugs = list(demand_by_drug.keys())
        machines = [m['machine_id'] for m in available_capacity]
        cap_per_machine = {
            m['machine_id']: m['capacity'] * self.days_planning_horizon
            for m in available_capacity
        }
        # Setup time in units: convert hours to units using daily capacity rate
        # Approximate rate per hour from daily capacity
        setup_units_per_machine = {
            mid: (cap_per_machine[mid] / (self.days_planning_horizon * self.shift_hours_per_day)) * self.setup_time_hours
            for mid in cap_per_machine
        }

        prob = LpProblem('SupplyPlanning', LpMinimize)
        # Decision variables
        x = {(mid, d): LpVariable(f"x_{mid}_{d}", lowBound=0) for mid in machines for d in drugs}
        y = {(mid, d): LpVariable(f"y_{mid}_{d}", cat=LpBinary) for mid in machines for d in drugs}
        shortage = {d: LpVariable(f"short_{d}", lowBound=0) for d in drugs}

        bigM = {d: max(self.big_m_per_drug, demand_by_drug[d] * 2.0) for d in drugs}

        # Constraints: demand satisfaction with shortage
        for d in drugs:
            needed = demand_by_drug[d] + safety_stock_by_drug[d] - current_stock_by_drug[d]
            prob += lpSum(x[(mid, d)] for mid in machines) + shortage[d] >= max(0, needed), f"demand_{d}"

        # Machine capacity with setup time overhead
        for mid in machines:
            prob += lpSum(x[(mid, d)] for d in drugs) + lpSum(setup_units_per_machine[mid] * y[(mid, d)] for d in drugs) <= cap_per_machine[mid], f"cap_{mid}"

        # Link production to binary selection
        for mid in machines:
            for d in drugs:
                prob += x[(mid, d)] <= bigM[d] * y[(mid, d)], f"link_{mid}_{d}"

        # Objective: minimize shortages and small production cost
        shortage_penalty = 1.0
        prod_cost = 0.001
        obj = shortage_penalty * lpSum(shortage[d] for d in drugs) + prod_cost * lpSum(x[(mid, d)] for mid in machines for d in drugs)
        prob += obj

        prob.solve()

        result = {}
        for d in drugs:
            prod_total = sum(value(x[(mid, d)]) for mid in machines)
            allocation = {mid: value(x[(mid, d)]) for mid in machines if value(x[(mid, d)]) and value(x[(mid, d)]) > 1e-6}
            needed = max(0, demand_by_drug[d] + safety_stock_by_drug[d] - current_stock_by_drug[d])
            stockout = max(0.0, needed - prod_total)
            risk = min(1.0, stockout / needed) if needed > 0 else 0.0
            priority = 'HIGH' if risk > 0.5 else ('MEDIUM' if risk > 0.2 else 'LOW')
            result[d] = {
                'production': prod_total,
                'priority': priority,
                'stockout_risk': risk,
                'machine_allocation': allocation
            }
        return result
    
    def get_drug_reasoning(self, drug, recommendation, forecast_data):
        """Generate reasoning for drug recommendation"""
        reasoning = f"Drug {drug}: "
        reasoning += f"Current stock: {recommendation['current_stock']:.0f} units, "
        reasoning += f"7-day demand forecast: {recommendation['forecasted_demand']:.0f} units. "
        
        if recommendation['recommended_production'] > 0:
            reasoning += f"Recommended production: {recommendation['recommended_production']:.0f} units "
            reasoning += f"(Priority: {recommendation['priority']}). "
            
            if recommendation['expected_stockout_risk'] > 0:
                reasoning += f"Stockout risk: {recommendation['expected_stockout_risk']:.1%}. "
                
            reasoning += f"Safety stock target: {recommendation['safety_stock']:.0f} units."
        else:
            reasoning += "No additional production needed - sufficient inventory."
            
        return reasoning