"""
Streamlit Dashboard for Multi-Agent Pharmaceutical Supply Chain System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MultiAgentOrchestrator
from utils import generate_mock_data

# Simple chat responder using latest results
def answer_chat_query(query, results):
    query_l = query.lower()
    if not results:
        return "Please run analysis first."
    # Forecast question
    if "forecast" in query_l or "predict" in query_l:
        # try to extract drug name
        for drug, fc in results.get('demand_forecasts', {}).items():
            if drug.lower() in query_l:
                if not fc:
                    return f"No forecast available for {drug}."
                total = float(np.sum(fc['forecast']))
                return f"7-day forecast for {drug} is ~{total:.0f} units. Daily mean {np.mean(fc['forecast']):.1f}."
        return "Specify a drug name to get its forecast."
    # Machine risk question
    if "machine" in query_l and ("risk" in query_l or "failure" in query_l):
        risky = [
            (m, s['failure_probability'])
            for m, s in results.get('machine_status', {}).items()
            if s['failure_probability'] > 0.4
        ]
        if not risky:
            return "No machines with elevated risk right now."
        risky.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join([f"{m} ({p:.0%})" for m, p in risky[:5]])
        return f"High risk machines: {top}."
    # Production recommendation per drug
    if ("how much" in query_l or "production" in query_l or "recommend" in query_l) and ("drug" in query_l or "for" in query_l):
        for drug, rec in results.get('supply_recommendations', {}).items():
            if drug.lower() in query_l:
                return (
                    f"Recommended production for {drug}: {rec['recommended_production']:.0f} units "
                    f"(priority {rec['priority']}, stockout risk {rec['expected_stockout_risk']:.0%})."
                )
        return "Specify a drug name to get its production recommendation."
    # Default
    return "I can answer about forecasts, machine risks, and production recommendations."

def _safe_run_analysis(orchestrator, demand_scale, failed_machines, use_milp):
    try:
        return orchestrator.run_analysis(
            demand_scale=demand_scale,
            failed_machines=failed_machines,
            use_milp=use_milp,
        )
    except TypeError:
        # Backward-compatible fallback to old signature
        return orchestrator.run_analysis()

# Page configuration
st.set_page_config(
    page_title="Pharma Supply Chain AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #000; /* ensure readable text */
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #b71c1c; /* dark red text */
        font-weight: 600;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #e65100; /* dark orange text */
        font-weight: 600;
    }
    .alert-success {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1b5e20; /* dark green text */
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
def to_rgba(color, alpha=0.2):
            """Convert hex or rgb string to rgba string"""
            try:
                # If hex string
                rgb = px.colors.hex_to_rgb(color)
            except ValueError:
                # If already 'rgb(r,g,b)' string
                rgb = tuple(map(int, color.strip('rgb()').split(',')))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
@st.cache_data
def load_data():
    """Load and cache data"""
    try:
        sales_data = pd.read_csv('data/historical_sales.csv')
        machine_data = pd.read_csv('data/machine_logs.csv')
        inventory_data = pd.read_csv('data/inventory.csv')
        return sales_data, machine_data, inventory_data
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def initialize_orchestrator():
    """Initialize and cache the orchestrator"""
    orchestrator = MultiAgentOrchestrator()
    return orchestrator

def create_machine_health_chart(machine_status):
    """Create machine health visualization"""
    machines = list(machine_status.keys())
    health_scores = [machine_status[m]['health_score'] for m in machines]
    failure_probs = [machine_status[m]['failure_probability'] * 100 for m in machines]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Machine Health Scores', 'Failure Probability (%)'),
        vertical_spacing=0.15
    )
    
    # Health scores
    colors = ['red' if score < 50 else 'orange' if score < 75 else 'green' 
              for score in health_scores]
    
    fig.add_trace(
        go.Bar(x=machines, y=health_scores, name='Health Score', marker_color=colors),
        row=1, col=1
    )
    
    # Failure probabilities
    fig.add_trace(
        go.Bar(x=machines, y=failure_probs, name='Failure Probability', 
               marker_color='rgba(255, 0, 0, 0.6)'),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
    
    return fig

def create_demand_forecast_chart(demand_forecasts):
    """Create demand forecast visualization"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (drug, forecast) in enumerate(demand_forecasts.items()):
        if forecast is None:
            continue
            
        dates = forecast['forecast_dates']
        values = forecast['forecast']
        upper_bound = forecast['confidence_upper']
        lower_bound = forecast['confidence_lower']
        
        color = colors[i % len(colors)]
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=f'{drug} Forecast',
            line=dict(color=color, width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor=to_rgba(color, alpha=0.2),
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{drug} Confidence',
            showlegend=False
        ))
    
    fig.update_layout(
        title="7-Day Demand Forecasts",
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_supply_recommendations_chart(supply_recommendations):
    """Create supply recommendations visualization"""
    drugs = list(supply_recommendations.keys())
    current_stock = [supply_recommendations[d]['current_stock'] for d in drugs]
    forecasted_demand = [supply_recommendations[d]['forecasted_demand'] for d in drugs]
    recommended_production = [supply_recommendations[d]['recommended_production'] for d in drugs]
    safety_stock = [supply_recommendations[d]['safety_stock'] for d in drugs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Stock',
        x=drugs, y=current_stock,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Forecasted Demand',
        x=drugs, y=forecasted_demand,
        marker_color='orange'
    ))
    
    fig.add_trace(go.Bar(
        name='Recommended Production',
        x=drugs, y=recommended_production,
        marker_color='green'
    ))
    
    fig.add_trace(go.Scatter(
        name='Safety Stock Level',
        x=drugs, y=safety_stock,
        mode='markers',
        marker=dict(color='red', size=10, symbol='diamond'),
        yaxis='y'
    ))
    
    fig.update_layout(
        title="Supply Analysis and Recommendations",
        xaxis_title="Drug",
        yaxis_title="Units",
        barmode='group',
        height=400
    )
    
    return fig

def create_sales_overview_chart(sales_df):
    """Create historical sales time series per drug"""
    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    fig = px.line(
        df,
        x='date', y='quantity_sold', color='drug',
        title='Historical Sales by Drug',
    )
    fig.update_layout(height=400, hovermode='x unified')
    return fig

def create_recent_machine_timeseries_chart(machine_df):
    """Create recent machine sensor time series (last 24h)"""
    df = machine_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = datetime.now() - timedelta(days=1)
    df = df[df['timestamp'] >= cutoff]
    if df.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for machine_id, g in df.groupby('machine_id'):
        fig.add_trace(
            go.Scatter(x=g['timestamp'], y=g['temperature'], mode='lines', name=f"{machine_id} Temp"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=g['timestamp'], y=g['vibration'], mode='lines', name=f"{machine_id} Vib"),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="Temperature", secondary_y=False)
    fig.update_yaxes(title_text="Vibration", secondary_y=True)
    fig.update_layout(title="Recent Machine Sensors (24h)", height=400, hovermode='x unified')
    return fig

def display_alerts(machine_status, supply_recommendations):
    """Display system alerts"""
    st.subheader("🚨 System Alerts")
    
    critical_alerts = []
    warning_alerts = []
    
    # Machine alerts
    for machine_id, status in machine_status.items():
        if status['next_maintenance'] == 'IMMEDIATE':
            critical_alerts.append(f"Machine {machine_id} requires immediate maintenance")
        elif status['next_maintenance'] == 'WITHIN_WEEK':
            warning_alerts.append(f"Machine {machine_id} needs maintenance within a week")
        
        if status['failure_probability'] > 0.7:
            critical_alerts.append(f"Machine {machine_id} has very high failure risk ({status['failure_probability']:.1%})")
        elif status['failure_probability'] > 0.4:
            warning_alerts.append(f"Machine {machine_id} has elevated failure risk ({status['failure_probability']:.1%})")
    
    # Supply alerts
    for drug, rec in supply_recommendations.items():
        if rec['priority'] == 'HIGH':
            critical_alerts.append(f"HIGH PRIORITY: Urgent production needed for {drug}")
        if rec['expected_stockout_risk'] > 0.5:
            critical_alerts.append(f"High stockout risk for {drug} ({rec['expected_stockout_risk']:.1%})")
        elif rec['expected_stockout_risk'] > 0.2:
            warning_alerts.append(f"Moderate stockout risk for {drug} ({rec['expected_stockout_risk']:.1%})")
    
    # Display alerts
    if critical_alerts:
        for alert in critical_alerts:
            st.markdown(f'<div class="alert-critical">🔴 <strong>CRITICAL:</strong> {alert}</div>', 
                       unsafe_allow_html=True)
    
    if warning_alerts:
        for alert in warning_alerts:
            st.markdown(f'<div class="alert-warning">🟡 <strong>WARNING:</strong> {alert}</div>', 
                       unsafe_allow_html=True)
    
    if not critical_alerts and not warning_alerts:
        st.markdown('<div class="alert-success">✅ <strong>All systems operating normally</strong></div>', 
                   unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.title("💊 Multi-Agent Pharmaceutical Supply Chain System")
    st.markdown("*AI-powered demand forecasting, machine monitoring, and supply optimization*")
    
    # Sidebar
    st.sidebar.title("Control Panel")
    
    # Check if data exists
    sales_data, machine_data, inventory_data = load_data()
    
    if sales_data is None:
        st.sidebar.warning("No data found. Generate mock data first.")
        if st.sidebar.button("Generate Mock Data"):
            with st.spinner("Generating mock data..."):
                generate_mock_data()
            st.rerun()
        return
    
    # Initialize orchestrator
    ORCHESTRATOR_VERSION = "2"
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = initialize_orchestrator()
        st.session_state.orchestrator_initialized = False
        st.session_state.orchestrator_version = ORCHESTRATOR_VERSION
    # Recreate orchestrator if version changed
    if st.session_state.get('orchestrator_version') != ORCHESTRATOR_VERSION:
        st.session_state.orchestrator = MultiAgentOrchestrator()
        st.session_state.orchestrator_initialized = False
        st.session_state.orchestrator_version = ORCHESTRATOR_VERSION
    if not st.session_state.get('orchestrator_initialized'):
        with st.spinner("Initializing AI agents..."):
            st.session_state.orchestrator.initialize_system()
            st.session_state.orchestrator_initialized = True

    # Sidebar control to force re-init if needed
    if st.sidebar.checkbox("Force reinitialize agents", value=False, help="Reload and retrain if artifacts missing"):
        with st.spinner("Reinitializing agents..."):
            # Recreate a fresh orchestrator to ensure updated methods are used
            st.session_state.orchestrator = MultiAgentOrchestrator()
            st.session_state.orchestrator.initialize_system()
            st.session_state.orchestrator_initialized = True
            st.session_state.orchestrator_version = ORCHESTRATOR_VERSION
    
    # Scenario controls
    st.sidebar.markdown("**Scenario Simulation**")
    demand_scale = st.sidebar.slider("Demand scale", 0.5, 2.0, 1.0, 0.1, help="Scale all forecasts")
    failed_machine = st.sidebar.text_input("Fail machine ID (optional)", value="")
    use_milp = st.sidebar.checkbox("Use MILP optimization", value=True)

    # Control buttons
    if st.sidebar.button("🔄 Run Analysis", type="primary"):
        with st.spinner("Running multi-agent analysis..."):
            failed = [failed_machine.strip()] if failed_machine.strip() else []
            st.session_state.results = _safe_run_analysis(st.session_state.orchestrator, demand_scale, failed, use_milp)
        st.success("Analysis complete!")
    
    if st.sidebar.button("📊 Generate New Data"):
        with st.spinner("Generating new mock data..."):
            generate_mock_data()
        st.success("New data generated!")
        st.rerun()
    
    # Auto-run once so charts appear without extra clicks
    if 'results' not in st.session_state:
        with st.spinner("Running multi-agent analysis..."):
            failed = [failed_machine.strip()] if failed_machine.strip() else []
            st.session_state.results = _safe_run_analysis(st.session_state.orchestrator, demand_scale, failed, use_milp)
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            healthy_machines = sum(1 for s in results['machine_status'].values() 
                                 if s['health_score'] > 75)
            total_machines = len(results['machine_status'])
            st.metric("Healthy Machines", f"{healthy_machines}/{total_machines}")
        
        with col2:
            high_priority_drugs = sum(1 for r in results['supply_recommendations'].values() 
                                    if r['priority'] == 'HIGH')
            st.metric("High Priority Drugs", high_priority_drugs)
        
        with col3:
            avg_health = np.mean([s['health_score'] for s in results['machine_status'].values()])
            st.metric("Avg Machine Health", f"{avg_health:.1f}%")
        
        with col4:
            total_production = sum(r['recommended_production'] 
                                 for r in results['supply_recommendations'].values())
            st.metric("Total Production Rec.", f"{total_production:.0f} units")
        
        # Alerts
        display_alerts(results['machine_status'], results['supply_recommendations'])
        
        # Charts
        st.subheader("🔧 Machine Health Analysis")
        machine_chart = create_machine_health_chart(results['machine_status'])
        st.plotly_chart(machine_chart, use_container_width=True)
        
        # Machine details
        with st.expander("Machine Details"):
            for machine_id, status in results['machine_status'].items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{machine_id}**")
                    st.write(f"Health: {status['health_score']:.1f}%")
                with col2:
                    st.write(f"Failure Risk: {status['failure_probability']:.1%}")
                    st.write(f"Maintenance: {status['next_maintenance']}")
                with col3:
                    if status['alerts']:
                        st.write("**Alerts:**")
                        for alert in status['alerts']:
                            st.write(f"- {alert}")
                st.divider()
        
        # Demand forecasts
        if results['demand_forecasts']:
            st.subheader("📈 Demand Forecasts")
            forecast_chart = create_demand_forecast_chart(results['demand_forecasts'])
            st.plotly_chart(forecast_chart, use_container_width=True)

        # Historical sales overview
        st.subheader("🧾 Sales Overview")
        sales_chart = create_sales_overview_chart(sales_data)
        st.plotly_chart(sales_chart, use_container_width=True)
        
        # Supply recommendations
        st.subheader("📦 Supply Recommendations")
        supply_chart = create_supply_recommendations_chart(results['supply_recommendations'])
        st.plotly_chart(supply_chart, use_container_width=True)
        
        # Supply details
        with st.expander("Supply Details"):
            for drug, rec in results['supply_recommendations'].items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{drug}**")
                    st.write(f"Current Stock: {rec['current_stock']:.0f}")
                    st.write(f"Priority: {rec['priority']}")
                with col2:
                    st.write(f"Forecasted Demand: {rec['forecasted_demand']:.0f}")
                    st.write(f"Recommended Production: {rec['recommended_production']:.0f}")
                with col3:
                    st.write(f"Safety Stock: {rec['safety_stock']:.0f}")
                    if rec['expected_stockout_risk'] > 0:
                        st.write(f"Stockout Risk: {rec['expected_stockout_risk']:.1%}")
                
                # Machine allocation
                if rec['machine_allocation']:
                    st.write("**Machine Allocation:**")
                    for machine, allocation in rec['machine_allocation'].items():
                        st.write(f"- {machine}: {allocation:.0f} units")
                st.divider()
        
        # Inter-agent messages
        if results['messages']:
            st.subheader("🤖 Inter-Agent Communications")
            for msg in results['messages'][-10:]:  # Show last 10 messages
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
                st.write(f"**[{timestamp}] {msg['sender']}:** {msg['message']}")

        # Recent machine time series
        st.subheader("📉 Recent Machine Sensors (24h)")
        recent_machine_chart = create_recent_machine_timeseries_chart(pd.read_csv('data/machine_logs.csv'))
        st.plotly_chart(recent_machine_chart, use_container_width=True)

        # Chatbot
        st.subheader("💬 Ask the AI")
        user_q = st.text_input("Ask a question (e.g., 'What is the forecast for Aspirin next week?')", key="chat_q")
        if user_q:
            with st.spinner("Thinking..."):
                reply = answer_chat_query(user_q, results)
            st.info(reply)
    
    else:
        st.info("👆 Click 'Run Analysis' in the sidebar to start the multi-agent analysis")
    
    # Data overview
    with st.expander("📋 Data Overview"):
        st.write("**Historical Sales Data:**")
        st.dataframe(sales_data.head())
        
        st.write("**Machine Logs:**")
        st.dataframe(machine_data.head())
        
        st.write("**Current Inventory:**")
        st.dataframe(inventory_data)

if __name__ == "__main__":
    main()