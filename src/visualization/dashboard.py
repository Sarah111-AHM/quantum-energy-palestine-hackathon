"""
Streamlit Dashboard Components for Gaza Strip Energy Infrastructure Planning.
Provides interactive dashboard elements for real-time visualization and control.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import time
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for Streamlit dashboard."""
    page_title: str = "GazaGrid - Quantum Energy Infrastructure Planner"
    page_icon: str = "⚡"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Theme colors
    primary_color: str = "#1e3c72"
    secondary_color: str = "#2a5298"
    success_color: str = "#10b981"
    warning_color: str = "#f59e0b"
    danger_color: str = "#ef4444"
    info_color: str = "#3b82f6"
    
    # Refresh settings
    auto_refresh: bool = False
    refresh_interval: int = 30  # seconds


class GazaDashboard:
    """
    Enhanced Streamlit dashboard for GazaGrid.
    
    Features:
    - Interactive parameter tuning
    - Real-time optimization feedback
    - Comparative analysis tools
    - Export capabilities
    - Progress tracking
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self._setup_page_config()
        self._apply_custom_css()
        
        # Initialize session state
        self._init_session_state()
        
        logger.info("Dashboard initialized")
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout=self.config.layout,
            initial_sidebar_state=self.config.initial_sidebar_state
        )
    
    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown(f"""
        <style>
            /* Main header */
            .main-header {{
                background: linear-gradient(90deg, {self.config.primary_color} 0%, {self.config.secondary_color} 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
            }}
            
            .main-header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
            }}
            
            .main-header p {{
                margin: 0.5rem 0 0;
                opacity: 0.9;
                font-size: 1.1rem;
            }}
            
            /* Quantum badge */
            .quantum-badge {{
                background-color: {self.config.secondary_color};
                color: white;
                padding: 0.25rem 1rem;
                border-radius: 20px;
                font-size: 0.8rem;
                display: inline-block;
                margin-bottom: 1rem;
            }}
            
            /* Metric cards */
            .metric-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid {self.config.primary_color};
                transition: transform 0.2s;
            }}
            
            .metric-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            
            .metric-label {{
                color: #6b7280;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: {self.config.primary_color};
                margin-top: 0.5rem;
            }}
            
            /* Risk indicators */
            .risk-low {{
                color: {self.config.success_color};
                font-weight: bold;
            }}
            
            .risk-medium {{
                color: {self.config.warning_color};
                font-weight: bold;
            }}
            
            .risk-high {{
                color: {self.config.danger_color};
                font-weight: bold;
            }}
            
            /* Progress steps */
            .step-indicator {{
                display: flex;
                justify-content: space-between;
                margin: 2rem 0;
                padding: 0 1rem;
            }}
            
            .step {{
                flex: 1;
                text-align: center;
                position: relative;
            }}
            
            .step.active .step-number {{
                background: {self.config.primary_color};
                color: white;
            }}
            
            .step-number {{
                width: 40px;
                height: 40px;
                background: #e5e7eb;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.5rem;
                font-weight: bold;
                color: #4b5563;
                transition: all 0.3s;
            }}
            
            .step-label {{
                font-size: 0.9rem;
                color: #6b7280;
            }}
            
            .step.active .step-label {{
                color: {self.config.primary_color};
                font-weight: 500;
            }}
            
            /* Progress bar */
            .stProgress > div > div > div > div {{
                background-color: {self.config.primary_color};
            }}
            
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(90deg, {self.config.primary_color} 0%, {self.config.secondary_color} 100%);
                color: white;
                border: none;
                padding: 0.75rem 2rem;
                font-weight: 500;
                border-radius: 8px;
                transition: all 0.3s;
                width: 100%;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(30,60,114,0.3);
            }}
            
            /* Info boxes */
            .info-box {{
                background-color: #f0f9ff;
                border-left: 4px solid {self.config.info_color};
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                margin: 1rem 0;
            }}
            
            .warning-box {{
                background-color: #fffbeb;
                border-left: 4px solid {self.config.warning_color};
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                margin: 1rem 0;
            }}
            
            .success-box {{
                background-color: #f0fdf4;
                border-left: 4px solid {self.config.success_color};
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                margin: 1rem 0;
            }}
            
            /* Tables */
            .data-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            .data-table th {{
                background-color: {self.config.primary_color};
                color: white;
                padding: 0.75rem;
                text-align: left;
            }}
            
            .data-table td {{
                padding: 0.75rem;
                border-bottom: 1px solid #e5e7eb;
            }}
            
            .data-table tr:hover {{
                background-color: #f9fafb;
            }}
            
            /* Footer */
            .footer {{
                text-align: center;
                padding: 2rem;
                color: #9ca3af;
                font-size: 0.9rem;
                border-top: 1px solid #e5e7eb;
                margin-top: 2rem;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'step' not in st.session_state:
            st.session_state.step = 1
        if 'optimized' not in st.session_state:
            st.session_state.optimized = False
        if 'selected_sites' not in st.session_state:
            st.session_state.selected_sites = []
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = {}
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown(f"""
        <div class="main-header">
            <div class="quantum-badge">⚡ QUANTUM-ENHANCED DECISION SYSTEM</div>
            <h1>{self.config.page_title}</h1>
            <p>AI + Quantum Computing for Critical Infrastructure Planning in Gaza Strip</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_step_indicator(self, current_step: int):
        """
        Render step progress indicator.
        
        Args:
            current_step: Current step (1-3)
        """
        steps = [
            ("1", "Data Analysis"),
            ("2", "Quantum Optimization"),
            ("3", "Results & Deployment")
        ]
        
        html = '<div class="step-indicator">'
        for i, (num, label) in enumerate(steps, 1):
            active_class = 'active' if i == current_step else ''
            html += f'''
            <div class="step {active_class}">
                <div class="step-number">{num}</div>
                <div class="step-label">{label}</div>
            </div>
            '''
        html += '</div>'
        
        st.markdown(html, unsafe_allow_html=True)
        st.session_state.step = current_step
    
    def render_sidebar(
        self,
        default_n_sites: int = 5,
        default_layers: int = 2
    ) -> Dict[str, Any]:
        """
        Render sidebar with controls.
        
        Args:
            default_n_sites: Default number of sites to select
            default_layers: Default QAOA layers
            
        Returns:
            Dictionary with parameter values
        """
        with st.sidebar:
            st.markdown("## ⚙️ Control Panel")
            
            # Optimization parameters
            with st.expander("🎯 Optimization Targets", expanded=True):
                n_sites = st.slider(
                    "Number of sites to select",
                    min_value=3,
                    max_value=15,
                    value=default_n_sites,
                    help="Target number of optimal sites"
                )
                
                quantum_layers = st.select_slider(
                    "QAOA Circuit Depth",
                    options=[1, 2, 3, 4],
                    value=default_layers,
                    help="Number of QAOA layers (higher = more accurate but slower)"
                )
                
                use_quantum = st.checkbox(
                    "Enable Quantum Enhancement",
                    value=True,
                    help="Use QAOA instead of classical fallback"
                )
            
            # Weighting factors
            with st.expander("⚖️ Weighting Factors"):
                col1, col2 = st.columns(2)
                
                with col1:
                    w_solar = st.slider(
                        "☀️ Solar",
                        0.0, 1.0, 0.4, 0.05,
                        help="Importance of solar potential"
                    )
                    w_wind = st.slider(
                        "💨 Wind",
                        0.0, 1.0, 0.3, 0.05,
                        help="Importance of wind potential"
                    )
                
                with col2:
                    w_risk = st.slider(
                        "⚠️ Risk Avoidance",
                        0.0, 1.0, 0.7, 0.05,
                        help="Penalty for high-risk areas"
                    )
                    w_grid = st.slider(
                        "🔌 Grid Proximity",
                        0.0, 1.0, 0.2, 0.05,
                        help="Importance of existing infrastructure"
                    )
            
            # Advanced options
            with st.expander("🔬 Advanced"):
                show_technical = st.checkbox(
                    "Show Technical Details",
                    value=False,
                    help="Display quantum circuit information"
                )
                
                comparison_mode = st.checkbox(
                    "Enable Comparison Mode",
                    value=False,
                    help="Compare multiple optimization methods"
                )
                
                noise_level = st.slider(
                    "Noise Level",
                    0.0, 0.1, 0.01, 0.01,
                    format="%.3f",
                    help="Simulated quantum noise level"
                )
            
            st.markdown("---")
            
            # Run button
            run_button = st.button(
                "🚀 RUN QUANTUM OPTIMIZATION",
                use_container_width=True,
                type="primary"
            )
            
            st.markdown("---")
            
            # Info section
            st.markdown("""
            <div style="font-size:0.8rem; color:#6b7280;">
                <p>📊 <strong>Data Summary</strong><br>
                Total sites: 45<br>
                Accessible: 36 (80%)<br>
                Avg Solar: 5.2 kWh<br>
                High Risk: 12 sites</p>
            </div>
            """, unsafe_allow_html=True)
        
        return {
            'n_sites': n_sites,
            'quantum_layers': quantum_layers,
            'use_quantum': use_quantum,
            'weights': {
                'solar': w_solar,
                'wind': w_wind,
                'risk': w_risk,
                'grid': w_grid
            },
            'show_technical': show_technical,
            'comparison_mode': comparison_mode,
            'noise_level': noise_level,
            'run_button': run_button
        }
    
    def render_metrics_grid(self, metrics: List[Tuple[str, str, str, Optional[str]]]):
        """
        Render metrics in a grid.
        
        Args:
            metrics: List of (label, value, color_class, tooltip)
        """
        cols = st.columns(len(metrics))
        
        for i, (label, value, color_class, tooltip) in enumerate(metrics):
            with cols[i]:
                tooltip_attr = f' title="{tooltip}"' if tooltip else ''
                st.markdown(f"""
                <div class="metric-card"{tooltip_attr}>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_progress(self, message: str, progress: float):
        """
        Render progress bar with message.
        
        Args:
            message: Progress message
            progress: Progress value (0-1)
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(message)
        progress_bar.progress(progress)
        
        return progress_bar, status_text
    
    def render_optimization_progress(self, callback: Optional[Callable] = None):
        """
        Render real-time optimization progress.
        
        Args:
            callback: Function to call for progress updates
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(msg: str, progress: float):
            status_text.text(msg)
            progress_bar.progress(progress)
            if callback:
                callback(msg, progress)
        
        return update_progress
    
    def render_map(
        self,
        df: pd.DataFrame,
        selected_indices: Optional[List[int]] = None,
        height: int = 500
    ):
        """
        Render interactive map.
        
        Args:
            df: DataFrame with site data
            selected_indices: Selected site indices
            height: Map height in pixels
        """
        from .map_visualizer import create_energy_map
        
        m = create_energy_map(df, selected_indices)
        folium_static(m, width=1200, height=height)
    
    def render_comparison_chart(
        self,
        results: Dict[str, Any],
        title: str = "Method Comparison"
    ):
        """
        Render comparison chart for different methods.
        
        Args:
            results: Dictionary of results by method
            title: Chart title
        """
        methods = list(results.keys())
        scores = [r.get('score', 0) for r in results.values()]
        times = [r.get('time', 0) for r in results.values()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Solution Quality', 'Execution Time'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=scores,
                name='Score',
                marker_color=self.config.primary_color,
                text=[f'{s:.1f}' for s in scores],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=times,
                name='Time (s)',
                marker_color=self.config.secondary_color,
                text=[f'{t:.2f}s' for t in times],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self, df: pd.DataFrame, selected_indices: List[int]):
        """
        Render risk analysis charts.
        
        Args:
            df: Full dataset
            selected_indices: Selected site indices
        """
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            risk_categories = []
            for risk in df['Risk_Score']:
                if risk <= 4:
                    risk_categories.append('Low Risk (0-4)')
                elif risk <= 7:
                    risk_categories.append('Medium Risk (5-7)')
                else:
                    risk_categories.append('High Risk (8-10)')
            
            risk_df = pd.DataFrame({
                'Risk Level': risk_categories,
                'Count': 1
            }).groupby('Risk Level').count().reset_index()
            
            fig = px.pie(
                risk_df,
                values='Count',
                names='Risk Level',
                title='Risk Distribution - All Sites',
                color='Risk Level',
                color_discrete_map={
                    'Low Risk (0-4)': self.config.success_color,
                    'Medium Risk (5-7)': self.config.warning_color,
                    'High Risk (8-10)': self.config.danger_color
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Selected sites risk
            selected_risks = df.iloc[selected_indices]['Risk_Score']
            
            fig = px.bar(
                x=[f"Site {i}" for i in range(len(selected_risks))],
                y=selected_risks,
                title='Risk Scores - Selected Sites',
                labels={'x': 'Site', 'y': 'Risk Score'},
                color=selected_risks,
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_suitability_chart(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        score_col: str = 'Suitability'
    ):
        """
        Render suitability comparison chart.
        
        Args:
            df: DataFrame with scores
            selected_indices: Selected site indices
            score_col: Score column name
        """
        if score_col not in df.columns:
            st.warning("Suitability scores not available")
            return
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Site': df['Region_ID'],
            'Type': ['Selected' if i in selected_indices else 'Candidate' for i in range(len(df))],
            'Score': df[score_col]
        })
        
        fig = px.bar(
            comparison.sort_values('Score', ascending=True),
            y='Site',
            x='Score',
            color='Type',
            title='Site Suitability Comparison',
            color_discrete_map={
                'Selected': self.config.success_color,
                'Candidate': self.config.info_color
            },
            orientation='h'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Suitability Score",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_optimization_history(self):
        """Render optimization history chart."""
        if not st.session_state.optimization_history:
            st.info("No optimization history available")
            return
        
        history = st.session_state.optimization_history
        
        df_history = pd.DataFrame(history)
        
        fig = px.line(
            df_history,
            x=range(len(df_history)),
            y='energy',
            title='Optimization Convergence History',
            labels={'x': 'Iteration', 'energy': 'Energy Value'},
            markers=True
        )
        
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_data_table(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        columns: Optional[List[str]] = None
    ):
        """
        Render interactive data table.
        
        Args:
            df: DataFrame
            selected_indices: Selected site indices
            columns: Columns to display
        """
        if columns is None:
            columns = ['Region_ID', 'Solar_Irradiance', 'Wind_Speed', 'Risk_Score', 'Accessibility']
        
        display_df = df[columns].copy()
        
        # Add selection indicator
        display_df['Selected'] = ['✅' if i in selected_indices else '' for i in range(len(df))]
        
        # Apply styling
        def color_risk(val):
            if isinstance(val, (int, float)):
                if val > 7:
                    return 'background-color: #fee2e2'
                elif val > 4:
                    return 'background-color: #fef3c7'
            return ''
        
        styled_df = display_df.style.applymap(
            color_risk,
            subset=['Risk_Score'] if 'Risk_Score' in columns else []
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    def render_export_section(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        results: Dict[str, Any]
    ):
        """
        Render data export section.
        
        Args:
            df: Full dataset
            selected_indices: Selected site indices
            results: Optimization results
        """
        st.markdown("### 📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        selected_df = df.iloc[selected_indices].copy()
        
        with col1:
            # CSV export
            csv = selected_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                "selected_sites.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = selected_df.to_json(orient='records', indent=2)
            st.download_button(
                "📋 Download JSON",
                json_str,
                "selected_sites.json",
                "application/json",
                use_container_width=True
            )
        
        with col3:
            # Report export
            if st.button("📑 Generate Report", use_container_width=True):
                from .report_generator import generate_optimization_report
                
                with st.spinner("Generating report..."):
                    report_path = generate_optimization_report(
                        df, selected_indices, results,
                        output_format='html',
                        filename='optimization_report'
                    )
                    
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Report",
                            f,
                            "optimization_report.html",
                            "text/html",
                            use_container_width=True
                        )
    
    def render_info_box(self, message: str, type: str = 'info'):
        """
        Render info/warning/success box.
        
        Args:
            message: Message to display
            type: Box type ('info', 'warning', 'success')
        """
        box_class = {
            'info': 'info-box',
            'warning': 'warning-box',
            'success': 'success-box'
        }.get(type, 'info-box')
        
        st.markdown(f"""
        <div class="{box_class}">
            {message}
        </div>
        """, unsafe_allow_html=True)
    
    def render_footer(self):
        """Render dashboard footer."""
        st.markdown("---")
        st.markdown(f"""
        <div class="footer">
            <p>⚡ Quantum-Enhanced Infrastructure Planning System | Developed for Hackathon Palestine</p>
            <p>🔬 Using QAOA for combinatorial optimization under uncertainty | v{self.config.page_title.split('v')[-1] if 'v' in self.config.page_title else '1.0.0'}</p>
            <p>© {time.strftime('%Y')} GazaGrid Quantum Solutions</p>
        </div>
        """, unsafe_allow_html=True)


# Utility function for quick dashboard creation
def create_dashboard(
    df: pd.DataFrame,
    config: Optional[DashboardConfig] = None
) -> GazaDashboard:
    """
    Quick dashboard creation utility.
    
    Example:
        >>> dashboard = create_dashboard(df)
        >>> dashboard.render_header()
        >>> params = dashboard.render_sidebar()
    """
    return GazaDashboard(config)


# Example usage in Streamlit app
if __name__ == "__main__":
    # This would be run with: streamlit run dashboard.py
    
    # Load data
    try:
        df = pd.read_csv('../../data/gaza_energy_data.csv')
    except:
        # Create sample data
        df = pd.DataFrame({
            'Region_ID': [f'Site_{i}' for i in range(20)],
            'Latitude': np.random.uniform(31.25, 31.58, 20),
            'Longitude': np.random.uniform(34.20, 34.55, 20),
            'Solar_Irradiance': np.random.uniform(4.5, 6.0, 20),
            'Wind_Speed': np.random.uniform(2.5, 6.5, 20),
            'Risk_Score': np.random.randint(2, 10, 20),
            'Accessibility': np.random.choice([0, 1], 20, p=[0.2, 0.8]),
            'Grid_Distance': np.random.randint(100, 5000, 20)
        })
    
    # Initialize dashboard
    dashboard = create_dashboard(df)
    
    # Render header
    dashboard.render_header()
    
    # Render step indicator
    dashboard.render_step_indicator(1)
    
    # Render sidebar
    params = dashboard.render_sidebar()
    
    # Main content
    if params['run_button']:
        dashboard.render_step_indicator(2)
        
        # Simulate optimization
        progress = dashboard.render_optimization_progress()
        progress("Initializing quantum circuit...", 0.2)
        time.sleep(1)
        progress("Running QAOA optimization...", 0.6)
        time.sleep(2)
        progress("Optimization complete!", 1.0)
        
        dashboard.render_step_indicator(3)
        
        # Sample results
        selected = [0, 3, 7, 12, 15]
        
        # Metrics
        metrics = [
            ("Selected Sites", "5", "metric-value", ""),
            ("Total Capacity", "850 kW", "metric-value", ""),
            ("Avg Risk", "4.2/10", "risk-low", ""),
            ("Quantum Score", "-1245.6", "metric-value", "")
        ]
        dashboard.render_metrics_grid(metrics)
        
        # Map
        dashboard.render_map(df, selected)
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Comparison", "📋 Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                dashboard.render_risk_analysis(df, selected)
            with col2:
                # Add suitability scores for demo
                df['Suitability'] = np.random.uniform(60, 95, len(df))
                dashboard.render_suitability_chart(df, selected)
        
        with tab2:
            # Comparison chart
            results = {
                'Greedy': {'score': 850, 'time': 0.1},
                'Genetic': {'score': 920, 'time': 2.3},
                'QAOA': {'score': 980, 'time': 5.6}
            }
            dashboard.render_comparison_chart(results)
        
        with tab3:
            dashboard.render_data_table(df, selected)
        
        # Export section
        dashboard.render_export_section(df, selected, {})
    
    # Footer
    dashboard.render_footer()
