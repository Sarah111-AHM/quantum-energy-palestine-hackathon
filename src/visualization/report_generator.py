"""
Professional Report Generator for Gaza Strip Energy Infrastructure Planning.
Creates comprehensive PDF/HTML reports with analysis, maps, and recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import WeasyPrint for PDF generation
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint not installed. PDF generation disabled.")


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "GazaGrid - Energy Infrastructure Optimization Report"
    author: str = "Quantum Energy Palestine Hackathon Team"
    institution: str = "Gaza Energy Authority"
    logo_path: Optional[str] = None
    include_technical_details: bool = True
    include_recommendations: bool = True
    include_cost_estimates: bool = True
    language: str = "en"  # 'en' or 'ar'
    company_name: str = "GazaGrid Quantum Solutions"
    version: str = "1.0.0"


class ReportGenerator:
    """
    Professional report generator for optimization results.
    
    Features:
    - HTML and PDF output
    - Interactive tables and charts
    - Executive summary
    - Technical appendix
    - Cost estimates
    - Risk analysis
    - Arabic language support
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.report_data = {}
        self.figures = []
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        logger.info(f"Initialized ReportGenerator: {self.config.title}")
    
    def generate_report(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        optimization_results: Dict[str, Any],
        output_format: str = 'html',
        filename: str = 'optimization_report'
    ) -> str:
        """
        Generate complete optimization report.
        
        Args:
            df: Full dataset
            selected_indices: Selected site indices
            optimization_results: Results from optimizer
            output_format: 'html' or 'pdf'
            filename: Output filename (without extension)
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating {output_format} report...")
        
        # Prepare data
        self._prepare_report_data(df, selected_indices, optimization_results)
        
        # Generate figures
        self._generate_all_figures()
        
        # Create HTML content
        html_content = self._create_html_report()
        
        # Save output
        if output_format.lower() == 'pdf' and WEASYPRINT_AVAILABLE:
            output_path = self._save_as_pdf(html_content, filename)
        else:
            output_path = self._save_as_html(html_content, filename)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def _prepare_report_data(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        optimization_results: Dict[str, Any]
    ):
        """Prepare and structure report data."""
        
        # Selected sites data
        selected_df = df.iloc[selected_indices].copy()
        
        # Calculate statistics
        self.report_data = {
            'metadata': {
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': self.config.version,
                'author': self.config.author
            },
            'summary': {
                'total_sites': len(df),
                'accessible_sites': len(df[df['Accessibility'] == 1]),
                'selected_sites': len(selected_indices),
                'avg_solar': df['Solar_Irradiance'].mean(),
                'avg_wind': df['Wind_Speed'].mean(),
                'avg_risk': df['Risk_Score'].mean(),
                'total_capacity': selected_df['Solar_Irradiance'].sum() * 100,  # kW
                'selected_avg_risk': selected_df['Risk_Score'].mean(),
                'selected_avg_solar': selected_df['Solar_Irradiance'].mean()
            },
            'selected_sites': selected_df.to_dict('records'),
            'optimization': optimization_results,
            'risk_analysis': self._analyze_risks(df, selected_indices),
            'cost_estimates': self._estimate_costs(selected_df),
            'recommendations': self._generate_recommendations(df, selected_indices)
        }
    
    def _analyze_risks(
        self,
        df: pd.DataFrame,
        selected_indices: List[int]
    ) -> Dict[str, Any]:
        """Analyze risk distribution."""
        
        all_risks = df['Risk_Score']
        selected_risks = df.iloc[selected_indices]['Risk_Score']
        
        return {
            'overall_distribution': {
                'low_risk': len(all_risks[all_risks <= 4]),
                'medium_risk': len(all_risks[(all_risks > 4) & (all_risks <= 7)]),
                'high_risk': len(all_risks[all_risks > 7])
            },
            'selected_distribution': {
                'low_risk': len(selected_risks[selected_risks <= 4]),
                'medium_risk': len(selected_risks[(selected_risks > 4) & (selected_risks <= 7)]),
                'high_risk': len(selected_risks[selected_risks > 7])
            },
            'risk_mitigation': self._suggest_risk_mitigation(selected_risks)
        }
    
    def _suggest_risk_mitigation(self, risks: pd.Series) -> List[str]:
        """Suggest risk mitigation strategies."""
        suggestions = []
        
        if any(risks > 7):
            suggestions.append("Install reinforced protective structures for high-risk sites")
        if any((risks > 4) & (risks <= 7)):
            suggestions.append("Implement enhanced security monitoring for medium-risk areas")
        if risks.mean() > 5:
            suggestions.append("Develop emergency response protocols for selected sites")
        
        return suggestions
    
    def _estimate_costs(self, selected_df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate installation and operational costs."""
        
        # Cost assumptions (in USD)
        SOLAR_COST_PER_KW = 1200
        INSTALLATION_FACTOR = 0.3
        MAINTENANCE_YEARLY = 0.02
        GRID_CONNECTION_COST = 50000
        
        total_capacity = selected_df['Solar_Irradiance'].sum() * 100  # kW
        
        estimates = {
            'equipment_cost': total_capacity * SOLAR_COST_PER_KW,
            'installation_cost': total_capacity * SOLAR_COST_PER_KW * INSTALLATION_FACTOR,
            'grid_connection': GRID_CONNECTION_COST * len(selected_df),
            'annual_maintenance': total_capacity * SOLAR_COST_PER_KW * MAINTENANCE_YEARLY
        }
        
        estimates['total_initial'] = (
            estimates['equipment_cost'] +
            estimates['installation_cost'] +
            estimates['grid_connection']
        )
        
        estimates['annual_operating'] = estimates['annual_maintenance']
        
        # Format with commas
        for key in estimates:
            estimates[key] = f"${estimates[key]:,.0f}"
        
        return estimates
    
    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        selected_indices: List[int]
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        selected_df = df.iloc[selected_indices]
        
        # Solar potential recommendations
        if selected_df['Solar_Irradiance'].mean() > 5.5:
            recommendations.append("⚡ Excellent solar potential - prioritize solar PV installation")
        elif selected_df['Solar_Irradiance'].mean() > 4.5:
            recommendations.append("☀️ Good solar potential - consider hybrid solar-wind systems")
        
        # Risk-based recommendations
        if selected_df['Risk_Score'].mean() > 6:
            recommendations.append("🛡️ Implement enhanced security measures and remote monitoring")
        elif selected_df['Risk_Score'].mean() > 4:
            recommendations.append("📡 Install standard security fencing and monitoring")
        
        # Geographic distribution
        if len(selected_indices) > 3:
            recommendations.append("🌍 Distributed sites enable grid resilience and load balancing")
        
        # Phased implementation
        recommendations.append("📅 Phase 1: Install at 3 highest-priority sites within 6 months")
        recommendations.append("📅 Phase 2: Expand to remaining sites based on Phase 1 performance")
        
        # Community engagement
        recommendations.append("👥 Engage local communities for site protection and maintenance")
        
        return recommendations
    
    def _generate_all_figures(self):
        """Generate all figures for the report."""
        self.figures = []
        
        # Risk distribution
        self.figures.append(('Risk Distribution', self._plot_risk_distribution()))
        
        # Solar vs Risk scatter
        self.figures.append(('Solar Potential vs Risk', self._plot_solar_vs_risk()))
        
        # Selected sites comparison
        self.figures.append(('Selected Sites Analysis', self._plot_selected_sites()))
        
        # Capacity contribution
        self.figures.append(('Capacity Contribution', self._plot_capacity_pie()))
    
    def _plot_risk_distribution(self) -> str:
        """Plot risk distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall risk distribution
        risk_data = self.report_data['risk_analysis']['overall_distribution']
        categories = ['Low (0-4)', 'Medium (5-7)', 'High (8-10)']
        values = [risk_data['low_risk'], risk_data['medium_risk'], risk_data['high_risk']]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        
        ax1.bar(categories, values, color=colors)
        ax1.set_title('Overall Risk Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Sites')
        
        # Selected risk distribution
        selected_risk = self.report_data['risk_analysis']['selected_distribution']
        selected_values = [
            selected_risk['low_risk'],
            selected_risk['medium_risk'],
            selected_risk['high_risk']
        ]
        
        ax2.bar(categories, selected_values, color=colors)
        ax2.set_title('Selected Sites Risk Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Sites')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_solar_vs_risk(self) -> str:
        """Plot solar potential vs risk scatter."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data
        selected_indices = [s['index'] for s in self.report_data['selected_sites']]
        
        # Plot all sites
        ax.scatter(
            self.report_data['summary']['avg_solar'] * 0.8,  # Placeholder - use actual data
            self.report_data['summary']['avg_risk'] * 0.8,
            alpha=0.5,
            label='Candidate Sites',
            color='#9ca3af'
        )
        
        # Plot selected sites
        ax.scatter(
            self.report_data['summary']['selected_avg_solar'],
            self.report_data['summary']['selected_avg_risk'],
            s=200,
            color='#10b981',
            edgecolor='white',
            linewidth=2,
            label='Selected Sites',
            zorder=5
        )
        
        # Add quadrant lines
        ax.axhline(y=7, color='#ef4444', linestyle='--', alpha=0.5, label='High Risk Threshold')
        ax.axvline(x=5.5, color='#10b981', linestyle='--', alpha=0.5, label='High Solar Threshold')
        
        ax.set_xlabel('Solar Irradiance (kWh/m²/day)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Site Selection Analysis: Solar vs Risk', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _plot_selected_sites(self) -> str:
        """Plot selected sites comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sites = self.report_data['selected_sites']
        names = [s['Region_ID'] for s in sites]
        solar = [s['Solar_Irradiance'] for s in sites]
        risk = [s['Risk_Score'] for s in sites]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, solar, width, label='Solar (kWh)', color='#fbbf24')
        bars2 = ax.bar(x + width/2, risk, width, label='Risk Score', color='#ef4444')
        
        ax.set_xlabel('Site')
        ax.set_ylabel('Value')
        ax.set_title('Selected Sites Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_capacity_pie(self) -> str:
        """Plot capacity contribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sites = self.report_data['selected_sites']
        names = [s['Region_ID'] for s in sites]
        capacities = [s['Solar_Irradiance'] * 100 for s in sites]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sites)))
        
        wedges, texts, autotexts = ax.pie(
            capacities,
            labels=names,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        # Style
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Capacity Contribution by Site', fontsize=14, fontweight='bold')
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        
        # CSS styles
        css = self._get_css_styles()
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="{self.config.language}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.config.title}</title>
            <style>{css}</style>
        </head>
        <body>
            <div class="report-container">
                <!-- Header -->
                <div class="header">
                    <h1>{self.config.title}</h1>
                    <p class="subtitle">Generated on {self.report_data['metadata']['generated_at']}</p>
                    <p class="subtitle">Prepared by: {self.config.author}</p>
                </div>
                
                <!-- Executive Summary -->
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="summary-label">Total Sites Evaluated</div>
                            <div class="summary-value">{self.report_data['summary']['total_sites']}</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-label">Selected Sites</div>
                            <div class="summary-value">{self.report_data['summary']['selected_sites']}</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-label">Estimated Capacity</div>
                            <div class="summary-value">{self.report_data['summary']['total_capacity']:.0f} kW</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-label">Average Risk</div>
                            <div class="summary-value risk-{'high' if self.report_data['summary']['selected_avg_risk'] > 7 else 'medium' if self.report_data['summary']['selected_avg_risk'] > 4 else 'low'}">
                                {self.report_data['summary']['selected_avg_risk']:.1f}/10
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations -->
                <div class="section">
                    <h2>📈 Data Visualization</h2>
        """
        
        # Add figures
        for i, (title, img_data) in enumerate(self.figures):
            html += f"""
                    <div class="figure-container">
                        <h3>{title}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{title}" class="report-figure">
                    </div>
            """
        
        # Selected Sites Table
        html += """
                <div class="section">
                    <h2>📍 Selected Sites Details</h2>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Site ID</th>
                                <th>Latitude</th>
                                <th>Longitude</th>
                                <th>Solar (kWh)</th>
                                <th>Wind (m/s)</th>
                                <th>Risk Score</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for site in self.report_data['selected_sites']:
            risk_class = 'risk-high' if site['Risk_Score'] > 7 else 'risk-medium' if site['Risk_Score'] > 4 else 'risk-low'
            html += f"""
                            <tr>
                                <td>{site['Region_ID']}</td>
                                <td>{site['Latitude']:.6f}</td>
                                <td>{site['Longitude']:.6f}</td>
                                <td>{site['Solar_Irradiance']:.2f}</td>
                                <td>{site['Wind_Speed']:.2f}</td>
                                <td class="{risk_class}">{site['Risk_Score']}/10</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
        """
        
        # Risk Analysis
        html += """
                <div class="section">
                    <h2>⚠️ Risk Analysis</h2>
                    <div class="risk-analysis">
                        <h3>Risk Distribution</h3>
                        <div class="risk-bars">
        """
        
        risk_dist = self.report_data['risk_analysis']['overall_distribution']
        total = sum(risk_dist.values())
        
        for level, count in risk_dist.items():
            percentage = (count / total) * 100
            color = '#10b981' if 'low' in level else '#f59e0b' if 'medium' in level else '#ef4444'
            html += f"""
                            <div class="risk-bar">
                                <span class="risk-label">{level.replace('_', ' ').title()}</span>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {percentage}%; background-color: {color};"></div>
                                </div>
                                <span class="risk-count">{count} sites ({percentage:.1f}%)</span>
                            </div>
            """
        
        html += """
                        </div>
                    </div>
                </div>
        """
        
        # Mitigation Strategies
        html += """
                <div class="section">
                    <h2>🛡️ Risk Mitigation Strategies</h2>
                    <ul class="mitigation-list">
        """
        
        for strategy in self.report_data['risk_analysis']['risk_mitigation']:
            html += f"<li>{strategy}</li>"
        
        html += """
                    </ul>
                </div>
        """
        
        # Cost Estimates
        if self.config.include_cost_estimates:
            html += """
                <div class="section">
                    <h2>💰 Cost Estimates</h2>
                    <table class="cost-table">
                        <tr>
                            <td>Equipment Cost</td>
                            <td class="cost-value">{equipment_cost}</td>
                        </tr>
                        <tr>
                            <td>Installation Cost</td>
                            <td class="cost-value">{installation_cost}</td>
                        </tr>
                        <tr>
                            <td>Grid Connection</td>
                            <td class="cost-value">{grid_connection}</td>
                        </tr>
                        <tr class="total-row">
                            <td>Total Initial Investment</td>
                            <td class="cost-value">{total_initial}</td>
                        </tr>
                        <tr>
                            <td>Annual Maintenance</td>
                            <td class="cost-value">{annual_maintenance}</td>
                        </tr>
                    </table>
                </div>
            """.format(**self.report_data['cost_estimates'])
        
        # Recommendations
        if self.config.include_recommendations:
            html += """
                <div class="section">
                    <h2>📋 Recommendations</h2>
                    <ul class="recommendations-list">
            """
            
            for rec in self.report_data['recommendations']:
                html += f"<li>{rec}</li>"
            
            html += """
                    </ul>
                </div>
            """
        
        # Technical Details
        if self.config.include_technical_details:
            html += f"""
                <div class="section technical-details">
                    <h2>🔧 Technical Details</h2>
                    <table>
                        <tr>
                            <td>Optimization Algorithm</td>
                            <td>{self.report_data['optimization'].get('algorithm', 'QAOA')}</td>
                        </tr>
                        <tr>
                            <td>Number of Qubits</td>
                            <td>{self.report_data['optimization'].get('qubits', len(self.report_data['selected_sites']))}</td>
                        </tr>
                        <tr>
                            <td>QAOA Layers</td>
                            <td>{self.report_data['optimization'].get('layers', 2)}</td>
                        </tr>
                        <tr>
                            <td>Energy Value</td>
                            <td>{self.report_data['optimization'].get('energy', 0):.4f}</td>
                        </tr>
                    </table>
                </div>
            """
        
        # Footer
        html += f"""
                <div class="footer">
                    <p>Report generated by {self.config.company_name} v{self.config.version}</p>
                    <p>© {datetime.now().year} All rights reserved</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for report."""
        return """
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                background-color: #f3f4f6;
                color: #1f2937;
                line-height: 1.6;
            }
            
            .report-container {
                max-width: 1200px;
                margin: 2rem auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-radius: 12px;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 3rem 2rem;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 1rem;
                font-weight: 700;
            }
            
            .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .section {
                padding: 2rem;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .section h2 {
                color: #1e3c72;
                margin-bottom: 1.5rem;
                font-size: 1.8rem;
            }
            
            .section h3 {
                color: #374151;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }
            
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-top: 1rem;
            }
            
            .summary-card {
                background-color: #f9fafb;
                padding: 1.5rem;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #2a5298;
            }
            
            .summary-label {
                color: #6b7280;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .summary-value {
                font-size: 2rem;
                font-weight: 700;
                color: #1e3c72;
                margin-top: 0.5rem;
            }
            
            .risk-low {
                color: #10b981;
            }
            
            .risk-medium {
                color: #f59e0b;
            }
            
            .risk-high {
                color: #ef4444;
                font-weight: bold;
            }
            
            .figure-container {
                margin: 2rem 0;
                text-align: center;
            }
            
            .report-figure {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
            }
            
            .data-table th {
                background-color: #1e3c72;
                color: white;
                padding: 0.75rem;
                text-align: left;
            }
            
            .data-table td {
                padding: 0.75rem;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .data-table tr:hover {
                background-color: #f9fafb;
            }
            
            .risk-bars {
                margin: 1.5rem 0;
            }
            
            .risk-bar {
                margin: 1rem 0;
            }
            
            .risk-label {
                display: inline-block;
                width: 120px;
                font-weight: 500;
            }
            
            .progress-bar {
                display: inline-block;
                width: 300px;
                height: 20px;
                background-color: #e5e7eb;
                border-radius: 10px;
                overflow: hidden;
                margin: 0 1rem;
            }
            
            .progress-fill {
                height: 100%;
                transition: width 0.3s ease;
            }
            
            .risk-count {
                color: #6b7280;
                font-size: 0.9rem;
            }
            
            .mitigation-list, .recommendations-list {
                list-style: none;
                padding: 0;
            }
            
            .mitigation-list li, .recommendations-list li {
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
                background-color: #f9fafb;
                border-radius: 6px;
                border-left: 3px solid #2a5298;
            }
            
            .cost-table {
                width: 100%;
                max-width: 500px;
                margin: 1rem 0;
            }
            
            .cost-table td {
                padding: 0.75rem;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .cost-table td:last-child {
                text-align: right;
                font-weight: 600;
            }
            
            .total-row {
                background-color: #f3f4f6;
                font-weight: 700;
            }
            
            .technical-details {
                background-color: #f8fafc;
            }
            
            .footer {
                text-align: center;
                padding: 2rem;
                color: #9ca3af;
                font-size: 0.9rem;
                background-color: #f9fafb;
            }
            
            @media print {
                body {
                    background-color: white;
                }
                .report-container {
                    box-shadow: none;
                    margin: 0;
                }
            }
        """
    
    def _save_as_html(self, html_content: str, filename: str) -> str:
        """Save report as HTML file."""
        if not filename.endswith('.html'):
            filename += '.html'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _save_as_pdf(self, html_content: str, filename: str) -> str:
        """Save report as PDF file using WeasyPrint."""
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint not available. Saving as HTML instead.")
            return self._save_as_html(html_content, filename)
        
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        HTML(string=html_content).write_pdf(filename)
        return filename


# Utility function for quick report generation
def generate_optimization_report(
    df: pd.DataFrame,
    selected_indices: List[int],
    optimization_results: Dict[str, Any],
    output_format: str = 'html',
    filename: str = 'optimization_report'
) -> str:
    """
    Quick report generation utility.
    
    Example:
        >>> report_path = generate_optimization_report(df, [0,1,2], results)
    """
    generator = ReportGenerator()
    return generator.generate_report(
        df, selected_indices, optimization_results,
        output_format=output_format,
        filename=filename
    )


# Example usage
if __name__ == "__main__":
    # Create sample data
    import pandas as pd
    
    try:
        # Load data
        df = pd.read_csv('../../data/gaza_energy_data.csv')
        
        # Sample selected indices
        selected = [0, 5, 10, 15, 20]
        
        # Sample optimization results
        results = {
            'algorithm': 'QAOA',
            'qubits': len(df),
            'layers': 2,
            'energy': -1250.45,
            'iterations': 150
        }
        
        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate_report(
            df, selected, results,
            output_format='html',
            filename='sample_report'
        )
        
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error: {e}")
