"""
Real plotting utilities for creating trend radar visualizations.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)


class TrendRadarPlotter:
    """Creates actual visual plots for trend radar analysis"""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Plotter initialized with output directory: {self.output_dir.absolute()}")
        
        # Set up matplotlib to work without display (for headless systems)
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Set up matplotlib style
        plt.style.use('default')  # Use default instead of seaborn which may not be available
        
        # Color schemes for categories
        self.colors = {
            "technology": "#FF6B6B",
            "business": "#4ECDC4", 
            "social": "#45B7D1",
            "environmental": "#96CEB4",
            "unknown": "#95A5A6"
        }
    
    def create_trend_radar(
        self, 
        radar_data: List[Dict[str, Any]], 
        title: str = "Trend Radar Analysis",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create the main trend radar scatter plot
        
        Args:
            radar_data: List of trend data points
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot file
        """
        if not radar_data:
            logger.warning("No radar data provided for plotting")
            return ""
        
        logger.info(f"Creating trend radar plot with {len(radar_data)} data points")
        
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 10))
            logger.debug("Figure created successfully")
            
            # Extract data for plotting
            x_coords = [float(point.get('x', 3)) for point in radar_data]
            y_coords = [float(point.get('y', 2)) for point in radar_data] 
            sizes = [float(point.get('size', 10)) for point in radar_data]
            categories = [point.get('category', 'unknown') for point in radar_data]
            titles = [point.get('title', 'Unknown')[:30] + '...' if len(point.get('title', '')) > 30 else point.get('title', 'Unknown') for point in radar_data]
            
            logger.debug(f"Data extracted: {len(x_coords)} points")
            
            # Create color map for categories
            unique_categories = list(set(categories))
            colors = [self.colors.get(cat, self.colors['unknown']) for cat in categories]
            
            logger.debug(f"Categories: {unique_categories}")
            
            # Create the scatter plot
            scatter = ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=2)
            logger.debug("Scatter plot created")
            
            # Add quadrant backgrounds
            self._add_quadrant_backgrounds(ax)
            logger.debug("Quadrant backgrounds added")
            
            # Add quadrant labels
            self._add_quadrant_labels(ax)
            logger.debug("Quadrant labels added")
            
            # Customize axes
            ax.set_xlim(0.5, 4.5)
            ax.set_ylim(0.5, 4.5)
            ax.set_xlabel('Time Horizon →', fontsize=12, fontweight='bold')
            ax.set_ylabel('Impact Level ↑', fontsize=12, fontweight='bold')
            
            # Set custom tick labels
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(['Emerging\n(0-6m)', 'Short-term\n(6-18m)', 'Medium-term\n(1-3y)', 'Long-term\n(3y+)'])
            ax.set_yticks([1, 2, 3, 4])
            ax.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add title
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Create legend for categories
            legend_elements = []
            for cat in unique_categories:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=self.colors.get(cat, self.colors['unknown']),
                                               markersize=10, label=cat.title()))
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
                     title='Categories')
            
            logger.debug("Legend and axes configured")
            
            # Add trend labels for important trends (top 5 by size)
            trend_data_with_index = [(i, point) for i, point in enumerate(radar_data)]
            trend_data_with_index.sort(key=lambda x: x[1].get('size', 10), reverse=True)
            
            for i, (idx, point) in enumerate(trend_data_with_index[:5]):
                x, y = point.get('x', 3), point.get('y', 2)
                title_short = point.get('title', 'Unknown')[:20] + '...' if len(point.get('title', '')) > 20 else point.get('title', 'Unknown')
                
                # Add annotation with arrow
                ax.annotate(title_short, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            logger.debug("Annotations added")
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            # Generate save path if not provided
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = str(self.output_dir / f"trend_radar_{timestamp}.png")
            
            logger.info(f"Saving plot to: {save_path}")
            
            # Ensure output directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Plot saved successfully: {save_path}")
            
            # Verify file was created
            if Path(save_path).exists():
                file_size = Path(save_path).stat().st_size
                logger.info(f"File verified: {file_size} bytes")
            else:
                logger.error(f"File was not created: {save_path}")
                plt.close()
                return ""
            
            # Also save as interactive HTML using Plotly
            try:
                html_path = save_path.replace('.png', '_interactive.html')
                self._create_interactive_radar(radar_data, title, html_path)
                logger.info(f"Interactive plot saved: {html_path}")
            except Exception as e:
                logger.warning(f"Failed to create interactive plot: {e}")
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create trend radar plot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up matplotlib state
            plt.close('all')
            return ""
    
    def _add_quadrant_backgrounds(self, ax):
        """Add colored backgrounds for quadrants"""
        quadrants = [
            {'bounds': (2.5, 4.5, 2.5, 4.5), 'color': '#ffebee', 'alpha': 0.3},  # High impact, long-term
            {'bounds': (0.5, 2.5, 2.5, 4.5), 'color': '#e8f5e8', 'alpha': 0.3},  # High impact, short-term  
            {'bounds': (2.5, 4.5, 0.5, 2.5), 'color': '#f5f5f5', 'alpha': 0.3},  # Low impact, long-term
            {'bounds': (0.5, 2.5, 0.5, 2.5), 'color': '#e3f2fd', 'alpha': 0.3}   # Low impact, short-term
        ]
        
        for quad in quadrants:
            x_min, x_max, y_min, y_max = quad['bounds']
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   facecolor=quad['color'], alpha=quad['alpha'])
            ax.add_patch(rect)
    
    def _add_quadrant_labels(self, ax):
        """Add labels to quadrants"""
        labels = [
            {'pos': (3.5, 3.5), 'text': 'Strategic Bets\n(High Impact, Long-term)', 'ha': 'center', 'va': 'center'},
            {'pos': (1.5, 3.5), 'text': 'Quick Wins\n(High Impact, Short-term)', 'ha': 'center', 'va': 'center'},
            {'pos': (3.5, 1.5), 'text': 'Background Noise\n(Low Impact, Long-term)', 'ha': 'center', 'va': 'center'},
            {'pos': (1.5, 1.5), 'text': 'Tactical Moves\n(Low Impact, Short-term)', 'ha': 'center', 'va': 'center'}
        ]
        
        for label in labels:
            ax.text(label['pos'][0], label['pos'][1], label['text'], 
                   fontsize=9, ha=label['ha'], va=label['va'],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   fontweight='bold', color='#666666')
    
    def _create_interactive_radar(
        self, 
        radar_data: List[Dict[str, Any]], 
        title: str,
        save_path: str
    ):
        """Create interactive Plotly version of the radar chart"""
        
        if not radar_data:
            return
        
        # Prepare data for Plotly
        df_data = []
        for point in radar_data:
            df_data.append({
                'x': point.get('x', 3),
                'y': point.get('y', 2),
                'size': point.get('size', 10),
                'category': point.get('category', 'unknown'),
                'title': point.get('title', 'Unknown'),
                'description': point.get('description', 'No description')[:100] + '...',
                'confidence': point.get('confidence', 0.5),
                'time_horizon': point.get('time_horizon_label', 'Medium-term'),
                'impact_level': point.get('impact_level', 'medium')
            })
        
        df = pd.DataFrame(df_data)
        
        # Create interactive scatter plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            size='size',
            color='category',
            hover_data=['confidence', 'time_horizon', 'impact_level'],
            title=title,
            color_discrete_map=self.colors
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Time Horizon →",
            yaxis_title="Impact Level ↑",
            xaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Emerging<br>(0-6m)', 'Short-term<br>(6-18m)', 'Medium-term<br>(1-3y)', 'Long-term<br>(3y+)'],
                range=[0.5, 4.5]
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Low', 'Medium', 'High', 'Critical'],
                range=[0.5, 4.5]
            ),
            showlegend=True,
            width=900,
            height=700
        )
        
        # Add quadrant shapes
        shapes = [
            # Strategic Bets
            dict(type="rect", x0=2.5, y0=2.5, x1=4.5, y1=4.5, 
                 fillcolor="rgba(255,0,0,0.1)", line=dict(width=0)),
            # Quick Wins
            dict(type="rect", x0=0.5, y0=2.5, x1=2.5, y1=4.5,
                 fillcolor="rgba(0,255,0,0.1)", line=dict(width=0)),
            # Background Noise
            dict(type="rect", x0=2.5, y0=0.5, x1=4.5, y1=2.5,
                 fillcolor="rgba(128,128,128,0.1)", line=dict(width=0)),
            # Tactical Moves  
            dict(type="rect", x0=0.5, y0=0.5, x1=2.5, y1=2.5,
                 fillcolor="rgba(0,0,255,0.1)", line=dict(width=0))
        ]
        
        fig.update_layout(shapes=shapes)
        
        # Save interactive plot
        fig.write_html(save_path)
        logger.info(f"Interactive radar saved to: {save_path}")
    
    def create_supporting_charts(
        self, 
        supporting_data: Dict[str, Any], 
        radar_data: List[Dict[str, Any]],
        save_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Create supporting visualizations"""
        
        if not save_dir:
            save_dir = str(self.output_dir)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        saved_plots = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Category Distribution Pie Chart
        if supporting_data.get('category_distribution'):
            pie_path = self._create_category_pie_chart(
                supporting_data['category_distribution'],
                str(save_dir / f"category_distribution_{timestamp}.png")
            )
            saved_plots['category_distribution'] = pie_path
        
        # 2. Impact vs Confidence Scatter Plot
        if radar_data:
            scatter_path = self._create_confidence_impact_scatter(
                radar_data,
                str(save_dir / f"confidence_vs_impact_{timestamp}.png")
            )
            saved_plots['confidence_vs_impact'] = scatter_path
        
        # 3. Timeline Distribution Bar Chart
        if supporting_data.get('timeline_distribution'):
            timeline_path = self._create_timeline_bar_chart(
                supporting_data['timeline_distribution'],
                str(save_dir / f"timeline_distribution_{timestamp}.png")
            )
            saved_plots['timeline_distribution'] = timeline_path
        
        # 4. Trend Summary Dashboard
        if radar_data and supporting_data:
            dashboard_path = self._create_summary_dashboard(
                radar_data, supporting_data,
                str(save_dir / f"trend_dashboard_{timestamp}.png")
            )
            saved_plots['dashboard'] = dashboard_path
        
        logger.info(f"Created {len(saved_plots)} supporting charts")
        return saved_plots
    
    def _create_category_pie_chart(self, category_data: Dict[str, int], save_path: str) -> str:
        """Create category distribution pie chart"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        categories = list(category_data.keys())
        values = list(category_data.values())
        colors_list = [self.colors.get(cat, self.colors['unknown']) for cat in categories]
        
        wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors_list, autopct='%1.1f%%',
                                         startangle=90, explode=[0.05] * len(categories))
        
        # Beautify the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Trend Distribution by Category', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _create_confidence_impact_scatter(self, radar_data: List[Dict[str, Any]], save_path: str) -> str:
        """Create confidence vs impact scatter plot"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        confidences = [point.get('confidence', 0.5) for point in radar_data]
        impacts = [point.get('y', 2) for point in radar_data]
        categories = [point.get('category', 'unknown') for point in radar_data]
        sizes = [point.get('size', 10) for point in radar_data]
        
        # Create scatter plot
        for category in set(categories):
            cat_confidences = [conf for conf, cat in zip(confidences, categories) if cat == category]
            cat_impacts = [imp for imp, cat in zip(impacts, categories) if cat == category]
            cat_sizes = [size for size, cat in zip(sizes, categories) if cat == category]
            
            ax.scatter(cat_confidences, cat_impacts, s=cat_sizes, 
                      c=self.colors.get(category, self.colors['unknown']), 
                      label=category.title(), alpha=0.7, edgecolors='white', linewidth=1)
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impact Level', fontsize=12, fontweight='bold')
        ax.set_title('Confidence vs Impact Analysis', fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _create_timeline_bar_chart(self, timeline_data: Dict[str, Any], save_path: str) -> str:
        """Create timeline distribution bar chart"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract timeline data
        timelines = []
        counts = []
        
        for timeline, data in timeline_data.items():
            if isinstance(data, dict) and 'count' in data:
                timelines.append(timeline.replace('_', ' ').title())
                counts.append(data['count'])
            elif isinstance(data, int):
                timelines.append(timeline.replace('_', ' ').title())
                counts.append(data)
        
        if not timelines:
            # Fallback if data structure is different
            timelines = list(timeline_data.keys())
            counts = list(timeline_data.values())
        
        # Create bar chart
        bars = ax.bar(timelines, counts, color=[self.colors['technology'], self.colors['business'], 
                                               self.colors['social'], self.colors['environmental']][:len(timelines)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Time Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Trends', fontsize=12, fontweight='bold')
        ax.set_title('Trend Distribution by Time Horizon', fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _create_summary_dashboard(
        self, 
        radar_data: List[Dict[str, Any]], 
        supporting_data: Dict[str, Any],
        save_path: str
    ) -> str:
        """Create a comprehensive summary dashboard"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trend Radar Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Mini radar chart
        x_coords = [point.get('x', 3) for point in radar_data]
        y_coords = [point.get('y', 2) for point in radar_data]
        sizes = [point.get('size', 10) * 0.5 for point in radar_data]  # Smaller for dashboard
        categories = [point.get('category', 'unknown') for point in radar_data]
        colors_list = [self.colors.get(cat, self.colors['unknown']) for cat in categories]
        
        ax1.scatter(x_coords, y_coords, s=sizes, c=colors_list, alpha=0.7, edgecolors='white')
        ax1.set_xlim(0.5, 4.5)
        ax1.set_ylim(0.5, 4.5)
        ax1.set_title('Trend Radar Overview', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Category distribution
        if supporting_data.get('category_distribution'):
            cat_data = supporting_data['category_distribution']
            categories = list(cat_data.keys())
            values = list(cat_data.values())
            colors_list = [self.colors.get(cat, self.colors['unknown']) for cat in categories]
            
            ax2.pie(values, labels=categories, colors=colors_list, autopct='%1.1f%%')
            ax2.set_title('Category Distribution', fontweight='bold')
        
        # 3. Confidence histogram
        confidences = [point.get('confidence', 0.5) for point in radar_data]
        ax3.hist(confidences, bins=10, color=self.colors['technology'], alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Number of Trends')
        ax3.set_title('Confidence Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Impact vs Time correlation
        impacts = [point.get('y', 2) for point in radar_data]
        times = [point.get('x', 3) for point in radar_data]
        
        ax4.scatter(times, impacts, c=colors_list, alpha=0.7, s=50)
        ax4.set_xlabel('Time Horizon')
        ax4.set_ylabel('Impact Level')
        ax4.set_title('Impact vs Time Correlation', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    