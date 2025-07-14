"""
Advanced data analysis and visualization module.
Provides comprehensive EDA, statistical analysis, and interactive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from config import config

warnings.filterwarnings('ignore')

class AdvancedDataAnalyzer:
    """Advanced data analysis and visualization class."""
    
    def __init__(self, data_path: str = None):
        """Initialize the analyzer with data."""
        self.data_path = data_path or config.DATA_PATH
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and prepare the dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
    def basic_info(self):
        """Display basic information about the dataset."""
        print("="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print("\nDataset Shape:", self.df.shape)
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\nStatistical Summary:")
        print(self.df.describe(include='all'))
        
        print("\nTarget Distribution:")
        print(self.df[config.TARGET_COLUMN].value_counts().sort_index())
        
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for EDA."""
        plt.style.use(config.STYLE)
        
        # Target Distribution Analysis
        self._plot_target_distribution()
        
        # Categorical Features Analysis
        self._plot_categorical_features()
        
        # Numerical Features Analysis
        self._plot_numerical_features()
        
        # Correlation Analysis
        self._plot_correlation_analysis()
        
        # Feature Importance Analysis
        self._plot_feature_importance()
        
        # Interactive Visualizations
        self._create_interactive_plots()
        
    def _plot_target_distribution(self):
        """Plot target variable distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold')
        
        # Count plot
        sns.countplot(data=self.df, x=config.TARGET_COLUMN, 
                     order=['L', 'M', 'H'], ax=axes[0, 0])
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Performance Level')
        
        # Pie chart
        target_counts = self.df[config.TARGET_COLUMN].value_counts()
        axes[0, 1].pie(target_counts.values, labels=target_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Class Distribution (Pie Chart)')
        
        # Gender vs Class
        pd.crosstab(self.df['gender'], self.df[config.TARGET_COLUMN]).plot(
            kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Gender vs Performance')
        axes[1, 0].set_xlabel('Gender')
        axes[1, 0].legend(title='Performance Level')
        
        # Nationality vs Class
        nationality_class = pd.crosstab(self.df['NationalITy'], self.df[config.TARGET_COLUMN])
        nationality_class.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Nationality vs Performance')
        axes[1, 1].set_xlabel('Nationality')
        axes[1, 1].legend(title='Performance Level')
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/target_analysis.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def _plot_categorical_features(self):
        """Plot categorical features analysis."""
        categorical_features = [col for col in config.CATEGORICAL_FEATURES if col in self.df.columns]
        
        n_cols = 3
        n_rows = (len(categorical_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Categorical Features Analysis', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                sns.countplot(data=self.df, x=feature, hue=config.TARGET_COLUMN, 
                             hue_order=['L', 'M', 'H'], ax=axes[i])
                axes[i].set_title(f'{feature} vs Performance')
                axes[i].tick_params(axis='x', rotation=45)
                
        # Hide empty subplots
        for j in range(len(categorical_features), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/categorical_analysis.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def _plot_numerical_features(self):
        """Plot numerical features analysis."""
        numerical_features = [col for col in config.NUMERICAL_FEATURES if col in self.df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Numerical Features Analysis', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(numerical_features):
            if i < 4:  # Only plot first 4 features
                row, col = i // 2, i % 2
                
                # Box plot
                sns.boxplot(data=self.df, x=config.TARGET_COLUMN, y=feature, 
                           order=['L', 'M', 'H'], ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Distribution by Performance')
                
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/numerical_analysis.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
        # Additional numerical analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Numerical Features Distribution', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(numerical_features):
            if i < 4:
                row, col = i // 2, i % 2
                
                # Histogram with KDE
                sns.histplot(data=self.df, x=feature, hue=config.TARGET_COLUMN, 
                           kde=True, ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Distribution')
                
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/numerical_distribution.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def _plot_correlation_analysis(self):
        """Plot correlation analysis."""
        # Encode categorical variables for correlation
        df_encoded = self.df.copy()
        le = LabelEncoder()
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = le.fit_transform(df_encoded[col])
                
        # Correlation matrix
        corr_matrix = df_encoded.corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Full correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=axes[0], fmt='.2f')
        axes[0].set_title('Full Correlation Matrix')
        
        # Target correlation
        target_corr = corr_matrix[config.TARGET_COLUMN].sort_values(ascending=False)
        sns.barplot(x=target_corr.values, y=target_corr.index, ax=axes[1])
        axes[1].set_title('Features Correlation with Target')
        axes[1].set_xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/correlation_analysis.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def _plot_feature_importance(self):
        """Plot feature importance using mutual information."""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X = self.df.drop(config.TARGET_COLUMN, axis=1)
        y = self.df[config.TARGET_COLUMN]
        
        # Encode categorical variables
        X_encoded = X.copy()
        le = LabelEncoder()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                X_encoded[col] = le.fit_transform(X_encoded[col])
                
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_encoded, y)
        mi_scores = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=mi_scores.values, y=mi_scores.index)
        plt.title('Feature Importance (Mutual Information)', fontsize=16, fontweight='bold')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/feature_importance.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def _create_interactive_plots(self):
        """Create interactive visualizations using Plotly."""
        # Interactive scatter plot
        fig = px.scatter_matrix(
            self.df, 
            dimensions=config.NUMERICAL_FEATURES,
            color=config.TARGET_COLUMN,
            title="Interactive Scatter Matrix of Numerical Features"
        )
        fig.write_html(f'{config.FIGURES_DIR}/interactive_scatter_matrix.html')
        
        # Interactive parallel coordinates
        fig = px.parallel_coordinates(
            self.df,
            dimensions=config.NUMERICAL_FEATURES + [config.TARGET_COLUMN],
            color=config.TARGET_COLUMN,
            title="Parallel Coordinates Plot"
        )
        fig.write_html(f'{config.FIGURES_DIR}/parallel_coordinates.html')
        
        print("Interactive plots saved to figures directory!")
        
    def statistical_analysis(self):
        """Perform statistical analysis."""
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Chi-square test for categorical variables
        print("Chi-square Test Results:")
        print("-" * 40)
        
        categorical_features = [col for col in config.CATEGORICAL_FEATURES if col in self.df.columns]
        
        for feature in categorical_features:
            contingency_table = pd.crosstab(self.df[feature], self.df[config.TARGET_COLUMN])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            print(f"{feature}:")
            print(f"  Chi-square: {chi2:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            print()
            
        # ANOVA test for numerical variables
        print("\nANOVA Test Results:")
        print("-" * 40)
        
        numerical_features = [col for col in config.NUMERICAL_FEATURES if col in self.df.columns]
        
        for feature in numerical_features:
            groups = [self.df[self.df[config.TARGET_COLUMN] == level][feature] 
                     for level in ['L', 'M', 'H']]
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            print(f"{feature}:")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            print()
            
    def generate_insights(self):
        """Generate insights from the analysis."""
        print("="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        insights = []
        
        # Class distribution
        class_dist = self.df[config.TARGET_COLUMN].value_counts(normalize=True)
        insights.append(f"Class distribution: {class_dist.to_dict()}")
        
        # Gender performance
        gender_performance = pd.crosstab(self.df['gender'], self.df[config.TARGET_COLUMN], normalize='index')
        insights.append(f"Gender performance patterns: {gender_performance.to_dict()}")
        
        # Top performing features
        numerical_features = [col for col in config.NUMERICAL_FEATURES if col in self.df.columns]
        for feature in numerical_features:
            high_performers = self.df[self.df[config.TARGET_COLUMN] == 'H'][feature].mean()
            low_performers = self.df[self.df[config.TARGET_COLUMN] == 'L'][feature].mean()
            insights.append(f"{feature}: High performers avg = {high_performers:.2f}, Low performers avg = {low_performers:.2f}")
            
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
            
        return insights
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Advanced Data Analysis...")
        
        # Basic information
        self.basic_info()
        
        # Statistical analysis
        self.statistical_analysis()
        
        # Visualizations
        self.create_comprehensive_visualizations()
        
        # Generate insights
        insights = self.generate_insights()
        
        print("\nAnalysis complete! Check the figures directory for visualizations.")
        return insights

if __name__ == "__main__":
    analyzer = AdvancedDataAnalyzer()
    analyzer.run_complete_analysis()
