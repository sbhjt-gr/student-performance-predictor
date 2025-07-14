"""
Main execution script for the advanced student performance prediction project.
Orchestrates the complete data science pipeline.
"""

import os
import sys
import warnings
from datetime import datetime
from config import config
from advanced_analysis import AdvancedDataAnalyzer
from feature_engineering import AdvancedFeatureEngineer
from advanced_ml import AdvancedMLModel

warnings.filterwarnings('ignore')

def print_banner():
    """Print project banner."""
    print("="*80)
    print("ADVANCED STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*80)
    print(f"Project: TianChi Student Performance Prediction")
    print(f"Version: 2.0 (Advanced)")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'plotly', 'xgboost', 'joblib', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All dependencies are installed!")
    return True

def run_data_analysis():
    """Run comprehensive data analysis."""
    print("\n" + "="*60)
    print("STEP 1: ADVANCED DATA ANALYSIS")
    print("="*60)
    
    try:
        analyzer = AdvancedDataAnalyzer()
        insights = analyzer.run_complete_analysis()
        print("Data analysis completed successfully!")
        return True
    except Exception as e:
        print(f"Data analysis failed: {str(e)}")
        return False

def run_feature_engineering():
    """Run advanced feature engineering."""
    print("\n" + "="*60)
    print("STEP 2: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    try:
        engineer = AdvancedFeatureEngineer()
        X, y, selected_features, stats = engineer.run_complete_feature_engineering()
        print("Feature engineering completed successfully!")
        return True, stats
    except Exception as e:
        print(f"Feature engineering failed: {str(e)}")
        return False, None

def run_machine_learning():
    """Run advanced machine learning pipeline."""
    print("\n" + "="*60)
    print("STEP 3: ADVANCED MACHINE LEARNING")
    print("="*60)
    
    try:
        ml_model = AdvancedMLModel()
        summary, best_model = ml_model.run_complete_ml_pipeline()
        print("Machine learning pipeline completed successfully!")
        return True, summary, best_model
    except Exception as e:
        print(f"Machine learning pipeline failed: {str(e)}")
        return False, None, None

def generate_final_report(ml_summary=None, best_model=None, feature_stats=None):
    """Generate final project report."""
    print("\n" + "="*60)
    print("FINAL PROJECT REPORT")
    print("="*60)
    
    report_path = f"{config.OUTPUT_DIR}/final_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("ADVANCED STUDENT PERFORMANCE PREDICTION - FINAL REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Project overview
        f.write("PROJECT OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write("This project implements an advanced machine learning pipeline for predicting student performance.\n")
        f.write("The system includes comprehensive data analysis, feature engineering, and model comparison.\n\n")
        
        # Feature engineering summary
        if feature_stats:
            f.write("FEATURE ENGINEERING SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Features: {feature_stats['total_features']}\n")
            f.write(f"Numerical Features: {feature_stats['numerical_features']}\n")
            f.write(f"Categorical Features: {feature_stats['categorical_features']}\n")
            f.write(f"Data Shape: {feature_stats['data_shape']}\n")
            f.write(f"Target Distribution: {feature_stats['target_distribution']}\n\n")
        
        # Model performance summary
        if ml_summary is not None:
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(ml_summary.to_string(index=False))
            f.write("\n\n")
        
        if best_model:
            f.write(f"BEST MODEL: {best_model}\n")
            f.write("-" * 20 + "\n")
            f.write("This model achieved the highest accuracy on the test set.\n\n")
        
        # File locations
        f.write("OUTPUT FILES:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Figures: {config.FIGURES_DIR}/\n")
        f.write(f"Models: {config.MODELS_DIR}/\n")
        f.write(f"Reports: {config.OUTPUT_DIR}/\n")
        
    print(f"Final report saved to: {report_path}")

def main():
    """Main execution function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists(config.DATA_PATH):
        print(f"Data file not found: {config.DATA_PATH}")
        sys.exit(1)
    
    print(f"Data file found: {config.DATA_PATH}")
    
    # Initialize results
    feature_stats = None
    ml_summary = None
    best_model = None
    
    # Step 1: Data Analysis
    if not run_data_analysis():
        print("Stopping due to data analysis failure")
        sys.exit(1)
    
    # Step 2: Feature Engineering
    success, feature_stats = run_feature_engineering()
    if not success:
        print("Stopping due to feature engineering failure")
        sys.exit(1)
    
    # Step 3: Machine Learning
    success, ml_summary, best_model = run_machine_learning()
    if not success:
        print("Stopping due to machine learning failure")
        sys.exit(1)
    
    # Generate final report
    generate_final_report(ml_summary, best_model, feature_stats)
    
    # Final message
    print("\n" + "="*80)
    print("ADVANCED STUDENT PERFORMANCE PREDICTION SYSTEM COMPLETED!")
    print("="*80)
    print(f"Best performing model: {best_model}")
    print(f"All outputs saved to respective directories")
    print(f"Check '{config.FIGURES_DIR}' for visualizations")
    print(f"Check '{config.MODELS_DIR}' for trained models")
    print(f"Check '{config.OUTPUT_DIR}' for reports")
    print("\nThank you for using the Advanced Student Performance Prediction System!")
    print("="*80)

if __name__ == "__main__":
    main()
