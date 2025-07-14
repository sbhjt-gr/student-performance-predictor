#!/usr/bin/env python3
"""
Unified entry point for the Student Performance Prediction System.
Automatically detects available dependencies and runs the appropriate version.
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'xgboost', 'joblib']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def main():
    """Main entry point."""
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    
    # Check if data file exists
    if not os.path.exists('StudentPerformance.csv'):
        print("Error: StudentPerformance.csv not found!")
        print("Please ensure the data file is in the current directory.")
        sys.exit(1)
    
    # Check dependencies
    has_deps, missing = check_dependencies()
    
    if has_deps:
        print("All dependencies found! Running advanced system...")
        try:
            from main import main as advanced_main
            advanced_main()
        except Exception as e:
            print(f"Error running advanced system: {e}")
            print("Falling back to minimal system...")
            run_minimal_system()
    else:
        print("Missing dependencies. Running minimal system...")
        print(f"Missing packages: {', '.join(missing)}")
        print("For full functionality, install: pip install -r requirements.txt")
        print("="*60)
        run_minimal_system()

def run_minimal_system():
    """Run the minimal system that doesn't require external dependencies."""
    try:
        from minimal_ml import main as minimal_main
        minimal_main()
    except Exception as e:
        print(f"Error running minimal system: {e}")
        print("Trying simple demo...")
        try:
            from simple_demo import main as demo_main
            demo_main()
        except Exception as e2:
            print(f"Error running demo: {e2}")
            print("All systems failed. Please check your installation.")

if __name__ == "__main__":
    main()
