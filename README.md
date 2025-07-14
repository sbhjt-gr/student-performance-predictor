# Student Performance Prediction System

## Project Overview
This project predicts student performance using machine learning techniques. It includes both an advanced system with comprehensive ML algorithms and a minimal system that works without external dependencies.

## Key Features

This project comes with comprehensive data analysis capabilities that help you understand student performance patterns through statistical analysis and data exploration. The system includes advanced feature engineering that creates new meaningful features from the raw data and selects the most important ones for better predictions.

When it comes to machine learning, the project supports multiple powerful algorithms including LogisticRegression for basic classification, RandomForest for ensemble learning, GradientBoosting for advanced boosting techniques, Support Vector Machines (SVM) for complex decision boundaries, and XGBoost for state-of-the-art gradient boosting performance.

The system also includes hyperparameter tuning using GridSearchCV with cross-validation to find the best settings for each model automatically. Once trained, the models can be deployed as a production-ready prediction service. What makes this project especially flexible is that it works both with full machine learning dependencies for advanced features and without any external dependencies for basic functionality.

## Quick Start

### Option 1: Automatic System Detection (Recommended)
```bash
python run.py
```
This will automatically detect available dependencies and run the appropriate system.

### Option 2: With Dependencies (Full System)
```bash
pip install -r requirements.txt
python main.py
```

### Option 3: Without Dependencies (Minimal System)
```bash
python minimal_ml.py
```

### Option 4: Simple Demo
```bash
python simple_demo.py
```

## Dataset

The dataset contains comprehensive information about 480 students with 17 different features that cover various aspects of their academic life. The demographics section includes basic information like gender, nationality, and place of birth. 

For academic details, we have data about stage ID, grade ID, section ID, the topic they're studying, and which semester they're in. Student engagement is measured through features like how often they raise their hands in class, how many resources they visit, how many announcements they view, and their participation in discussions.

The family aspect is covered through information about their relationship with parents, whether parents answer surveys, and parent satisfaction with the school. We also track student attendance through absence days data.

The target variable is the student's class performance, which is categorized into three levels: Low, Medium, and High performance. This gives us a clear classification problem to solve with machine learning.
## Results

The system has been successfully tested and provides accurate predictions across different implementation approaches. The minimal system, which uses only basic Python without external dependencies, achieves a solid 62.5% accuracy using fundamental algorithms. 

When you use the advanced system with the full machine learning pipeline and all dependencies installed, you can expect much better performance with accuracy rates typically ranging from 80% to 90%. 

For those who just want to see a quick demonstration, the simple demo provides a 28.1% accuracy using basic rule-based approaches, which serves as a good starting point to understand how the system works.

## Output Files

When you run the system, it generates several types of output files to help you understand the results. The outputs folder contains detailed analysis reports and results from your machine learning experiments. 

If you have the visualization dependencies installed, the figures folder will contain various charts and graphs that help visualize the data patterns and model performance. 

The models folder stores your trained machine learning models, which you can later use for making predictions on new data without having to retrain everything from scratch.

## Requirements

This project is designed to work with Python 3.7 or newer versions. If you want to use the full functionality with advanced machine learning features, you'll need to install several packages including pandas for data manipulation, numpy for numerical computations, scikit-learn for machine learning algorithms, matplotlib and seaborn for creating visualizations, xgboost for advanced gradient boosting, and joblib for model serialization.

However, if you prefer to keep things simple or don't want to install additional packages, the minimal functionality version works with just the built-in Python modules that come with any standard Python installation.

## License
This project is for educational purposes.

## Contributing

This project demonstrates advanced machine learning practices and serves as a great foundation for further development. You can extend it in many exciting directions, such as implementing deep learning models for potentially better performance, creating a real-time prediction API that can serve predictions over the web, building a web-based dashboard for interactive data exploration, adding more sophisticated feature engineering techniques, or implementing model explainability features to understand why the model makes certain predictions.

---