"""
Comprehensive model performance analysis and visualization script.
This script analyzes and visualizes the performance of machine learning models
for student performance prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json

warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    """Class to analyze and visualize model performance."""
    
    def __init__(self, data_path='StudentPerformance.csv'):
        """Initialize the analyzer."""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading and preparing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        # Encode categorical variables
        label_encoders = {}
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                label_encoders[column] = le
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Target distribution: {self.data['Class'].value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models."""
        print("Training models...")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f}")
        
        return self.results
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations."""
        print("Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model Comparison - Accuracy
        plt.subplot(3, 3, 1)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        plt.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        plt.errorbar(x + width/2, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Metrics Comparison Heatmap
        plt.subplot(3, 3, 2)
        metrics_data = []
        for name in model_names:
            metrics_data.append([
                self.results[name]['accuracy'],
                self.results[name]['precision'],
                self.results[name]['recall'],
                self.results[name]['f1_score']
            ])
        
        metrics_df = pd.DataFrame(metrics_data, 
                                 index=model_names, 
                                 columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Score'})
        plt.title('Model Performance Metrics Heatmap')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 3. Best Model Confusion Matrix
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        plt.subplot(3, 3, 3)
        cm = best_result['confusion_matrix']
        class_names = ['Low', 'Medium', 'High']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 4. Predictions vs Actual for Best Model
        plt.subplot(3, 3, 4)
        y_test_array = np.array(self.y_test)
        y_pred_array = best_result['predictions']
        
        # Create scatter plot
        plt.scatter(y_test_array, y_pred_array, alpha=0.6)
        plt.plot([y_test_array.min(), y_test_array.max()], 
                [y_test_array.min(), y_test_array.max()], 'r--', lw=2)
        plt.xlabel('Actual Performance')
        plt.ylabel('Predicted Performance')
        plt.title(f'Predictions vs Actual - {best_model_name}')
        
        # Add perfect prediction line
        unique_classes = np.unique(y_test_array)
        for cls in unique_classes:
            plt.axhline(y=cls, color='gray', linestyle='--', alpha=0.3)
            plt.axvline(x=cls, color='gray', linestyle='--', alpha=0.3)
        
        # 5. Feature Importance (for Random Forest)
        if 'Random Forest' in self.results:
            plt.subplot(3, 3, 5)
            rf_model = self.results['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = self.X_train.columns
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
            
            plt.barh(range(len(indices)), feature_importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
        
        # 6. Model Performance Distribution
        plt.subplot(3, 3, 6)
        performance_data = []
        model_labels = []
        
        for name in model_names:
            # Get cross-validation scores
            cv_scores = cross_val_score(self.results[name]['model'], 
                                      self.X_train, self.y_train, cv=5, scoring='accuracy')
            performance_data.extend(cv_scores)
            model_labels.extend([name] * len(cv_scores))
        
        performance_df = pd.DataFrame({'Model': model_labels, 'CV_Score': performance_data})
        sns.boxplot(data=performance_df, x='Model', y='CV_Score')
        plt.xticks(rotation=45)
        plt.title('Cross-Validation Score Distribution')
        plt.ylabel('CV Accuracy Score')
        
        # 7. Prediction Confidence (for models with predict_proba)
        plt.subplot(3, 3, 7)
        if best_result['probabilities'] is not None:
            max_proba = np.max(best_result['probabilities'], axis=1)
            correct_predictions = (best_result['predictions'] == self.y_test).astype(int)
            
            plt.scatter(max_proba, correct_predictions, alpha=0.6)
            plt.xlabel('Prediction Confidence (Max Probability)')
            plt.ylabel('Correct Prediction (1=Correct, 0=Wrong)')
            plt.title(f'Prediction Confidence vs Accuracy - {best_model_name}')
            plt.yticks([0, 1], ['Wrong', 'Correct'])
        
        # 8. Error Analysis
        plt.subplot(3, 3, 8)
        errors = best_result['predictions'] != self.y_test
        error_counts = pd.Series(errors).value_counts()
        
        plt.pie(error_counts.values, labels=['Correct', 'Incorrect'], 
                autopct='%1.1f%%', startangle=90)
        plt.title(f'Prediction Accuracy Distribution - {best_model_name}')
        
        # 9. Model Comparison Summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Create summary text
        summary_text = f"MODEL PERFORMANCE SUMMARY\n\n"
        summary_text += f"Best Model: {best_model_name}\n"
        summary_text += f"Best Accuracy: {best_result['accuracy']:.4f}\n\n"
        
        summary_text += "All Models Performance:\n"
        for name in model_names:
            result = self.results[name]
            summary_text += f"{name}:\n"
            summary_text += f"  Accuracy: {result['accuracy']:.4f}\n"
            summary_text += f"  F1-Score: {result['f1_score']:.4f}\n"
            summary_text += f"  CV Mean: {result['cv_mean']:.4f}\n\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model_name, best_result
    
    def save_results(self, filename='model_results.json'):
        """Save results to JSON file."""
        results_summary = {}
        
        for name, result in self.results.items():
            results_summary[name] = {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std'])
            }
        
        # Add best model info
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        results_summary['best_model'] = best_model
        results_summary['best_accuracy'] = float(self.results[best_model]['accuracy'])
        
        with open(filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to {filename}")
        return results_summary
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("="*60)
        print("STUDENT PERFORMANCE PREDICTION - MODEL ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models()
        
        # Create visualizations
        best_model_name, best_result = self.create_performance_visualizations()
        
        # Save results
        results_summary = self.save_results()
        
        # Print final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_result['accuracy']:.4f}")
        print(f"Best F1-Score: {best_result['f1_score']:.4f}")
        print(f"Best CV Score: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})")
        
        return results_summary, best_model_name, best_result

def main():
    """Main function to run the analysis."""
    analyzer = ModelPerformanceAnalyzer()
    results, best_model, best_result = analyzer.run_complete_analysis()
    
    print("\nVisualization saved as 'model_performance_analysis.png'")
    print("Results saved as 'model_results.json'")

if __name__ == "__main__":
    main()
