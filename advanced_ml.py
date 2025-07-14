"""
Advanced machine learning module with multiple algorithms and hyperparameter tuning.
Provides comprehensive model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from config import config
from feature_engineering import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')

class AdvancedMLModel:
    """Advanced machine learning model class with multiple algorithms."""
    
    def __init__(self, data_path: str = None):
        """Initialize the ML model."""
        self.data_path = data_path or config.DATA_PATH
        self.feature_engineer = AdvancedFeatureEngineer(data_path)
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """Prepare data for machine learning."""
        print("Preparing data for machine learning...")
        
        # Run feature engineering
        X, y, selected_features, stats = self.feature_engineer.run_complete_feature_engineering()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def initialize_models(self):
        """Initialize all machine learning models."""
        print("Initializing models...")
        
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=config.RANDOM_STATE),
            'GradientBoosting': GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            'SVM': SVC(random_state=config.RANDOM_STATE),
            'XGBoost': xgb.XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='mlogloss')
        }\n        print(f"Initialized {len(self.models)} models")
        
    def train_models_with_grid_search(self):
        """Train all models with hyperparameter tuning."""
        if self.X_train is None:\n            self.prepare_data()
            
        if not self.models:
            self.initialize_models()
            
        print("Training models with hyperparameter tuning...")
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        for model_name, model in self.models.items():
            print(f\"\\nTraining {model_name}...\")
            
            # Get hyperparameter grid
            param_grid = config.MODELS_CONFIG[model_name]['param_grid']
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            # Fit grid search
            grid_search.fit(self.X_train, self.y_train)
            
            # Store best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            # Store results
            self.results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f\"Best parameters for {model_name}: {grid_search.best_params_}\")
            print(f\"Best cross-validation score: {grid_search.best_score_:.4f}\")
            
        print(\"\\nAll models trained successfully!\")
        
    def evaluate_models(self):
        \"\"\"Evaluate all trained models.\"\"\"
        if not self.best_models:
            self.train_models_with_grid_search()
            
        print(\"Evaluating models...\")
        
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            print(f\"\\nEvaluating {model_name}...\")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'classification_report': classification_report(self.y_test, y_pred, 
                                                             target_names=config.TARGET_LABELS)
            }
            
            print(f\"Accuracy: {accuracy:.4f}\")
            print(f\"Precision: {precision:.4f}\")
            print(f\"Recall: {recall:.4f}\")
            print(f\"F1-score: {f1:.4f}\")
            print(f\"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\")
            
        self.evaluation_results = evaluation_results
        return evaluation_results
        
    def create_model_comparison_plots(self):
        \"\"\"Create comprehensive model comparison visualizations.\"\"\"
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_models()
            
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            values = [self.evaluation_results[model][metric] for model in models]
            
            sns.barplot(x=models, y=values, ax=axes[row, col])
            axes[row, col].set_title(f'{metric.replace(\"_\", \" \").title()}')
            axes[row, col].set_ylim(0, 1)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/model_comparison.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
        # Cross-validation scores comparison
        plt.figure(figsize=(12, 6))
        cv_means = [self.evaluation_results[model]['cv_mean'] for model in models]
        cv_stds = [self.evaluation_results[model]['cv_std'] for model in models]
        
        plt.bar(models, cv_means, yerr=cv_stds, capsize=5)
        plt.title('Cross-Validation Scores Comparison')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/cv_scores_comparison.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
        # Confusion matrices
        n_models = len(models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, model_name in enumerate(models):
            y_pred = self.evaluation_results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=config.TARGET_LABELS,
                       yticklabels=config.TARGET_LABELS,
                       ax=axes[i])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
        # Hide empty subplot
        if len(models) < len(axes):
            axes[-1].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/confusion_matrices.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def get_feature_importance(self):
        \"\"\"Get feature importance for tree-based models.\"\"\"
        feature_importance = {}
        
        for model_name, model in self.best_models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = self.feature_engineer.feature_names
                
                # Create DataFrame for easier handling
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                feature_importance[model_name] = importance_df
                
        return feature_importance
        
    def plot_feature_importance(self):
        \"\"\"Plot feature importance for applicable models.\"\"\"
        feature_importance = self.get_feature_importance()
        
        if not feature_importance:
            print(\"No models with feature importance available.\")
            return
            
        n_models = len(feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
            
        fig.suptitle('Feature Importance', fontsize=16, fontweight='bold')
        
        for i, (model_name, importance_df) in enumerate(feature_importance.items()):
            # Plot top 15 features
            top_features = importance_df.head(15)
            
            sns.barplot(data=top_features, x='importance', y='feature', ax=axes[i])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Importance')
            
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_DIR}/feature_importance.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()
        
    def select_best_model(self):
        \"\"\"Select the best performing model.\"\"\"
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_models()
            
        # Find best model based on accuracy
        best_model_name = max(self.evaluation_results, 
                             key=lambda x: self.evaluation_results[x]['accuracy'])
        
        best_model = self.best_models[best_model_name]
        best_score = self.evaluation_results[best_model_name]['accuracy']
        
        print(f\"\\nBest model: {best_model_name}\")
        print(f\"Best accuracy: {best_score:.4f}\")
        
        return best_model_name, best_model, best_score
        
    def save_models(self):
        \"\"\"Save all trained models.\"\"\"
        print(\"Saving models...\")
        
        for model_name, model in self.best_models.items():
            joblib.dump(model, f'{config.MODELS_DIR}/{model_name}_model.pkl')
            
        # Save evaluation results
        joblib.dump(self.evaluation_results, f'{config.MODELS_DIR}/evaluation_results.pkl')
        
        print(\"Models saved successfully!\")
        
    def generate_model_report(self):
        \"\"\"Generate comprehensive model report.\"\"\"
        if not hasattr(self, 'evaluation_results'):
            self.evaluate_models()
            
        print(\"=\"*80)
        print(\"COMPREHENSIVE MODEL REPORT\")
        print(\"=\"*80)
        
        # Summary table
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f\"{results['accuracy']:.4f}\",
                'Precision': f\"{results['precision']:.4f}\",
                'Recall': f\"{results['recall']:.4f}\",
                'F1-Score': f\"{results['f1_score']:.4f}\",
                'CV Score': f\"{results['cv_mean']:.4f}±{results['cv_std']:.4f}\"
            })
            
        summary_df = pd.DataFrame(summary_data)
        print(\"\\nMODEL PERFORMANCE SUMMARY:\")
        print(summary_df.to_string(index=False))
        
        # Best model details
        best_model_name, best_model, best_score = self.select_best_model()
        
        print(f\"\\nBEST MODEL DETAILS:\")
        print(f\"Model: {best_model_name}\")
        print(f\"Accuracy: {best_score:.4f}\")
        print(f\"Parameters: {self.results[best_model_name]['best_params']}\")
        
        print(f\"\\nClassification Report for {best_model_name}:\")
        print(self.evaluation_results[best_model_name]['classification_report'])
        
        return summary_df, best_model_name
        
    def run_complete_ml_pipeline(self):
        \"\"\"Run the complete machine learning pipeline.\"\"\"
        print(\"Starting Advanced Machine Learning Pipeline...\")
        
        # Prepare data
        self.prepare_data()
        
        # Train models
        self.train_models_with_grid_search()
        
        # Evaluate models
        self.evaluate_models()
        
        # Create visualizations
        self.create_model_comparison_plots()
        self.plot_feature_importance()
        
        # Generate report
        summary_df, best_model_name = self.generate_model_report()
        
        # Save models
        self.save_models()
        
        print(\"\\nMachine Learning Pipeline Complete!\")
        print(f\"Best model: {best_model_name}\")
        print(f\"Check '{config.FIGURES_DIR}' for visualizations\")
        print(f\"Check '{config.MODELS_DIR}' for saved models\")
        
        return summary_df, best_model_name

if __name__ == \"__main__\":
    ml_model = AdvancedMLModel()
    summary, best_model = ml_model.run_complete_ml_pipeline()
    print(\"Advanced ML pipeline complete!\")
