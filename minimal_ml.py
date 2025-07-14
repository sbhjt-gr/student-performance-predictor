"""
Minimal working version of the student performance prediction system
Uses only built-in Python modules and creates basic functionality
"""

import csv
import json
import os
import random
import math
from collections import Counter, defaultdict

class MinimalConfig:
    """Minimal configuration class."""
    
    def __init__(self):
        self.DATA_PATH = "StudentPerformance.csv"
        self.OUTPUT_DIR = "outputs"
        self.MODELS_DIR = "models"
        self.FIGURES_DIR = "figures"
        self.RANDOM_STATE = 42
        self.TEST_SIZE = 0.2
        
        # Create directories
        for directory in [self.OUTPUT_DIR, self.MODELS_DIR, self.FIGURES_DIR]:
            os.makedirs(directory, exist_ok=True)

class MinimalDataAnalyzer:
    """Minimal data analyzer using only built-in modules."""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        
    def load_data(self):
        """Load data from CSV."""
        print("Loading data...")
        self.data = []
        with open(self.config.DATA_PATH, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append(row)
        print(f"Loaded {len(self.data)} records")
        return self.data
        
    def analyze_data(self):
        """Basic data analysis."""
        if not self.data:
            self.load_data()
            
        print("\n" + "="*50)
        print("DATA ANALYSIS")
        print("="*50)
        
        # Class distribution
        classes = [row['Class'] for row in self.data]
        class_counts = Counter(classes)
        
        print(f"Total students: {len(self.data)}")
        print(f"Features: {len(self.data[0])}")
        print("Class distribution:")
        for class_label, count in class_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {class_label}: {count} ({percentage:.1f}%)")
        
        # Numerical features analysis
        numerical_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
        
        print("\nNumerical feature statistics:")
        for feature in numerical_features:
            if feature in self.data[0]:
                values = [int(row[feature]) for row in self.data]
                avg = sum(values) / len(values)
                print(f"  {feature}: min={min(values)}, max={max(values)}, avg={avg:.1f}")
        
        return class_counts

class MinimalMLModel:
    """Minimal ML model using basic algorithms."""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def prepare_data(self):
        """Prepare data for modeling."""
        analyzer = MinimalDataAnalyzer(self.config)
        self.data = analyzer.load_data()
        
        # Split data
        random.seed(self.config.RANDOM_STATE)
        random.shuffle(self.data)
        split_idx = int(len(self.data) * (1 - self.config.TEST_SIZE))
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        
        print(f"Training set: {len(self.train_data)} samples")
        print(f"Test set: {len(self.test_data)} samples")
        
    def naive_bayes_classifier(self):
        """Simple Naive Bayes implementation."""
        if not self.train_data:
            self.prepare_data()
            
        print("\n" + "="*50)
        print("NAIVE BAYES CLASSIFIER")
        print("="*50)
        
        # Calculate class probabilities
        class_counts = Counter(row['Class'] for row in self.train_data)
        total_samples = len(self.train_data)
        class_probs = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Calculate feature probabilities for categorical features
        categorical_features = ['gender', 'NationalITy', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester']
        feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for feature in categorical_features:
            if feature in self.train_data[0]:
                for class_label in class_probs.keys():
                    class_samples = [row for row in self.train_data if row['Class'] == class_label]
                    feature_values = [row[feature] for row in class_samples]
                    value_counts = Counter(feature_values)
                    total_class_samples = len(class_samples)
                    
                    for value, count in value_counts.items():
                        feature_probs[feature][class_label][value] = count / total_class_samples
        
        # Make predictions
        correct_predictions = 0
        predictions = []
        
        for test_sample in self.test_data:
            class_scores = {}
            
            for class_label in class_probs.keys():
                score = math.log(class_probs[class_label])
                
                # Add feature contributions
                for feature in categorical_features:
                    if feature in test_sample:
                        feature_value = test_sample[feature]
                        if feature_value in feature_probs[feature][class_label]:
                            prob = feature_probs[feature][class_label][feature_value]
                            if prob > 0:
                                score += math.log(prob)
                
                class_scores[class_label] = score
            
            # Predict class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
            if predicted_class == test_sample['Class']:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(self.test_data)
        print(f"Naive Bayes accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy, predictions
        
    def rule_based_classifier(self):
        """Simple rule-based classifier."""
        if not self.train_data:
            self.prepare_data()
            
        print("\n" + "="*50)
        print("RULE-BASED CLASSIFIER")
        print("="*50)
        
        # Analyze patterns in training data
        high_performance_patterns = []
        medium_performance_patterns = []
        low_performance_patterns = []
        
        for row in self.train_data:
            engagement_score = (
                int(row['raisedhands']) + 
                int(row['VisITedResources']) + 
                int(row['AnnouncementsView']) + 
                int(row['Discussion'])
            )
            
            if row['Class'] == 'H':
                high_performance_patterns.append(engagement_score)
            elif row['Class'] == 'M':
                medium_performance_patterns.append(engagement_score)
            else:
                low_performance_patterns.append(engagement_score)
        
        # Calculate thresholds
        high_threshold = sum(high_performance_patterns) / len(high_performance_patterns) if high_performance_patterns else 200
        low_threshold = sum(low_performance_patterns) / len(low_performance_patterns) if low_performance_patterns else 100
        
        print(f"High performance threshold: {high_threshold:.1f}")
        print(f"Low performance threshold: {low_threshold:.1f}")
        
        # Make predictions
        correct_predictions = 0
        predictions = []
        
        for test_sample in self.test_data:
            engagement_score = (
                int(test_sample['raisedhands']) + 
                int(test_sample['VisITedResources']) + 
                int(test_sample['AnnouncementsView']) + 
                int(test_sample['Discussion'])
            )
            
            if engagement_score >= high_threshold:
                predicted_class = 'H'
            elif engagement_score <= low_threshold:
                predicted_class = 'L'
            else:
                predicted_class = 'M'
                
            predictions.append(predicted_class)
            
            if predicted_class == test_sample['Class']:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(self.test_data)
        print(f"Rule-based accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy, predictions
        
    def run_all_models(self):
        """Run all available models."""
        print("\n" + "="*60)
        print("RUNNING ALL MODELS")
        print("="*60)
        
        results = {}
        
        # Run Naive Bayes
        nb_accuracy, nb_predictions = self.naive_bayes_classifier()
        results['NaiveBayes'] = nb_accuracy
        
        # Run Rule-based
        rb_accuracy, rb_predictions = self.rule_based_classifier()
        results['RuleBased'] = rb_accuracy
        
        # Find best model
        best_model = max(results, key=results.get)
        best_accuracy = results[best_model]
        
        print(f"\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\nBest model: {best_model} with {best_accuracy:.3f} accuracy")
        
        return results, best_model

def main():
    """Main function to run the minimal ML pipeline."""
    print("="*60)
    print("MINIMAL STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    print("This version uses only built-in Python modules")
    print("="*60)
    
    # Initialize config
    config = MinimalConfig()
    
    # Check if data file exists
    if not os.path.exists(config.DATA_PATH):
        print(f"Error: Data file '{config.DATA_PATH}' not found!")
        return
    
    try:
        # Run data analysis
        analyzer = MinimalDataAnalyzer(config)
        analyzer.analyze_data()
        
        # Run ML models
        ml_model = MinimalMLModel(config)
        results, best_model = ml_model.run_all_models()
        
        # Save results
        results_path = os.path.join(config.OUTPUT_DIR, "minimal_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'model_results': results,
                'best_model': best_model,
                'best_accuracy': results[best_model]
            }, f, indent=2)
        
        print(f"\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {best_model}")
        print(f"Best accuracy: {results[best_model]:.3f}")
        print(f"Results saved to: {results_path}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
