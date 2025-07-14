"""
Simple student performance prediction demo.
Works with minimal dependencies - only requires built-in Python modules.
"""

import csv
import random
import os
from collections import Counter

def load_data():
    """Load data from CSV file."""
    print("Loading student performance data...")
    
    data = []
    with open('StudentPerformance.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    print(f"Loaded {len(data)} student records")
    return data

def analyze_data(data):
    """Basic data analysis."""
    print("\n" + "="*50)
    print("BASIC DATA ANALYSIS")
    print("="*50)
    
    # Count classes
    classes = [row['Class'] for row in data]
    class_counts = Counter(classes)
    
    print(f"Total students: {len(data)}")
    print(f"Features available: {len(data[0])} columns")
    print("\nClass distribution:")
    for class_label, count in class_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {class_label}: {count} students ({percentage:.1f}%)")
    
    # Basic statistics on numerical features
    numerical_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
    
    print("\nNumerical feature statistics:")
    for feature in numerical_features:
        if feature in data[0]:
            values = [int(row[feature]) for row in data]
            print(f"  {feature}: min={min(values)}, max={max(values)}, avg={sum(values)/len(values):.1f}")
    
    return class_counts

def simple_prediction(data):
    """Simple prediction based on basic rules."""
    print("\n" + "="*50)
    print("SIMPLE PREDICTION MODEL")
    print("="*50)
    
    # Split data into train and test (80/20)
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Training set: {len(train_data)} students")
    print(f"Test set: {len(test_data)} students")
    
    # Simple rule-based prediction
    # High engagement (raised hands + resources + announcements) -> Better performance
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for student in test_data:
        # Calculate engagement score
        engagement_score = (
            int(student['raisedhands']) + 
            int(student['VisITedResources']) + 
            int(student['AnnouncementsView']) + 
            int(student['Discussion'])
        )
        
        # Simple rule: High engagement -> High performance
        if engagement_score > 60:
            predicted_class = 'H'  # High
        elif engagement_score > 30:
            predicted_class = 'M'  # Medium
        else:
            predicted_class = 'L'  # Low
        
        # Check if prediction is correct
        if predicted_class == student['Class']:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"\nSimple rule-based model accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return accuracy

def main():
    """Main demonstration function."""
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION - SIMPLE DEMO")
    print("="*60)
    print("This demo runs without external dependencies")
    print("It demonstrates basic data analysis and prediction")
    print("="*60)
    
    try:
        # Load and analyze data
        data = load_data()
        class_counts = analyze_data(data)
        
        # Simple prediction
        accuracy = simple_prediction(data)
        
        print("\n" + "="*50)
        print("DEMO SUMMARY")
        print("="*50)
        print(f"✓ Successfully loaded {len(data)} student records")
        print(f"✓ Analyzed basic data distribution")
        print(f"✓ Implemented simple rule-based prediction")
        print(f"✓ Achieved {accuracy*100:.1f}% accuracy with basic rules")
        print("\nNote: This is a simplified version for demonstration.")
        print("The full advanced system would include additional features.")
        
    except FileNotFoundError:
        print("Error: StudentPerformance.csv not found!")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
