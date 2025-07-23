import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load datasets and perform initial exploration"""
    
    print("=== TITANIC SURVIVAL PREDICTION - DATA EXPLORATION ===\n")
    
    # Load the datasets
    print("Loading datasets...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        gender_submission = pd.read_csv('gender_submission.csv')
        
        print(f"✓ Training data shape: {train_df.shape}")
        print(f"✓ Test data shape: {test_df.shape}")
        print(f"✓ Gender submission shape: {gender_submission.shape}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure train.csv, test.csv, and gender_submission.csv are in the same directory")
        return None, None, None
    
    # Display basic info about the dataset
    print("=== DATASET OVERVIEW ===")
    print("\nTraining Data Info:")
    print(train_df.info())
    
    print("\nColumn descriptions:")
    column_descriptions = {
        'PassengerId': 'Unique identifier for each passenger',
        'Survived': 'Target variable (0 = No, 1 = Yes)',
        'Pclass': 'Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)',
        'Name': 'Passenger name',
        'Sex': 'Gender (male/female)',
        'Age': 'Age in years',
        'SibSp': 'Number of siblings/spouses aboard',
        'Parch': 'Number of parents/children aboard',
        'Ticket': 'Ticket number',
        'Fare': 'Passenger fare',
        'Cabin': 'Cabin number',
        'Embarked': 'Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)'
    }
    
    for col, desc in column_descriptions.items():
        if col in train_df.columns:
            print(f"- {col}: {desc}")
    
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print("\nBasic Statistics:")
    print(train_df.describe())
    
    return train_df, test_df, gender_submission

def analyze_missing_values(train_df, test_df):
    """Analyze missing values in both datasets"""
    
    print("\n=== MISSING VALUES ANALYSIS ===")
    
    # Training data missing values
    missing_train = train_df.isnull().sum()
    missing_train_pct = (missing_train / len(train_df)) * 100
    
    print("Missing values in training data:")
    print("-" * 40)
    for col in train_df.columns:
        if missing_train[col] > 0:
            print(f"{col}: {missing_train[col]} ({missing_train_pct[col]:.1f}%)")
    
    # Test data missing values
    missing_test = test_df.isnull().sum()
    missing_test_pct = (missing_test / len(test_df)) * 100
    
    print("\nMissing values in test data:")
    print("-" * 40)
    for col in test_df.columns:
        if missing_test[col] > 0:
            print(f"{col}: {missing_test[col]} ({missing_test_pct[col]:.1f}%)")
    
    # Visualize missing values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training data
    missing_train_plot = missing_train[missing_train > 0]
    if len(missing_train_plot) > 0:
        axes[0].bar(missing_train_plot.index, missing_train_plot.values)
        axes[0].set_title('Missing Values - Training Data')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Test data
    missing_test_plot = missing_test[missing_test > 0]
    if len(missing_test_plot) > 0:
        axes[1].bar(missing_test_plot.index, missing_test_plot.values)
        axes[1].set_title('Missing Values - Test Data')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def basic_statistics(train_df):
    """Display basic statistics about survival"""
    
    print("\n=== BASIC SURVIVAL STATISTICS ===")
    
    # Overall survival rate
    survival_rate = train_df['Survived'].mean()
    total_passengers = len(train_df)
    survivors = train_df['Survived'].sum()
    deaths = total_passengers - survivors
    
    print(f"Total passengers: {total_passengers}")
    print(f"Survivors: {survivors}")
    print(f"Deaths: {deaths}")
    print(f"Overall survival rate: {survival_rate:.2%}")
    
    # Survival by key features
    print("\nSurvival rates by key features:")
    print("-" * 40)
    
    # By gender
    survival_by_sex = train_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
    print("By Gender:")
    for sex in survival_by_sex.index:
        count = survival_by_sex.loc[sex, 'count']
        survivors = survival_by_sex.loc[sex, 'sum']
        rate = survival_by_sex.loc[sex, 'mean']
        print(f"  {sex.capitalize()}: {survivors}/{count} ({rate:.2%})")
    
    # By class
    survival_by_class = train_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
    print("\nBy Class:")
    class_names = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
    for pclass in survival_by_class.index:
        count = survival_by_class.loc[pclass, 'count']
        survivors = survival_by_class.loc[pclass, 'sum']
        rate = survival_by_class.loc[pclass, 'mean']
        print(f"  {class_names[pclass]}: {survivors}/{count} ({rate:.2%})")
    
    # By embarked port
    survival_by_embarked = train_df.groupby('Embarked')['Survived'].agg(['count', 'sum', 'mean'])
    print("\nBy Embarked Port:")
    port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    for port in survival_by_embarked.index:
        if pd.notna(port):
            count = survival_by_embarked.loc[port, 'count']
            survivors = survival_by_embarked.loc[port, 'sum']
            rate = survival_by_embarked.loc[port, 'mean']
            print(f"  {port_names.get(port, port)}: {survivors}/{count} ({rate:.2%})")

def save_exploration_summary(train_df, test_df):
    """Save exploration summary to a text file"""
    
    with open('data_exploration_summary.txt', 'w') as f:
        f.write("TITANIC DATASET EXPLORATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Training data shape: {train_df.shape}\n")
        f.write(f"Test data shape: {test_df.shape}\n\n")
        
        f.write("Missing values in training data:\n")
        missing_train = train_df.isnull().sum()
        for col in train_df.columns:
            if missing_train[col] > 0:
                pct = (missing_train[col] / len(train_df)) * 100
                f.write(f"  {col}: {missing_train[col]} ({pct:.1f}%)\n")
        
        f.write("\nMissing values in test data:\n")
        missing_test = test_df.isnull().sum()
        for col in test_df.columns:
            if missing_test[col] > 0:
                pct = (missing_test[col] / len(test_df)) * 100
                f.write(f"  {col}: {missing_test[col]} ({pct:.1f}%)\n")
        
        f.write(f"\nOverall survival rate: {train_df['Survived'].mean():.2%}\n")
    
    print("\n✓ Exploration summary saved to 'data_exploration_summary.txt'")

if __name__ == "__main__":
    # Load and explore data
    train_df, test_df, gender_submission = load_and_explore_data()
    
    if train_df is not None:
        # Analyze missing values
        analyze_missing_values(train_df, test_df)
        
        # Basic statistics
        basic_statistics(train_df)
        
        # Save summary
        save_exploration_summary(train_df, test_df)
        
        print("\n=== DATA EXPLORATION COMPLETE ===")
        print("Next step: Run 2_eda_visualization.py for detailed visualizations")
    else:
        print("Data exploration failed. Please check your data files.")