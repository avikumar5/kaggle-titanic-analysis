import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the original datasets"""
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        print(f"✓ Loaded training data: {train_df.shape}")
        print(f"✓ Loaded test data: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure train.csv and test.csv are in the same directory")
        return None, None

def extract_title(df):
    """Extract titles from passenger names"""
    print("Extracting titles from names...")
    
    # Extract title using regex
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Display title distribution before grouping
    print("Original title distribution:")
    print(df['Title'].value_counts())
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    
    print("\nGrouped title distribution:")
    print(df['Title'].value_counts())
    
    return df

def create_family_features(df):
    """Create family-related features"""
    print("Creating family features...")
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Family size categories
    df['FamilySizeGroup'] = 'Medium'
    df.loc[df['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    df.loc[df['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Large'
    
    print("Family size distribution:")
    print(df['FamilySize'].value_counts().sort_index())
    
    return df

def create_age_features(df):
    """Create age-related features"""
    print("Creating age features...")
    
    # Age groups
    df['AgeGroup'] = 'Adult'
    df.loc[df['Age'] <= 12, 'AgeGroup'] = 'Child'
    df.loc[(df['Age'] > 12) & (df['Age'] <= 18), 'AgeGroup'] = 'Teen'
    df.loc[(df['Age'] > 18) & (df['Age'] <= 35), 'AgeGroup'] = 'Young_Adult'
    df.loc[(df['Age'] > 35) & (df['Age'] <= 60), 'AgeGroup'] = 'Middle_Age'
    df.loc[df['Age'] > 60, 'AgeGroup'] = 'Senior'
    
    # Age bands (more granular)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 50, 65, 100], 
                          labels=['0-12', '13-18', '19-25', '26-35', '36-50', '51-65', '65+'])
    
    print("Age group distribution:")
    print(df['AgeGroup'].value_counts())
    
    return df

def create_fare_features(df):
    """Create fare-related features"""
    print("Creating fare features...")
    
    # Fare per person (in case of shared tickets)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Fare quartiles
    fare_quartiles = df['Fare'].quantile([0.25, 0.5, 0.75]).values
    df['FareGroup'] = 'Low'
    df.loc[df['Fare'] > fare_quartiles[0], 'FareGroup'] = 'Medium'
    df.loc[df['Fare'] > fare_quartiles[1], 'FareGroup'] = 'High'
    df.loc[df['Fare'] > fare_quartiles[2], 'FareGroup'] = 'Very_High'
    
    # Log fare (to handle skewness)
    df['LogFare'] = np.log1p(df['Fare'])
    
    print("Fare group distribution:")
    print(df['FareGroup'].value_counts())
    
    return df

def create_cabin_features(df):
    """Create cabin-related features"""
    print("Creating cabin features...")
    
    # Has cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Cabin deck (first letter)
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')
    
    # Number of cabins (count of spaces + 1)
    df['NumCabins'] = df['Cabin'].str.count(' ') + 1
    df.loc[df['Cabin'].isna(), 'NumCabins'] = 0
    
    print("Deck distribution:")
    print(df['Deck'].value_counts())
    
    return df

def create_ticket_features(df):
    """Create ticket-related features"""
    print("Creating ticket features...")
    
    # Ticket prefix (letters before numbers)
    df['TicketPrefix'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False)
    df['TicketPrefix'] = df['TicketPrefix'].fillna('None')
    
    # Ticket length
    df['TicketLength'] = df['Ticket'].str.len()
    
    # Has ticket prefix
    df['HasTicketPrefix'] = (df['TicketPrefix'] != 'None').astype(int)
    
    # Group ticket prefixes (keep common ones, group rare ones)
    ticket_counts = df['TicketPrefix'].value_counts()
    common_prefixes = ticket_counts[ticket_counts >= 10].index
    df['TicketPrefixGroup'] = df['TicketPrefix'].apply(
        lambda x: x if x in common_prefixes else 'Rare'
    )
    
    print("Common ticket prefixes:")
    print(df['TicketPrefixGroup'].value_counts())
    
    return df

def handle_missing_values(train_df, test_df):
    """Handle missing values intelligently"""
    print("\n=== HANDLING MISSING VALUES ===")
    
    # Combine datasets for consistent preprocessing
    all_data = [train_df, test_df]
    
    for dataset in all_data:
        # Age: Fill based on Title and Pclass
        print("Filling missing Age values...")
        for title in dataset['Title'].unique():
            for pclass in dataset['Pclass'].unique():
                mask = (dataset['Title'] == title) & (dataset['Pclass'] == pclass)
                age_median = dataset[mask]['Age'].median()
                
                # If no median for this combination, use overall median for the title
                if pd.isna(age_median):
                    age_median = dataset[dataset['Title'] == title]['Age'].median()
                
                # If still no median, use overall median
                if pd.isna(age_median):
                    age_median = dataset['Age'].median()
                
                dataset.loc[mask & dataset['Age'].isna(), 'Age'] = age_median
        
        # Embarked: Fill with mode
        mode_embarked = dataset['Embarked'].mode()[0]
        dataset['Embarked'].fillna(mode_embarked, inplace=True)
        
        # Fare: Fill with median based on Pclass
        if 'Fare' in dataset.columns:
            for pclass in dataset['Pclass'].unique():
                mask = dataset['Pclass'] == pclass
                fare_median = dataset[mask]['Fare'].median()
                dataset.loc[mask & dataset['Fare'].isna(), 'Fare'] = fare_median
        
        # Cabin: Already handled by creating HasCabin feature
        
        print(f"Remaining missing values: {dataset.isnull().sum().sum()}")
    
    return train_df, test_df

def encode_categorical_features(train_df, test_df):
    """Encode categorical features"""
    print("\n=== ENCODING CATEGORICAL FEATURES ===")
    
    categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 
                           'Deck', 'FamilySizeGroup', 'TicketPrefixGroup', 'AgeBand']
    
    # Use label encoding for ordinal features and one-hot for nominal
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in train_df.columns:
            print(f"Encoding {feature}...")
            
            # Combine unique values from both datasets
            all_values = list(train_df[feature].unique()) + list(test_df[feature].unique())
            unique_values = list(set(all_values))
            
            # Create label encoder
            le = LabelEncoder()
            le.fit(unique_values)
            
            # Transform both datasets
            train_df[f'{feature}_Encoded'] = le.transform(train_df[feature])
            test_df[f'{feature}_Encoded'] = le.transform(test_df[feature])
            
            label_encoders[feature] = le
    
    return train_df, test_df, label_encoders

def create_interaction_features(df):
    """Create interaction features"""
    print("Creating interaction features...")
    
    # Age * Class interaction
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    
    # Fare * Class interaction
    df['Fare_Pclass'] = df['Fare'] * df['Pclass']
    
    # Gender * Class interaction
    df['Sex_Pclass'] = df['Sex_Encoded'] * df['Pclass']
    
    # Family size * Class interaction
    df['FamilySize_Pclass'] = df['FamilySize'] * df['Pclass']
    
    # Age * Gender interaction
    df['Age_Sex'] = df['Age'] * df['Sex_Encoded']
    
    return df

def feature_selection_analysis(train_df):
    """Analyze feature importance"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get numeric features for correlation analysis
    numeric_features = train_df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != 'PassengerId']
    
    # Calculate correlation with target
    if 'Survived' in train_df.columns:
        correlations = train_df[numeric_features].corr()['Survived'].abs().sort_values(ascending=False)
        
        print("Feature correlations with Survival (top 15):")
        print("-" * 50)
        for feature, corr in correlations.head(15).items():
            if feature != 'Survived':
                print(f"{feature:25s}: {corr:.4f}")
        
        # Visualize top correlations
        plt.figure(figsize=(12, 8))
        top_features = correlations.drop('Survived').head(15)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Absolute Correlation with Survival')
        plt.title('Top 15 Features by Correlation with Survival')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Feature importance plot saved as 'feature_importance.png'")

def save_processed_data(train_df, test_df, label_encoders):
    """Save processed datasets and encoders"""
    print("\n=== SAVING PROCESSED DATA ===")
    
    # Save processed datasets
    train_df.to_csv('train_processed.csv', index=False)
    test_df.to_csv('test_processed.csv', index=False)
    
    # Save feature list
    feature_list = list(train_df.columns)
    with open('feature_list.txt', 'w') as f:
        f.write("PROCESSED FEATURES LIST\n")
        f.write("="*30 + "\n\n")
        
        f.write("Original Features:\n")
        original_features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                           'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        for feature in original_features:
            if feature in feature_list:
                f.write(f"  ✓ {feature}\n")
        
        f.write("\nEngineered Features:\n")
        engineered_features = [f for f in feature_list if f not in original_features]
        for feature in engineered_features:
            f.write(f"  + {feature}\n")
        
        f.write(f"\nTotal features: {len(feature_list)}\n")
    
    # Save encoding information
    with open('encoders_info.txt', 'w') as f:
        f.write("CATEGORICAL ENCODING INFORMATION\n")
        f.write("="*40 + "\n\n")
        
        for feature, encoder in label_encoders.items():
            f.write(f"{feature}:\n")
            for i, class_name in enumerate(encoder.classes_):
                f.write(f"  {class_name} -> {i}\n")
            f.write("\n")
    
    print("✓ Processed datasets saved:")
    print("  - train_processed.csv")
    print("  - test_processed.csv")
    print("  - feature_list.txt")
    print("  - encoders_info.txt")
    
    print(f"\nFinal dataset shapes:")
    print(f"  Training: {train_df.shape}")
    print(f"  Test: {test_df.shape}")

def feature_engineering_summary(train_df):
    """Generate feature engineering summary"""
    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    
    original_features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                        'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    current_features = list(train_df.columns)
    new_features = [f for f in current_features if f not in original_features]
    
    print(f"Original features: {len(original_features)}")
    print(f"Total features after engineering: {len(current_features)}")
    print(f"New features created: {len(new_features)}")
    
    print("\nNew features by category:")
    
    categories = {
        'Title': [f for f in new_features if 'title' in f.lower()],
        'Family': [f for f in new_features if any(x in f.lower() for x in ['family', 'alone'])],
        'Age': [f for f in new_features if 'age' in f.lower()],
        'Fare': [f for f in new_features if 'fare' in f.lower()],
        'Cabin': [f for f in new_features if any(x in f.lower() for x in ['cabin', 'deck'])],
        'Ticket': [f for f in new_features if 'ticket' in f.lower()],
        'Encoded': [f for f in new_features if 'encoded' in f.lower()],
        'Interaction': [f for f in new_features if '_' in f and any(x in f.lower() for x in ['pclass', 'sex'])],
        'Other': []
    }
    
    # Categorize remaining features
    categorized = []
    for cat_features in categories.values():
        categorized.extend(cat_features)
    
    categories['Other'] = [f for f in new_features if f not in categorized]
    
    for category, features in categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
            for feature in features[:3]:  # Show first 3
                print(f"    - {feature}")
            if len(features) > 3:
                print(f"    ... and {len(features)-3} more")

if __name__ == "__main__":
    print("=== TITANIC FEATURE ENGINEERING ===\n")
    
    # Load data
    train_df, test_df = load_data()
    
    if train_df is not None and test_df is not None:
        # Store original shapes
        orig_train_shape = train_df.shape
        orig_test_shape = test_df.shape
        
        print(f"Starting with {orig_train_shape[1]} features")
        
        # Apply feature engineering
        print("\n1. Extracting titles...")
        train_df = extract_title(train_df)
        test_df = extract_title(test_df)
        
        print("\n2. Creating family features...")
        train_df = create_family_features(train_df)
        test_df = create_family_features(test_df)
        
        print("\n3. Creating age features...")
        train_df = create_age_features(train_df)
        test_df = create_age_features(test_df)
        
        print("\n4. Creating fare features...")
        train_df = create_fare_features(train_df)
        test_df = create_fare_features(test_df)
        
        print("\n5. Creating cabin features...")
        train_df = create_cabin_features(train_df)
        test_df = create_cabin_features(test_df)
        
        print("\n6. Creating ticket features...")
        train_df = create_ticket_features(train_df)
        test_df = create_ticket_features(test_df)
        
        # Handle missing values
        train_df, test_df = handle_missing_values(train_df, test_df)
        
        # Encode categorical features
        train_df, test_df, label_encoders = encode_categorical_features(train_df, test_df)
        
        print("\n7. Creating interaction features...")
        train_df = create_interaction_features(train_df)
        test_df = create_interaction_features(test_df)
        
        # Feature importance analysis
        feature_selection_analysis(train_df)
        
        # Feature engineering summary
        feature_engineering_summary(train_df)
        
        # Save processed data
        save_processed_data(train_df, test_df, label_encoders)
        
        print("\n=== FEATURE ENGINEERING COMPLETE ===")
        print("Next step: Run 4_model_training.py")
        
    else:
        print("Feature engineering failed. Please check your data files.")