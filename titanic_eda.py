import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the preprocessed data"""
    try:
        train_df = pd.read_csv('train.csv')
        print(f"✓ Loaded training data: {train_df.shape}")
        return train_df
    except FileNotFoundError:
        print("Error: train.csv not found. Make sure the file is in the current directory.")
        return None

def survival_analysis_plots(train_df):
    """Create comprehensive survival analysis plots"""
    
    print("\n=== CREATING SURVIVAL ANALYSIS PLOTS ===")
    
    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Titanic Survival Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')
    
    # 1. Overall survival distribution
    survival_counts = train_df['Survived'].value_counts()
    axes[0, 0].pie(survival_counts.values, labels=['Died', 'Survived'], autopct='%1.1f%%', 
                   colors=['#ff6b6b', '#4ecdc4'])
    axes[0, 0].set_title('Overall Survival Distribution')
    
    # 2. Survival by Gender
    survival_by_sex = pd.crosstab(train_df['Sex'], train_df['Survived'])
    survival_by_sex.plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 1].set_title('Survival by Gender')
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend(['Died', 'Survived'])
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Survival by Class
    survival_by_class = pd.crosstab(train_df['Pclass'], train_df['Survived'])
    survival_by_class.plot(kind='bar', ax=axes[0, 2], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 2].set_title('Survival by Passenger Class')
    axes[0, 2].set_xlabel('Class')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].legend(['Died', 'Survived'])
    axes[0, 2].tick_params(axis='x', rotation=0)
    
    # 4. Age distribution by survival
    survived = train_df[train_df['Survived'] == 1]['Age'].dropna()
    died = train_df[train_df['Survived'] == 0]['Age'].dropna()
    axes[1, 0].hist([died, survived], bins=30, alpha=0.7, label=['Died', 'Survived'], 
                    color=['#ff6b6b', '#4ecdc4'])
    axes[1, 0].set_title('Age Distribution by Survival')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 5. Fare distribution by survival
    survived_fare = train_df[train_df['Survived'] == 1]['Fare'].dropna()
    died_fare = train_df[train_df['Survived'] == 0]['Fare'].dropna()
    axes[1, 1].hist([died_fare, survived_fare], bins=30, alpha=0.7, label=['Died', 'Survived'],
                    color=['#ff6b6b', '#4ecdc4'])
    axes[1, 1].set_title('Fare Distribution by Survival')
    axes[1, 1].set_xlabel('Fare')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 200)  # Limit x-axis for better visualization
    
    # 6. Survival by Embarked Port
    survival_by_embarked = pd.crosstab(train_df['Embarked'], train_df['Survived'])
    survival_by_embarked.plot(kind='bar', ax=axes[1, 2], color=['#ff6b6b', '#4ecdc4'])
    axes[1, 2].set_title('Survival by Embarked Port')
    axes[1, 2].set_xlabel('Port')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend(['Died', 'Survived'])
    axes[1, 2].tick_params(axis='x', rotation=0)
    
    # 7. Family size analysis
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    family_survival = train_df.groupby('FamilySize')['Survived'].mean()
    axes[2, 0].bar(family_survival.index, family_survival.values, color='#45b7d1')
    axes[2, 0].set_title('Survival Rate by Family Size')
    axes[2, 0].set_xlabel('Family Size')
    axes[2, 0].set_ylabel('Survival Rate')
    
    # 8. Survival rate by gender and class
    survival_gender_class = train_df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
    survival_gender_class.plot(kind='bar', ax=axes[2, 1], color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2, 1].set_title('Survival Rate by Gender and Class')
    axes[2, 1].set_xlabel('Gender')
    axes[2, 1].set_ylabel('Survival Rate')
    axes[2, 1].legend(title='Class', labels=['1st', '2nd', '3rd'])
    axes[2, 1].tick_params(axis='x', rotation=0)
    
    # 9. Age vs Fare scatter plot
    scatter_colors = ['#ff6b6b' if x == 0 else '#4ecdc4' for x in train_df['Survived']]
    axes[2, 2].scatter(train_df['Age'], train_df['Fare'], c=scatter_colors, alpha=0.6)
    axes[2, 2].set_title('Age vs Fare (colored by survival)')
    axes[2, 2].set_xlabel('Age')
    axes[2, 2].set_ylabel('Fare')
    axes[2, 2].set_ylim(0, 300)  # Limit y-axis for better visualization
    
    plt.tight_layout()
    plt.savefig('survival_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Survival analysis plots saved as 'survival_analysis_plots.png'")

def correlation_analysis(train_df):
    """Analyze correlations between variables"""
    
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Select numeric columns for correlation
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Create correlation matrix
    correlation_matrix = train_df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display correlation with survival
    print("Correlation with Survival (sorted by absolute value):")
    print("-" * 50)
    correlation_with_survival = correlation_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=False)
    for feature, corr in correlation_with_survival.items():
        print(f"{feature:15s}: {corr:6.3f}")
    
    print("✓ Correlation matrix saved as 'correlation_matrix.png'")

def detailed_feature_analysis(train_df):
    """Detailed analysis of individual features"""
    
    print("\n=== DETAILED FEATURE ANALYSIS ===")
    
    # Create feature analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Feature Analysis', fontsize=16, fontweight='bold')
    
    # 1. Title extraction and analysis
    train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_survival = train_df.groupby('Title')['Survived'].agg(['count', 'mean']).sort_values('count', ascending=False)
    
    # Keep only titles with more than 10 occurrences
    common_titles = title_survival[title_survival['count'] >= 10]
    axes[0, 0].bar(common_titles.index, common_titles['mean'], color='skyblue')
    axes[0, 0].set_title('Survival Rate by Title (>10 occurrences)')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Cabin deck analysis
    train_df['Deck'] = train_df['Cabin'].str[0]
    deck_survival = train_df.groupby('Deck')['Survived'].mean().sort_values(ascending=False)
    axes[0, 1].bar(deck_survival.index, deck_survival.values, color='lightcoral')
    axes[0, 1].set_title('Survival Rate by Cabin Deck')
    axes[0, 1].set_ylabel('Survival Rate')
    
    # 3. Age group analysis
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=bins, labels=labels)
    age_group_survival = train_df.groupby('AgeGroup')['Survived'].mean()
    axes[1, 0].bar(age_group_survival.index, age_group_survival.values, color='lightgreen')
    axes[1, 0].set_title('Survival Rate by Age Group')
    axes[1, 0].set_ylabel('Survival Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Fare quartile analysis
    train_df['FareQuartile'] = pd.qcut(train_df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    fare_quartile_survival = train_df.groupby('FareQuartile')['Survived'].mean()
    axes[1, 1].bar(fare_quartile_survival.index, fare_quartile_survival.values, color='gold')
    axes[1, 1].set_title('Survival Rate by Fare Quartile')
    axes[1, 1].set_ylabel('Survival Rate')
    
    plt.tight_layout()
    plt.savefig('detailed_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Detailed feature analysis plots saved as 'detailed_feature_analysis.png'")

def statistical_tests(train_df):
    """Perform statistical tests to validate insights"""
    
    print("\n=== STATISTICAL TESTS ===")
    
    # Chi-square test for categorical variables
    from scipy.stats import chi2_contingency
    
    # Test 1: Gender vs Survival
    gender_survival_crosstab = pd.crosstab(train_df['Sex'], train_df['Survived'])
    chi2_gender, p_gender, _, _ = chi2_contingency(gender_survival_crosstab)
    print(f"Gender vs Survival:")
    print(f"  Chi-square statistic: {chi2_gender:.4f}")
    print(f"  P-value: {p_gender:.6f}")
    print(f"  Significant: {'Yes' if p_gender < 0.05 else 'No'}")
    
    # Test 2: Class vs Survival
    class_survival_crosstab = pd.crosstab(train_df['Pclass'], train_df['Survived'])
    chi2_class, p_class, _, _ = chi2_contingency(class_survival_crosstab)
    print(f"\nClass vs Survival:")
    print(f"  Chi-square statistic: {chi2_class:.4f}")
    print(f"  P-value: {p_class:.6f}")
    print(f"  Significant: {'Yes' if p_class < 0.05 else 'No'}")
    
    # T-test for continuous variables
    from scipy.stats import ttest_ind
    
    # Test 3: Age difference between survivors and non-survivors
    survived_ages = train_df[train_df['Survived'] == 1]['Age'].dropna()
    died_ages = train_df[train_df['Survived'] == 0]['Age'].dropna()
    t_stat_age, p_age = ttest_ind(survived_ages, died_ages)
    print(f"\nAge difference (Survivors vs Non-survivors):")
    print(f"  T-statistic: {t_stat_age:.4f}")
    print(f"  P-value: {p_age:.6f}")
    print(f"  Significant: {'Yes' if p_age < 0.05 else 'No'}")
    print(f"  Survivors mean age: {survived_ages.mean():.2f}")
    print(f"  Non-survivors mean age: {died_ages.mean():.2f}")
    
    # Test 4: Fare difference between survivors and non-survivors
    survived_fares = train_df[train_df['Survived'] == 1]['Fare'].dropna()
    died_fares = train_df[train_df['Survived'] == 0]['Fare'].dropna()
    t_stat_fare, p_fare = ttest_ind(survived_fares, died_fares)
    print(f"\nFare difference (Survivors vs Non-survivors):")
    print(f"  T-statistic: {t_stat_fare:.4f}")
    print(f"  P-value: {p_fare:.6f}")
    print(f"  Significant: {'Yes' if p_fare < 0.05 else 'No'}")
    print(f"  Survivors mean fare: £{survived_fares.mean():.2f}")
    print(f"  Non-survivors mean fare: £{died_fares.mean():.2f}")

def generate_insights_report(train_df):
    """Generate a comprehensive insights report"""
    
    print("\n=== GENERATING INSIGHTS REPORT ===")
    
    insights = []
    
    # Gender insights
    gender_survival = train_df.groupby('Sex')['Survived'].mean()
    insights.append(f"Women had a {gender_survival['female']:.1%} survival rate vs {gender_survival['male']:.1%} for men")
    
    # Class insights
    class_survival = train_df.groupby('Pclass')['Survived'].mean()
    insights.append(f"1st class passengers had {class_survival[1]:.1%} survival rate vs {class_survival[3]:.1%} for 3rd class")
    
    # Age insights
    child_survival = train_df[train_df['Age'] < 18]['Survived'].mean()
    adult_survival = train_df[train_df['Age'] >= 18]['Survived'].mean()
    insights.append(f"Children (<18) had {child_survival:.1%} survival rate vs {adult_survival:.1%} for adults")
    
    # Family size insights
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    alone_survival = train_df[train_df['FamilySize'] == 1]['Survived'].mean()
    family_survival = train_df[train_df['FamilySize'] > 1]['Survived'].mean()
    insights.append(f"Passengers traveling alone had {alone_survival:.1%} survival rate vs {family_survival:.1%} for those with family")
    
    # Save insights to file
    with open('eda_insights.txt', 'w') as f:
        f.write("TITANIC EDA INSIGHTS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 20 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
        
        f.write("\nDETAILED STATISTICS:\n")
        f.write("-" * 25 + "\n")
        
        f.write("Survival by Gender:\n")
        for sex in gender_survival.index:
            f.write(f"  {sex.capitalize()}: {gender_survival[sex]:.1%}\n")
        
        f.write("\nSurvival by Class:\n")
        for pclass in class_survival.index:
            f.write(f"  Class {pclass}: {class_survival[pclass]:.1%}\n")
        
        f.write(f"\nOverall survival rate: {train_df['Survived'].mean():.1%}\n")
        f.write(f"Total passengers: {len(train_df)}\n")
    
    print("✓ Insights report saved as 'eda_insights.txt'")
    
    # Display key insights
    print("\nKEY INSIGHTS:")
    print("-" * 20)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

if __name__ == "__main__":
    # Load data
    train_df = load_data()
    
    if train_df is not None:
        # Create survival analysis plots
        survival_analysis_plots(train_df)
        
        # Correlation analysis
        correlation_analysis(train_df)
        
        # Detailed feature analysis
        detailed_feature_analysis(train_df)
        
        # Statistical tests
        statistical_tests(train_df)
        
        # Generate insights report
        generate_insights_report(train_df)
        
        print("\n=== EDA VISUALIZATION COMPLETE ===")
        print("Generated files:")
        print("- survival_analysis_plots.png")
        print("- correlation_matrix.png") 
        print("- detailed_feature_analysis.png")
        print("- eda_insights.txt")
        print("\nNext step: Run 3_feature_engineering.py")
    else:
        print("EDA failed. Please check your data files.")