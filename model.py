import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_processed_data():
    """Load the processed datasets"""
    try:
        train_df = pd.read_csv('train_processed.csv')
        test_df = pd.read_csv('test_processed.csv')
        print(f"✓ Loaded processed training data: {train_df.shape}")
        print(f"✓ Loaded processed test data: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 3_feature_engineering.py first to generate processed data files")
        return None, None

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    print("\n=== PREPARING FEATURES FOR MODELING ===")
    
    # Features to exclude from modeling
    exclude_features = [
        'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin',
        'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 
        'Deck', 'FamilySizeGroup', 'TicketPrefixGroup', 'AgeBand',
        'TicketPrefix'  # Keep encoded versions only
    ]
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_features]
    
    # Prepare X and y
    X = train_df[feature_cols].fillna(0)
    y = train_df['Survived']
    X_test = test_df[feature_cols].fillna(0)
    
    print(f"Selected {len(feature_cols)} features for modeling")
    print("Features:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    return X, y, X_test, feature_cols

def feature_selection_analysis(X, y, feature_cols):
    """Analyze and select best features"""
    print("\n=== FEATURE SELECTION ANALYSIS ===")
    
    # Method 1: Univariate feature selection
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('score', ascending=False)
    
    print("Top 15 features by univariate F-score:")
    print("-" * 50)
    for idx, row in feature_scores.head(15).iterrows():
        print(f"{row['feature']:25s}: {row['score']:8.2f} (p={row['p_value']:.4f})")
    
    # Method 2: Random Forest feature importance
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features by Random Forest importance:")
    print("-" * 50)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:25s}: {row['importance']:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features by Random Forest Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_modeling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Select top features
    top_k = 20
    selected_features = feature_importance.head(top_k)['feature'].tolist()
    
    print(f"\nSelected top {top_k} features for modeling:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i:2d}. {feature}")
    
    return selected_features, feature_importance

def initialize_models():
    """Initialize various ML models"""
    print("\n=== INITIALIZING MODELS ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    print(f"Initialized {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

def evaluate_models_cv(models, X, y, cv_folds=5):
    """Evaluate models using cross-validation"""
    print(f"\n=== CROSS-VALIDATION EVALUATION ({cv_folds}-fold) ===")
    
    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Sort results by mean CV score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
    
    print("\nRanking by Cross-Validation Accuracy:")
    print("-" * 50)
    for i, (name, scores) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:20s}: {scores['cv_mean']:.4f} (+/- {scores['cv_std']*2:.4f})")
    
    return results

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning for top models"""
    print("\n=== HYPERPARAMETER TUNING ===")
    
    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    models_to_tune = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    tuned_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
    
    for name, model in models_to_tune.items():
        print(f"Tuning {name}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        tuned_models[name] = grid_search.best_estimator_
        
        print(f"  Best score: {grid_search.best_score_:.4f}")
        print(f"  Best params: {grid_search.best_params_}")
    
    return tuned_models

def create_ensemble_model(tuned_models, X, y):
    """Create ensemble model using best performing models"""
    print("\n=== CREATING ENSEMBLE MODEL ===")
    
    # Select top 3-4 models for ensemble
    ensemble_models = [
        ('rf', tuned_models['Random Forest']),
        ('gb', tuned_models['Gradient Boosting']),
        ('xgb', tuned_models['XGBoost'])
    ]
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'  # Use probability averaging
    )
    
    # Evaluate ensemble
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
    print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return ensemble

def final_model_evaluation(model, X, y):
    """Detailed evaluation of the final model"""
    print("\n=== FINAL MODEL EVALUATION ===")
    
    # Split data for final evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # AUC-ROC
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print("-" * 30)
    print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Died', 'Survived'], 
                yticklabels=['Died', 'Survived'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, auc_score

def make_predictions(model, X_test, test_df):
    """Make predictions on test data"""
    print("\n=== MAKING PREDICTIONS ===")
    
    # Make predictions
    predictions = model.predict(X_test)
    prediction_probabilities = model.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # Save submission file
    submission.to_csv('titanic_submission.csv', index=False)
    
    # Also save with probabilities for analysis
    submission_with_proba = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions,
        'Probability': prediction_probabilities
    })
    submission_with_proba.to_csv('titanic_submission_with_probabilities.csv', index=False)
    
    # Summary statistics
    survival_rate = predictions.mean()
    print(f"Predicted survival rate: {survival_rate:.1%}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted survivors: {predictions.sum()}")
    print(f"Predicted deaths: {len(predictions) - predictions.sum()}")
    
    print("\n✓ Submission files saved:")
    print("  - titanic_submission.csv (for Kaggle)")
    print("  - titanic_submission_with_probabilities.csv (with probabilities)")
    
    return submission

def save_model_results(results, tuned_models, final_accuracy, final_auc):
    """Save comprehensive model results"""
    print("\n=== SAVING MODEL RESULTS ===")
    
    with open('model_results_summary.txt', 'w') as f:
        f.write("TITANIC MODEL TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CROSS-VALIDATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        for i, (name, scores) in enumerate(sorted_results, 1):
            f.write(f"{i:2d}. {name:20s}: {scores['cv_mean']:.4f} (+/- {scores['cv_std']*2:.4f})\n")
        
        f.write("\nFINAL MODEL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Validation Accuracy: {final_accuracy:.4f}\n")
        f.write(f"AUC-ROC Score: {final_auc:.4f}\n")
        
        f.write("\nTUNED MODEL PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        for name, model in tuned_models.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Parameters: {model.get_params()}\n")
    
    print("✓ Model results saved to 'model_results_summary.txt'")

def create_prediction_analysis(submission_with_proba, test_df):
    """Analyze the predictions made"""
    print("\n=== PREDICTION ANALYSIS ===")
    
    # Merge with test data for analysis
    analysis_df = submission_with_proba.merge(test_df, on='PassengerId')
    
    # Analyze predictions by key features
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prediction Analysis by Key Features', fontsize=16, fontweight='bold')
    
    # By Gender
    if 'Sex_Encoded' in analysis_df.columns:
        gender_pred = analysis_df.groupby('Sex_Encoded')['Survived'].mean()
        axes[0, 0].bar(['Male', 'Female'], gender_pred.values, color=['skyblue', 'pink'])
        axes[0, 0].set_title('Predicted Survival Rate by Gender')
        axes[0, 0].set_ylabel('Predicted Survival Rate')
    
    # By Class
    class_pred = analysis_df.groupby('Pclass')['Survived'].mean()
    axes[0, 1].bar(['1st', '2nd', '3rd'], class_pred.values, color=['gold', 'silver', 'brown'])
    axes[0, 1].set_title('Predicted Survival Rate by Class')
    axes[0, 1].set_ylabel('Predicted Survival Rate')
    
    # Probability distribution
    axes[1, 0].hist(analysis_df['Probability'], bins=30, alpha=0.7, color='green')
    axes[1, 0].set_title('Distribution of Survival Probabilities')
    axes[1, 0].set_xlabel('Survival Probability')
    axes[1, 0].set_ylabel('Count')
    
    # Age vs Probability scatter
    if 'Age' in analysis_df.columns:
        scatter_colors = ['red' if x == 0 else 'blue' for x in analysis_df['Survived']]
        axes[1, 1].scatter(analysis_df['Age'], analysis_df['Probability'], 
                          c=scatter_colors, alpha=0.6)
        axes[1, 1].set_title('Age vs Survival Probability')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Survival Probability')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Prediction analysis plot saved as 'prediction_analysis.png'")

if __name__ == "__main__":
    print("=== TITANIC MODEL TRAINING AND PREDICTION ===\n")
    
    # Load processed data
    train_df, test_df = load_processed_data()
    
    if train_df is not None and test_df is not None:
        # Prepare features
        X, y, X_test, feature_cols = prepare_features(train_df, test_df)
        
        # Feature selection analysis
        selected_features, feature_importance = feature_selection_analysis(X, y, feature_cols)
        
        # Use selected features
        X_selected = X[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Initialize models
        models = initialize_models()
        
        # Evaluate models with cross-validation
        results = evaluate_models_cv(models, X_selected, y)
        
        # Hyperparameter tuning
        tuned_models = hyperparameter_tuning(X_selected, y)
        
        # Create ensemble model
        ensemble_model = create_ensemble_model(tuned_models, X_selected, y)
        
        # Final model evaluation
        final_accuracy, final_auc = final_model_evaluation(ensemble_model, X_selected, y)
        
        # Make predictions
        submission = make_predictions(ensemble_model, X_test_selected, test_df)
        
        # Create prediction analysis
        submission_with_proba = pd.read_csv('titanic_submission_with_probabilities.csv')
        create_prediction_analysis(submission_with_proba, test_df)
        
        # Save results
        save_model_results(results, tuned_models, final_accuracy, final_auc)
        
        print("\n=== MODEL TRAINING COMPLETE ===")
        print("Generated files:")
        print("- feature_importance_modeling.png")
        print("- confusion_matrix.png")
        print("- roc_curve.png")
        print("- prediction_analysis.png")
        print("- titanic_submission.csv")
        print("- titanic_submission_with_probabilities.csv")
        print("- model_results_summary.txt")
        print("\nSubmission file ready for Kaggle!")
        
    else:
        print("Model training failed. Please run feature engineering first.")