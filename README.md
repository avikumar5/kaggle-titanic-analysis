# 🚢 Titanic - Machine Learning from Disaster

This project is an end-to-end solution for the classic Kaggle Titanic dataset challenge. It includes data loading, EDA, feature engineering, model training, evaluation, and prediction, all using Python and popular data science libraries.

## 📁 Project Structure

```
.
├── train.csv
├── test.csv
├── gender_submission.csv
├── 1_data_exploration.py
├── 2_titanic_eda.py
├── 3_feature_engineering.py
├── 4_model.py
├── data_exploration_summary.txt
├── eda_insights.txt
├── feature_list.txt
├── encoders_info.txt
├── train_processed.csv
├── test_processed.csv
├── titanic_submission.csv
├── titanic_submission_with_probabilities.csv
├── *.png (plots)
├── model_results_summary.txt
```

## 🚀 Pipeline Overview

### 1️⃣ Data Exploration (`1_data_exploration.py`)
- Loads and inspects the training/test datasets
- Analyzes missing values and basic survival stats
- Saves a summary to `data_exploration_summary.txt`

### 2️⃣ Exploratory Data Analysis (`2_titanic_eda.py`)
- Generates visual insights: survival by gender, class, age, fare, etc.
- Performs correlation heatmaps and statistical tests
- Saves plots and key findings in `eda_insights.txt`

### 3️⃣ Feature Engineering (`3_feature_engineering.py`)
- Extracts and engineers useful features (e.g. title, family size, cabin deck)
- Handles missing data intelligently
- Encodes categorical variables
- Performs feature importance analysis
- Outputs processed datasets and encoder metadata

### 4️⃣ Model Training & Prediction (`4_model.py`)
- Trains multiple models: Random Forest, XGBoost, Logistic Regression, etc.
- Performs feature selection and hyperparameter tuning
- Builds a soft voting ensemble
- Evaluates model with metrics and plots
- Generates Kaggle-ready predictions in `titanic_submission.csv`

## 🤖 Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors
- Decision Tree
- Voting Ensemble (Soft Voting)


### Key Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `scipy`

## 🧪 How to Run

1. Place `train.csv`, `test.csv`, and `gender_submission.csv` in the root directory.
2. Run scripts in order:

```bash
python 1_data_exploration.py
python 2_eda_visualization.py
python 3_feature_engineering.py
python 4_model_training.py
```

3. Final predictions will be saved as `titanic_submission.csv` and `titanic_submission_with_probabilities.csv`.

## 📊 Output Highlights

- `feature_importance.png`
- `survival_analysis_plots.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `prediction_analysis.png`
- `titanic_submission.csv` – Submit this to Kaggle!

## 📈 Insights

Key findings include:
- Higher survival rate for females and 1st class passengers
- Survival is positively correlated with fare and negatively with family size over 4
- Children had a higher chance of survival than adults

See: `eda_insights.txt`

## 📌 License

This project is provided for educational purposes. Customize and extend it as needed!

---

Happy learning and good luck on Kaggle! 🧠🚢