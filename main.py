# Import necessary libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Load data
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

# Drop 'patient_id' column
train.drop(columns=['patient_id'], inplace=True)
test.drop(columns=['patient_id'], inplace=True)

# Define function to handle missing values
def impute_categorical(df, columns):
    """Impute missing values in categorical columns using mode."""
    mode_values = df[columns].mode().iloc[0]
    df[columns] = df[columns].fillna(mode_values)
    return df

def impute_numerical(df, columns):
    """Impute missing values in numerical columns using median."""
    median_values = df[columns].median()
    df[columns] = df[columns].fillna(median_values)
    return df

def handle_missing_values(df):
    """Handle missing values in a DataFrame."""
    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns

    # Impute missing values
    df = impute_categorical(df, categorical_columns)
    df = impute_numerical(df, numerical_cols)

    return df

# Handle missing values in train and test data
handle_missing_values(train)
handle_missing_values(test)

# Combine train and test data
test['DiagPeriodL90D'] = 2
df = pd.concat([train, test])

# Encode categorical columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# Define columns for training
cols = ['breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code', 'patient_zip3', 'patient_age', 'payer_type',
        'patient_state', 'breast_cancer_diagnosis_desc']

# Separate train and test data
train = df[df['DiagPeriodL90D'] != 2]
test = df[df['DiagPeriodL90D'] == 2].drop(columns=['DiagPeriodL90D'])

# Initialize AUC scores and test predictions
auc_scores = []
test_preds = []  # Change to list

# Cross-validation settings
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model parameters
params = {
    'depth': 2,
    'random_state': 42,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.3,
    'iterations': 1000
}

X = train[cols]
y = train['DiagPeriodL90D']

# Perform cross-validation and predict
for train_indices, test_indices in cv.split(X, y):
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Initialize CatBoost classifier
    model = CatBoostClassifier(**params)

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Make predictions on the test set
    preds = model.predict_proba(X_test)[:, 1]
    preds_test = model.predict_proba(test[cols])[:, 1]
    test_preds.append(preds_test)  # Append directly to list

    # Calculate AUC score
    auc_score = roc_auc_score(y_test, preds)
    auc_scores.append(auc_score)
    print(f"AUC Score: {auc_score}")

# Print average AUC score
print(f"Average AUC Score: {np.mean(auc_scores)}")

# Combine predictions from different folds
ensemble_preds = np.mean(test_preds, axis=0)

# Prepare submission file
submission = pd.DataFrame()

# Reset the index of the submission DataFrame
submission = submission.reset_index(drop=True)

# Assign predictions to the 'DiagPeriodL90D' column
submission['DiagPeriodL90D'] = ensemble_preds

# Save the submission file
submission.to_csv('submission.csv', index=False)
