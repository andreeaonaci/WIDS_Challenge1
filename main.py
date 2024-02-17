import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer

# Load the data
train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('test.csv')

# Separate features (X) and target variable (y)
X = train_df.iloc[:, 1:-1]
y = train_df.iloc[:, -1]

# Identify numeric and non-numeric columns
numeric_cols = X.select_dtypes(include=['number']).columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('target_encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, non_numeric_cols)
    ])

# Classifier pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(random_state=69))
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'xgb__learning_rate': [0.05],
    'xgb__max_depth': [3, 5],
    'xgb__min_child_weight': [1, 5],
    'xgb__subsample': [0.8],
    'xgb__colsample_bytree': [0.8],
    'xgb__gamma': [0],
    'xgb__reg_alpha': [0],
    'xgb__reg_lambda': [0],
    'xgb__scale_pos_weight': [1]  # Adjust for class imbalance if necessary
}

# Grid search
grid_search = GridSearchCV(estimator=xgb_pipeline, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=3, n_jobs=-1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Best model
best_model = grid_search.best_estimator_
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

# AUC score on the validation set
auc_score = roc_auc_score(y_val, y_val_pred_proba)

print(f'Best Parameters: {best_params}')
print(f'AUC score on the validation set: {auc_score}')

# Predictions on the test set
print("Making predictions on the test set...")
predictions_test_proba = best_model.predict_proba(test_df.iloc[:, 1:])[:, 1]
print("Predictions completed.")

# Write predicted probabilities to a text file using index as IDs
print("Writing predicted probabilities to file...")
with open('SampleSolution.txt', 'w') as file:
    for index, prob in zip(test_df.iloc[:, 0], predictions_test_proba):
        file.write(f'{index},{prob}\n')

print("Writing to file completed.")
