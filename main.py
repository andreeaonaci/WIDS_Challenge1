import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

train = pd.read_csv('/kaggle/input/widsdata/train.csv')
test = pd.read_csv('/kaggle/input/widsdata/test.csv')

train.drop(columns=['patient_id'], inplace=True)
test.drop(columns=['patient_id'], inplace=True)

train['state_zip'] = train['patient_state'] + train['patient_zip3'].astype(str)
test['state_zip'] = test['patient_state'] + test['patient_zip3'].astype(str)

test['DiagPeriodL90D'] = 2
df = pd.concat([train, test])

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
# Define columns for training
cols = ['breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code', 'patient_zip3', 'patient_age', 'payer_type',
        'patient_state', 'breast_cancer_diagnosis_desc']
        'state_zip', 'breast_cancer_diagnosis_desc']

train = df[df['DiagPeriodL90D'] != 2]
test = df[df['DiagPeriodL90D'] == 2].drop(columns=['DiagPeriodL90D'])

auc_scores = []
test_preds = []

cv = StratifiedKFold(n_splits=5, shuffle= True, random_state=42)

params = {
    'depth': None,
    'random_state': 69,
    'eval_metric': 'AUC',
    'verbose': False,
    'loss_function': 'Logloss',
    'learning_rate': 0.005,
    'iterations': 5000,
    'grow_policy': 'Lossguide',
    'l2_leaf_reg': 10,
    'border_count': 128,
    'min_child_samples': 9,
    'leaf_estimation_method': 'Gradient',
    'leaf_estimation_iterations': 16,
    'leaf_estimation_backtracking': 'Armijo',
    'bootstrap_type': 'Poisson',
    'bootstrap_type': 'Poisson',
    'subsample': 0.93,
    'allow_writing_files': False,
    'task_type': 'GPU',
    'max_depth': 10
}

X = train[cols]
y = train['DiagPeriodL90D']

for train_indices, test_indices in cv.split(X, y):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        model = CatBoostClassifier(**params)
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        
        preds = model.predict_proba(X_test)[:, 1]
        preds_test = model.predict_proba(test[cols])[:, 1]
        test_preds.append(preds_test)
       
        auc_score = roc_auc_score(y_test, preds)
        auc_scores.append(auc_score)
        print(f"AUC Score: {auc_score}")

print(f"Average AUC Score: {np.mean(auc_scores)}")

ensemble_preds = np.mean(test_preds, axis=0)

submission = pd.DataFrame()

submission = submission.reset_index(drop=True)

submission['DiagPeriodL90D'] = ensemble_preds

first_column = test.iloc[:, 0]
submission.insert(0, first_column.name, first_column)

submission.to_csv('submission_StateZip.csv', index=False)
