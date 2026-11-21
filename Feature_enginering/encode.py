# %% Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %% Load data
df = pd.read_csv('customer.csv')

# Encode the target explicitly (no LabelEncoder, so the mapping is clear)
y = df['purchased'].map({'No': 0, 'Yes': 1})

# %% Feature groups
ordinal_cols = ['education', 'review']
nominal_cols = ['gender']
numerical_cols = ['age']

X = df[ordinal_cols + nominal_cols + numerical_cols]

# stratify=y keeps the Yes/No ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% Preprocessing
# Scaling removed: tree-based models don't need it. Age is passed through as is.
# (If you switch to Logistic Regression / SVM / KNN, add StandardScaler on 'age'.)
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(
            categories=[['School', 'UG', 'PG'], ['Poor', 'Average', 'Good']]
        ), ordinal_cols),
        ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_cols),
        ('numerical', 'passthrough', numerical_cols),
    ]
)

# %% Models and small, regularized search spaces
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'model__n_estimators': [20, 50, 100],
            'model__max_depth': [1, 2, 3],
            'model__learning_rate': [0.05, 0.1],
            'model__min_samples_leaf': [2, 4],
        },
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100],
            'model__max_depth': [2, 3, 4],
            'model__min_samples_leaf': [2, 4],
        },
    ),
}

# %% Tune, then evaluate on the held-out test set
best_models = {}

for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])

    search = GridSearchCV(pipe, grid, cv=cv, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    best_models[name] = best

    print(f'\n=== {name} ===')
    print('Best params     :', search.best_params_)
    print('CV accuracy     : %.3f' % search.best_score_)
    print('Train accuracy  : %.3f' % accuracy_score(y_train, best.predict(X_train)))
    print('Test accuracy   : %.3f' % accuracy_score(y_test, best.predict(X_test)))

# %% More reliable comparison: cross-validation on ALL data
# With ~50 rows, one 10-row test split is very noisy, so trust this more.
# Note: this uses the tuned hyperparameters, so it is slightly optimistic.
print('\n=== 5-fold CV on full data (tuned models) ===')
for name, model in best_models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'{name:18s} mean={scores.mean():.3f}  std={scores.std():.3f}')

# %% Detailed report for the better model
best_name = max(
    best_models,
    key=lambda n: cross_val_score(best_models[n], X, y, cv=cv).mean()
)
print(f'\nBest model: {best_name}')
print(classification_report(y_test, best_models[best_name].predict(X_test)))