"""
Script to generate submission file for Spaceship Titanic competition
Target: ~80% accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# Store original PassengerId for submission
df2_original = df2.copy()

print("Preprocessing data...")

# Fill missing values
def preprocess_data(df):
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill boolean columns with mode
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].map({'True': True, 'False': False, 'nan': False})
        df[col].fillna(False, inplace=True)
    
    # Fill categorical columns with mode
    cat_cols = ['HomePlanet', 'Cabin', 'Destination']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Convert float to int
    float_cols = df.select_dtypes(include='float64').columns.tolist()
    for col in float_cols:
        df[col] = df[col].astype('int64')
    
    # Encode object columns
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        if col != 'PassengerId' and col != 'Name':  # Don't encode IDs
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Convert bool to int
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype('int64')
    
    return df

df1_processed = preprocess_data(df1)
df2_processed = preprocess_data(df2)

# Prepare training data - exclude PassengerId and Name as they're just IDs
train_x = df1_processed.drop(['Transported', 'PassengerId', 'Name'], axis=1, errors='ignore')
train_y = df1_processed['Transported']
df2_features = df2_processed.drop(['PassengerId', 'Name'], axis=1, errors='ignore')

# Ensure same columns
common_cols = train_x.columns.intersection(df2_features.columns)
train_x = train_x[common_cols]
df2_features = df2_features[common_cols]

print(f"Training features: {list(train_x.columns)}")
print(f"Training samples: {len(train_x)}")

# Try XGBoost first, fallback to RandomForest
try:
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    print("\nUsing XGBoost model")
except ImportError:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    print("\nUsing RandomForest model (install xgboost for better performance: pip install xgboost)")

# Train model
print("Training model...")
model.fit(train_x, train_y)

# Cross-validation
print("\nEvaluating model...")
cv_scores = cross_val_score(model, train_x, train_y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual CV scores: {cv_scores}")
print(f"Expected Kaggle score: ~{cv_scores.mean()*100:.2f}%")

# Make predictions
print("\nMaking predictions...")
pred = model.predict(df2_features)
pred = pred.astype(bool)  # Convert to boolean True/False

# Create submission
submission = pd.DataFrame({
    "PassengerId": df2_original["PassengerId"],
    "Transported": pred
})

# Save submission
submission.to_csv("submissions.csv", index=False)
print(f"\nSubmission file created: submissions.csv")
print(f"Total predictions: {len(submission)}")
print(f"Transported distribution: {submission['Transported'].value_counts().to_dict()}")
print("\nFirst few predictions:")
print(submission.head(10))
