import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
data = pd.read_csv('D:\\codyb\\COMP6970_Final_Project_Data\\AAPL_minute_data_advanced_labeled.csv')

# 2. Drop the 'Chart Pattern' column as per your instruction
data = data.drop(['Chart Pattern'], axis=1)

# 3. Separate features and target
# Corrected here: Use all columns except the last one for X, and the last column for y
X = data.iloc[:, :-1]  # All columns except the last one (23 columns)
y = data.iloc[:, -1]    # The last column as target

# 4. Handle missing values
# Impute numerical features with mean
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_features] = X[numerical_features].fillna(X[numerical_features].mean())

# Impute categorical features with mode or a placeholder
# Identify categorical features (assuming 'Candlestick Pattern' was the only categorical feature)
# Since 'Candlestick Pattern' is now the target, check for other categorical features if any
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    X[col] = X[col].fillna('None')  # Replace NaNs with 'None' or appropriate placeholder

# 5. Feature Engineering
# Convert 'datetime' to datetime object
X['datetime'] = pd.to_datetime(X['datetime'])

# Extract datetime features
X['hour'] = X['datetime'].dt.hour
X['minute'] = X['datetime'].dt.minute
X['day_of_week'] = X['datetime'].dt.dayofweek
X['day'] = X['datetime'].dt.day
X['month'] = X['datetime'].dt.month

# Drop original 'datetime' column
X = X.drop(['datetime'], axis=1)

# 6. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 7. Encode categorical features in X (if any remain after handling 'Candlestick Pattern')
categorical_features = X.select_dtypes(include=['object']).columns
if len(categorical_features) > 0:
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# 8. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 9. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 10. Handle class imbalance with SMOTE
print("Before SMOTE:", pd.Series(y_train).value_counts())
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", pd.Series(y_train_res).value_counts())

# 11. Initialize and train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_res, y_train_res)

# 12. Predict and evaluate
y_pred = rf_classifier.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 13. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Random Forest Confusion Matrix')
plt.show()

# 14. Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

grid_search.fit(X_train_res, y_train_res)
print(f"Best Parameters: {grid_search.best_params_}")

# 15. Train Best Model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_res, y_train_res)
y_pred_best = best_rf.predict(X_test)

# 16. Evaluate Best Model
print(f"Best Random Forest Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print("Best Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# 17. Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[:20], y=feature_importances.index[:20])
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
