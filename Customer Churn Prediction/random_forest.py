import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('customer.csv')
df.head()

df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.isnull().sum()
df = df.dropna()

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes":1, "No": 0})

nunique = X.nunique().sort_values()
binary_cat = nunique[nunique == 2].index.tolist()
multi_cat = nunique[(nunique > 2) & (nunique <= 10)].index.tolist()
numeric_cols = nunique[nunique > 10].index.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), multi_cat),
    ('bin', OrdinalEncoder(), binary_cat)
])


model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', max_depth=10))
])



params = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [5, 10, None]
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

grid = GridSearchCV(model, params, cv=5, scoring='recall', n_jobs=-1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print("Best params: ", grid.best_params_)
print(classification_report(y_test, y_pred))

joblib.dump(grid.best_estimator_, 'model.pkl')