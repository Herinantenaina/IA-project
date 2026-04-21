import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('customer.csv')
df.head()

# Dropping columns
df = df.drop('customerID', axis=1)

# Convert any sring to numeric in total charges
print(df["TotalCharges"].dtype)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print(df["TotalCharges"].dtype)


# Cleaning Data
df.isnull().sum()
df = df.dropna()

# Display balance of yes and no
# print(df["Churn"].value_counts())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert any text into numbers
# print(df.shape)

binary_cols = [
    "Partner", "SeniorCitizen", "gender",
    "Dependents", "PhoneService", "PaperlessBilling"
]

for col in binary_cols:
    df[col] = df[col].astype("category").cat.codes

multi_cols = [
    "PaymentMethod", "StreamingMovies", "TechSupport",
    "OnlineBackup", "StreamingTV", "DeviceProtection",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "Contract"
]

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Check every type of each value
# print(df.nunique().sort_values(ascending=False))

# Separating X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# x.to_csv("cleaned_customer.csv", index=False)
# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# df = pd.get_dummies(df, drop_first=True)

