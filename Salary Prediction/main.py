import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def categorize_experience(years):
    if pd.isnull(years):
        return 'Unknown'
    
    try:
        years = int(years)
    except: 
        return 'Unknown'
    
    if years <= 5:
        return '0-5'
    elif years <= 10:
        return '6-10'
    elif years <= 15:
        return '11-15'
    else:
        return '>20'
    
def age_to_int(age):
    if pd.isna(age):
        return np.nan
    
    if '-' in age:
        parts = age.split('-')
        lower = float(parts[0])
        upper = float(parts[1].split()[0])   # years old and keep the number
        return (lower + upper) / 2
    
    if 'older' in age:
        return float(age.split()[0]) + 4
    
    return np.nan

def get_top_values(col):
    counts = Counter()
    for i in df[col].dropna():
        counts.update(i.split(';'))

    return [val for val, count in counts.items() if count >= 100]

#-------------CLEANING DATA-------------
df = pd.read_csv('survey_results_public.csv')

# Keep the rows where ConvertedCompYearly is not missing
df = df.dropna(subset='ConvertedCompYearly')

# DROP these — free text write-ins, leakage, IDs 
drop_cols = [
    'ResponseId',
    'CompTotal', 'Currency', 'AIExplain',
    'TechEndorse_13_TEXT', 'TechOppose_15_TEXT',
    'JobSatPoints_15_TEXT', 'SO_Actions_15_TEXT',
    'AIAgentKnowWrite', 'AIAgentOrchWrite',
    'AIAgentObsWrite', 'AIAgentExtWrite', 'AIOpen',
    'CommPlatformHaveEntr', 'CommPlatformWantEntr',
]

# DROP columns >70% null
sparse_cols = df.columns[df.isnull().mean() > 0.7].tolist()

df = df.drop(columns=drop_cols + sparse_cols)

# Solve nan values
if df['ConvertedCompYearly'].isna().any():
    median = df['ConvertedCompYearly'].median()
    df['ConvertedCompYearly'] = df['ConvertedCompYearly'].fillna(median)


# Normalization
df['Salary'] = np.log1p(df['ConvertedCompYearly'])
df = df.drop(columns=['ConvertedCompYearly'])

# Int-ify the Age column
df['Age_int'] = df['Age'].apply(age_to_int)
df = df.drop(columns=['Age'])

# Binning experience to improve performance
if df['YearsCode'].isna().any():
    median = df['YearsCode'].median()
    df['YearsCode'] = df['YearsCode'].fillna(median)


df['ExperienceRange'] = df['YearsCode'].apply(categorize_experience)
df = df.drop(columns=['YearsCode'])

# Multi-value and single-value encoding
multi_value = [ 
    col for col in df.select_dtypes(include='object').columns
    if df[col].dropna().str.contains(';').any()
]

# single_value = [
#     col for col in df.select_dtypes(include='object').columns
#     if col not in multi_value
# ]

# mlb_dfs = []

# for col in multi_value:
#     split = df[col].fillna('').str.split(';')
#     mlb = MultiLabelBinarizer()
#     encoded = mlb.fit_transform(split)

#     col_names = [f"{col}_{val}" for val in mlb.classes_]
#     mlb_df = pd.DataFrame(encoded, columns=col_names, index=df.index)
#     mlb_dfs.append(mlb_df)

# mlb_result = pd.concat(mlb_dfs, axis=1)

# oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# df[single_value] = oe.fit_transform(df[single_value])

# df = df.drop(columns=multi_value)
# df = pd.concat([df, mlb_result], axis=1)

# top_langs = get_top_values('LanguageHaveWorkedWith')
# top_webFram = get_top_values('WebframeHaveWorkedWith')
# top_platform = get_top_values('PlatformHaveWorkedWith')
# top_db = get_top_values('DatabaseHaveWorkedWith')
# top_officeStack = get_top_values('OfficeStackAsyncHaveWorkedWith')
# top_AI = get_top_values('AIModelsHaveWorkedWith')
# top_SOTags = get_top_values('SOTagsHaveWorkedWith')
# top_webFram = get_top_values('DevEnvsHaveWorkedWith')

binary_cols = {
    'LanguageHaveWorkedWith': get_top_values('LanguageHaveWorkedWith'),
    'WebframeHaveWorkedWith': get_top_values('WebframeHaveWorkedWith'),
    'PlatformHaveWorkedWith': get_top_values('PlatformHaveWorkedWith'),
    'DatabaseHaveWorkedWith': get_top_values('DatabaseHaveWorkedWith'),
    'OfficeStackAsyncHaveWorkedWith': get_top_values('OfficeStackAsyncHaveWorkedWith'),
}

for col, top_vals in binary_cols.items():
    new_cols = pd.DataFrame(
        {f"{col}_{val}": df[col].fillna('').str.contains(val, regex=False).astype(int)
         for val in top_vals},
        index=df.index
    )
    df = pd.concat([df, new_cols], axis=1)

remaining_multival = [col for col in multi_value if col not in binary_cols]
for col in remaining_multival:
    df[f"{col}_count"] = df[col].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    )

df = df.drop(columns=multi_value)

singleval_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Salary']
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[singleval_cols] = oe.fit_transform(df[singleval_cols])

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )) 
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

cv_scores = cross_val_score(
    pipeline,
    X_train, y_train,
    cv=5,               # 5 folds
    scoring='r2',
    n_jobs=-1           # use all CPU cores
)

model = pipeline.named_steps['model']
importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# Top 20
print(importances.head(20))

# Plot
importances.head(20).plot(kind='barh', figsize=(8, 6))

# Keep only features with importance > 0
useful_features = importances[importances > 0].index.tolist()
print(f"Features with signal: {len(useful_features)} out of {X_train.shape[1]}")

X_train_reduced = X_train[useful_features]
X_test_reduced = X_test[useful_features]

# Retrain and check if score improves
cv_scores_reduced = cross_val_score(
    pipeline,
    X_train_reduced, y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_np = imputer.fit_transform(X_train_reduced)
X_train_np = scaler.fit_transform(X_train_np)

X_test_np = imputer.transform(X_test_reduced)
X_test_np = scaler.transform(X_test_np)

# ── Convert to PyTorch tensors ────────────────────────────────
X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# ── DataLoader ────────────────────────────────────────────────
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class SalaryNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

input_dim = X_train_np.shape[1]
model = SalaryNet(input_dim)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        # Check validation loss every 10 epochs
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t)
        print(f"Epoch {epoch+1}/{epochs}  Train loss: {total_loss/len(train_loader):.4f}  Val loss: {val_loss:.4f}")

model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_t).numpy().flatten()

nn_r2 = r2_score(y_test, y_pred_nn)
nn_mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_nn))

print(f"\nNeural Network Test R²:  {nn_r2:.3f}")
print(f"Neural Network MAE:      ${nn_mae:,.0f}")
