import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

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
    ('model', LinearRegression()) 
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Train R²:", pipeline.score(X_train, y_train))
print("Test R²:",  pipeline.score(X_test, y_test))