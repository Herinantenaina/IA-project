import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder

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

# DROP sparse columns >70% null
sparse_cols = df.columns[df.isnull().mean() > 0.7].tolist()

df = df.drop(columns=drop_cols + sparse_cols)

# Solve nan values
if df['ConvertedCompYearly'].isna().any():
    median = df['ConvertedCompYearly'].median()
    df['ConvertedCompYearly'] = df['ConvertedCompYearly'].fillna(median)

# Normalization
df['Salary'] = np.log1p(df['ConvertedCompYearly'])

# Int-ify the Age column
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

df['Age_int'] = df['Age'].apply(age_to_int)

# Binning experience to improve performance
if df['YearsCode'].isna().any():
    median = df['YearsCode'].median()
    df['YearsCode'] = df['YearsCode'].fillna(median)
print(df['YearsCode'].unique())

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

df['ExperienceRange'] = df['YearsCode'].apply(categorize_experience)
df = df.drop(columns=['YearsCode'])

# Multi-value and single-value encoding
multi_value = [ 
    col for col in df.select_dtypes(include='object').columns
    if df[col].dropna().str.contains(';').any()
]

single_value = [
    col for col in df.select_dtypes(include='object').columns
    if col not in multi_value
]

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
