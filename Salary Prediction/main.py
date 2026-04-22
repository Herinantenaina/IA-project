import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

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
print(len(df.columns))

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

if df['YearsCode'].isna().any():
    median = df['YearsCode'].median()
    df['YearsCode'] = df['YearsCode'].fillna(median)
  
multi_value = [ 
    col for col in df.select_dtypes(include='object').columns
    if df[col].dropna().str.contains(';').any()
]

single_value = [
    col for col in df.select_dtypes(include='object').columns
    if col not in multi_value
]

print(f"Multi-value columns ({len(multi_value)}):")
print(multi_value)

print(f"\nSingle-value columns ({len(single_value)}):")
print(single_value)
