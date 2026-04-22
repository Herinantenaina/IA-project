import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

#-------------CLEANING DATA-------------
df = pd.read_csv('survey_results_public.csv')

# Keep the rows where ConvertedCompYearly is not missing
df = df.dropna(subset='ConvertedCompYearly')

# Solve nan values
if df['ConvertedCompYearly'].isna().any():
    median = df['ConvertedCompYearly'].median()
    df['ConvertedCompYearly'] = df['ConvertedCompYearly'].fillna(median)
    
# Keep the rows where ConvertedCompYearly is between 10,000 and 500,000 (outliers problems) 
df = df[
    (df['ConvertedCompYearly'] >= 10_000)
    & (df['ConvertedCompYearly'] <= 500_000)
    ]

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

df['Age'] = df['Age'].apply(age_to_int)


df