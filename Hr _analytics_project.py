import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("hr_data.csv")

# Basic overview
print(df.head())
print(df.info())
print(df.describe())

# Convert dates
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])
df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])

# Calculate Tenure in years
df['Tenure_Years'] = (df['Exit_Date'].fillna(pd.Timestamp('today')) - df['Joining_Date']).dt.days / 365

# Attrition Encoding
df['Attrition_Flag'] = df['Attrition'].map({'Yes':1, 'No':0})

# Attrition Rate
attrition_rate = df['Attrition_Flag'].mean() * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")

# Tenure vs Attrition
sns.boxplot(x='Attrition', y='Tenure_Years', data=df)
plt.title("Attrition vs Tenure")
plt.show()

# Salary vs Performance Distribution
sns.scatterplot(x='Performance_Rating', y='Salary', hue='Attrition', data=df)
plt.title("Performance vs Salary (Attrition Highlighted)")
plt.show()

# Attrition by Department
dept_attrition = df.groupby('Department')['Attrition_Flag'].mean() * 100
dept_attrition.plot(kind='bar', figsize=(7,4), title="Attrition % by Department")
plt.show()

# Correlation Heatmap
numeric_cols = ['Salary', 'Experience_Years', 'Performance_Rating', 'Job_Satisfaction_Score', 'Tenure_Years']
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap — HR Dataset")
plt.show()
