import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set Seaborn style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv(r"C:\INT375_CA2\stolen_vehicles.csv")

# Clean and prepare
df = df.dropna(subset=['vehicle_type', 'model_year', 'color', 'date_stolen', 'location_id'])
df['date_stolen'] = pd.to_datetime(df['date_stolen'], errors='coerce')
df = df.dropna(subset=['date_stolen'])
df['month'] = df['date_stolen'].dt.month
df['year'] = df['date_stolen'].dt.year

# ========== EDA (Exploratory Data Analysis) ==========
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Objective 1: Top 10 stolen vehicle types - Horizontal Bar Chart (Matplotlib) - Color: Teal (#3B9AB2)
plt.figure(figsize=(10, 6))
vehicle_type_counts = df['vehicle_type'].value_counts().head(10)
plt.barh(vehicle_type_counts.index, vehicle_type_counts.values, color='#3B9AB2')
plt.title('Top 10 Stolen Vehicle Types')
plt.xlabel('Theft Count')
plt.ylabel('Vehicle Type')
plt.tight_layout()
plt.show()

# Objective 2: Thefts by vehicle model year - Line Plot (Seaborn) - Color: Red (#F21A00)
plt.figure(figsize=(10, 6))
model_year_counts = df['model_year'].value_counts().sort_index()
sns.lineplot(x=model_year_counts.index, y=model_year_counts.values, marker='o', color='#F21A00')
plt.title('Thefts by Vehicle Model Year')
plt.xlabel('Model Year')
plt.ylabel('Theft Count')
plt.tight_layout()
plt.show()

# Objective 3: Top 5 stolen vehicle colors - Pie Chart (Matplotlib) - Color Palette: Set2
top_colors = df['color'].value_counts().head(5)
plt.figure(figsize=(7, 7))
colors = sns.color_palette("Set2")
plt.pie(top_colors.values, labels=top_colors.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Top 5 Stolen Vehicle Colors')
plt.tight_layout()
plt.show()

# Objective 4: Monthly theft trends by model year - Box Plot (Seaborn) - Color Palette: YlGnBu
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='model_year', data=df, palette='YlGnBu')
plt.title('Monthly Theft Trends by Model Year')
plt.xlabel('Month')
plt.ylabel('Model Year')
plt.tight_layout()
plt.show()

# Objective 5: Top 10 theft locations - Count Plot (Seaborn) - Color Palette: coolwarm
plt.figure(figsize=(10, 6))
top_locations = df['location_id'].value_counts().head(10).index
sns.countplot(data=df[df['location_id'].isin(top_locations)], x='location_id', order=top_locations, palette='coolwarm')
plt.title('Top 10 Theft Locations')
plt.xlabel('Location ID')
plt.ylabel('Theft Count')
plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(df['vehicle_type'], df['color'])

# Perform the Chi-Square test
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-Value: {p_val:.4f}")

from scipy.stats import norm

# Drop nulls for model_year
model_years = df['model_year'].dropna()

# Test parameters
sample_mean = model_years.mean()
sample_std = model_years.std()
n = len(model_years)
hypothesized_mean = 2015

# Z-test calculation
z_score = (sample_mean - hypothesized_mean) / (sample_std / (n ** 0.5))
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Sample Mean: {sample_mean:.2f}")
print(f"Z-Score: {z_score:.2f}")
print(f"P-Value: {p_value:.4f}")
