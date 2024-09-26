import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_excel('dataset_njam.xlsx')

# Display the first 5 rows
print(data.head())

# Display the columns
print("Columns in dataset:", data.columns)

# Clean column names (strip leading/trailing spaces)
data.columns = data.columns.str.strip()

# Check if required columns exist after cleaning
required_columns = ['Pokok', 'usia', 'pendidikan', 'Jenis_Jenis_Kelamin']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Error: Missing columns in dataset: {missing_columns}")
else:
    # Descriptive statistics
    print(data.describe())

    # Info about data types and missing values
    print(data.info())

    # Handling missing values
    # Fill numeric columns with the mean
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Fill non-numeric columns with the mode
    data_non_numeric = data.select_dtypes(exclude=[np.number])
    for col in data_non_numeric.columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Convert columns that should be numeric (e.g., 'Pokok') and handle non-numeric values
    data['Pokok'] = pd.to_numeric(data['Pokok'], errors='coerce')
    data['usia'] = pd.to_numeric(data['usia'], errors='coerce')

    # Drop rows where 'Pokok' or 'usia' have non-convertible values (NaN after conversion)
    data.dropna(subset=['Pokok', 'usia'], inplace=True)

    # Scatterplot to visualize the relationship between 'Pokok' and 'usia'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='usia', y='Pokok', hue='Jenis_Jenis_Kelamin', style='pendidikan', data=data)
    plt.title('Scatterplot of Pokok vs Usia by Jenis Kelamin and Pendidikan')
    plt.xlabel('Usia')
    plt.ylabel('Pokok')
    plt.legend(title='Jenis Kelamin & Pendidikan')
    plt.show()

    # Boxplot to compare 'Pokok' across different 'pendidikan' levels
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='pendidikan', y='Pokok', hue='Jenis_Jenis_Kelamin', data=data)
    plt.title('Boxplot of Pokok by Pendidikan and Jenis Kelamin')
    plt.xlabel('Pendidikan')
    plt.ylabel('Pokok')
    plt.legend(title='Jenis Kelamin')
    plt.show()

    # Pairplot to see pairwise relationships between 'Pokok', 'usia', and 'pendidikan'
    # Encoding 'pendidikan' column for visualization
    data['pendidikan_encoded'] = data['pendidikan'].astype('category').cat.codes
    
    sns.pairplot(data, vars=['Pokok', 'usia'], hue='pendidikan_encoded', corner=True, plot_kws={'alpha': 0.5})
    plt.title('Pairplot of Pokok and Usia')
    plt.show()
