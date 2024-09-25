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

# Check if 'Pokok' column exists after cleaning
if 'Pokok' not in data.columns:
    print("Error: 'Pokok' column not found in dataset.")
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

    # Check for non-numeric data in numeric columns
    for col in data.columns:
        if data[col].dtype == 'object':
            print(f"Warning: Column '{col}' contains non-numeric values that will be excluded or encoded.")
    
    # Convert columns that should be numeric (e.g., 'Pokok') and handle non-numeric values
    data['Pokok'] = pd.to_numeric(data['Pokok'], errors='coerce')

    # Drop rows where 'Pokok' or other key numeric columns have non-convertible values (NaN after conversion)
    data.dropna(subset=['Pokok'], inplace=True)

    # Boxplot to check for outliers in the 'Pokok' column
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Pokok'])
    plt.title('Boxplot for Pokok')
    plt.show()

    # Barplot for the distribution of 'status_pinjaman'
    if 'status_pinjaman' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='status_pinjaman', data=data)
        plt.title('Distribution of Status Pinjaman')
        plt.show()

    # Remove outliers using IQR method
    Q1 = data['Pokok'].quantile(0.25)
    Q3 = data['Pokok'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data['Pokok'] < (Q1 - 1.5 * IQR)) | (data['Pokok'] > (Q3 + 1.5 * IQR)))]

    # One-hot encoding for categorical column 'Jenis_Jenis_Kelamin'
    data_encoded = pd.get_dummies(data, columns=['Jenis_Jenis_Kelamin'], drop_first=True)

    # Normalization of the 'Pokok' column
    scaler_minmax = MinMaxScaler()
    data_encoded['Pokok_normalized'] = scaler_minmax.fit_transform(data_encoded[['Pokok']])

    # Standardization of the 'Pokok' column
    scaler_standard = StandardScaler()
    data_encoded['Pokok_standardized'] = scaler_standard.fit_transform(data_encoded[['Pokok']])

    # Pairplot to visualize relationships between variables
    plt.figure(figsize=(10, 6))
    sns.pairplot(data_encoded[['Pokok', 'usia', 'status_pinjaman']])
    plt.title('Pairplot for Selected Features')
    plt.show()

    # Creating a new feature from 'Pokok' and 'usia'
    if 'usia' in data_encoded.columns:
        data_encoded['fitur_baru'] = data_encoded['Pokok'] * data_encoded['usia']

    # Preparing data for modeling
    if 'status_pinjaman' in data_encoded.columns:
        X = data_encoded.drop('status_pinjaman', axis=1)  # Features
        y = data_encoded['status_pinjaman']  # Target

        # Ensure all columns in X are numeric
        X = X.select_dtypes(include=[np.number])

        # Splitting the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)  # Fit PCA on training data

        # If you want to see the explained variance ratio
        print("Explained variance ratio:", pca.explained_variance_ratio_)

        # Visualizing the PCA result
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Status Pinjaman')
        plt.show()
    else:
        print("Error: 'status_pinjaman' column not found in dataset.")
