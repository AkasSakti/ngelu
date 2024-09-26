import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV
df = pd.read_csv("mall_customers_data.csv")

# Cek apakah ada nilai NaN di dalam dataset
print(df.isnull().any())

# Plot histogram untuk kolom 'Annual_Income'
df["Annual Income (k$)"].plot.hist(bins=10, figsize=(8, 6))
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()

# Plot scatter untuk melihat hubungan antara 'age' dan 'Annual_Income'
df.plot.scatter(x="Age", y="Annual Income (k$)", c="DarkBlue", figsize=(8, 6))
plt.title("Scatterplot of Age vs Annual Income")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.show()

# Plot boxplot untuk keseluruhan data
boxplot = df.boxplot(grid=False, rot=45, figsize=(8, 6))
plt.title("Boxplot of All Features")
plt.show()

# Menentukan nilai Q1 dan Q3 serta IQR untuk kolom 'Annual_Income'
Q1 = df["Annual Income (k$)"].quantile(0.25)
Q3 = df["Annual Income (k$)"].quantile(0.75)
IQR = Q3 - Q1

# Menentukan cutoff untuk mendeteksi outlier
cutoff = Q3 + 1.5 * IQR

# Membuat kolom baru 'Annual_Income_Outlier' untuk menandai outlier
df["Annual_Income_Outlier"] = False
for index, row in df.iterrows():
    if row["Annual Income (k$)"] > cutoff:
        df.at[index, 'Annual_Income_Outlier'] = True

# Menghitung rata-rata 'Annual_Income' untuk yang bukan outlier
mean = df.groupby("Annual_Income_Outlier")["Annual Income (k$)"].mean()

# Mengganti nilai 'Annual_Income' yang outlier dengan mean yang bukan outlier
for index, row in df.iterrows():
    if row["Annual_Income_Outlier"] == True:
        df.at[index, 'Annual Income (k$)'] = mean[False]

# Plot boxplot untuk kolom 'Annual_Income' setelah penggantian nilai outlier
boxplot = df.boxplot(grid=False, column=["Annual Income (k$)"], figsize=(8, 6))
plt.title("Boxplot of Annual Income After Removing Outliers")
plt.show()
