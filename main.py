import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def clean_data_file1(path, param="Sonucu"):
    df = pd.read_excel(path)
    """subset=df.columns[1:]"""
    df_cleaned = df.dropna(how='all')
    for col in df_cleaned.columns[:-1]:
        df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
    desired_column_names = ['yas','Sonucu']
    df_cleaned.dropna(inplace=True)
    """desired_column_names = ['yas', 'Test Adı', 'Test Adı', 'Sonucu', 'Referans', 'İstek Zamanı', 'İstek Zamanı',
                            'Test Toplamı']"""
    df_cleaned.columns = desired_column_names
    specific_string = "Sonucu"
    df_cleaned = df_cleaned[~df_cleaned['Sonucu'].str.contains(specific_string)]

    # df_cleaned = df_cleaned.drop_duplicates(subset=df_cleaned.columns[1:], keep=False)
    df_cleaned = df_cleaned.dropna(how='any')
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.duplicated(keep='first')]
    df_cleaned[param] = pd.to_numeric(df_cleaned[param], errors='coerce')



    return df_cleaned
def clean_data_file2(path, param="Sonucu"):
    df = pd.read_excel(path)
    df_cleaned = df.dropna(subset=df.columns[1:], how='all')
    for col in df_cleaned.columns[:-1]:
        df_cleaned[col] = df_cleaned[col].fillna(method='ffill')

    desired_column_names = ['yas', 'Test Adı', 'Test Adı', 'Sonucu', 'Referans']
    df_cleaned.columns = desired_column_names

    df_cleaned = df_cleaned.drop_duplicates(subset=df_cleaned.columns[1:], keep=False)
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.duplicated(keep='first')]
    df_cleaned[param] = pd.to_numeric(df_cleaned[param], errors='coerce')

    return df_cleaned
def save_to_new_excel(dataframe, output_path):
    dataframe.to_excel(output_path, index=False)

input_file_path = "C:/Users/alper/Downloads/AMH2021 YENİ.xlsx"
input_file_path2 = "C:/Users/alper/OneDrive/Masaüstü/mydata2.xlsx"
input_file_path3 = "C:/Users/alper/Downloads/AMH2021 YENİ.xlsx"

dataframe = clean_data_file1(input_file_path)
dataframe2 = clean_data_file2(input_file_path2)

max_value = max(max(dataframe['Sonucu']), max(dataframe2['Sonucu']))

fig, (ax1, ax2) = plt.subplots(1,2)
fig2, axx = plt.subplots()

ax1.set_ylim(0, max_value)
ax2.set_ylim(0, max_value)

sns.boxplot(y=dataframe['Sonucu'], orient='v', ax = ax1)
sns.boxplot(y=dataframe2['Sonucu'], orient='v', ax= ax2)
ax1.set_title('Tarih Aralığı (2021)')
ax2.set_title('Tarih Aralığı (2023)')



yas_values = pd.to_numeric(dataframe['yas'], errors='coerce')


yas_values2 = pd.to_numeric(dataframe2['yas'], errors='coerce')

print(yas_values)



valid_indices_2020 = ~np.isnan(yas_values) & ~np.isnan(dataframe['Sonucu'])
model2020 = LinearRegression()
model2020.fit(yas_values[valid_indices_2020].values.reshape(-1, 1), dataframe['Sonucu'][valid_indices_2020])
pred2020 = model2020.predict(yas_values[valid_indices_2020].values.reshape(-1, 1))

# Fit linear regression models for 2023 data
valid_indices_2023 = ~np.isnan(yas_values2) & ~np.isnan(dataframe2['Sonucu'])
model2023 = LinearRegression()
model2023.fit(yas_values2[valid_indices_2023].values.reshape(-1, 1), dataframe2['Sonucu'][valid_indices_2023])
pred2023 = model2023.predict(yas_values2[valid_indices_2023].values.reshape(-1, 1))

sns.scatterplot(yas_values, dataframe['Sonucu'], ax=axx, palette='blue', label='2021')
sns.scatterplot(yas_values2, dataframe2['Sonucu'], ax=axx, palette='red', label='2023')
plt.xlabel("Yaş")

axx.plot(yas_values[valid_indices_2020], pred2020, color='blue', label='2020 Linear Regression')
axx.plot(yas_values2[valid_indices_2023], pred2023, color='red', label='2023 Linear Regression')


plt.show()


output_file_path = 'yeni.xlsx'
output_file_path2 = 'yeni2.xlsx'

save_to_new_excel(dataframe, output_file_path)
save_to_new_excel(dataframe2, output_file_path)