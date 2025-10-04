import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('Updated_Car_Sales_Data.csv')

# 2. Проверка максимальной и минимальной цены для определения границ
min_price = data['Price'].min()
max_price = data['Price'].max()
print(f"Минимальная цена: ${min_price:.2f}, Максимальная цена: ${max_price:.2f}")

# 3. Разбиение на классы по цене с заданными границами
num_classes = 5
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Определяем границы: High начинается с 150,000, Very High — выше (например, 250,000)
# Границы до 150,000 делим равномерно для Very Low, Low, Medium
bins = [min_price, 5000, 20000, 50000, 150000, max_price + 1]  # +1 для включения максимума
data['Price_Class'] = pd.cut(data['Price'], bins=bins, labels=labels, include_lowest=True)

# 4. Вывод информации о классах
print("\nРаспределение по классам цен:")
print(data['Price_Class'].value_counts().reindex(labels, fill_value=0))

# 5. Вывод диапазонов цен для каждого класса
price_ranges = data.groupby('Price_Class', observed=True)['Price'].agg(['min', 'max'])
print("\nДиапазоны цен для каждого класса:")
print(price_ranges)

# 6. Визуализация распределения классов
plt.figure(figsize=(10, 6))
data['Price_Class'].value_counts().reindex(labels, fill_value=0).plot(
    kind='bar',
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
)
plt.title('Распределение автомобилей по классам цен')
plt.xlabel('Класс цены')
plt.ylabel('Количество автомобилей')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. Сохранение датасета с новым столбцом классов
data.to_csv('Updated_Car_Sales_Data_with_Classes_prepared.csv', index=False)
print("Датасет с классами сохранен в 'Updated_Car_Sales_Data_with_Classes_prepared.csv'.")