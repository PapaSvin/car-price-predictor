import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# 1. Определение улучшенной модели (должна совпадать с main_improved.py)
class ImprovedCarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(ImprovedCarPricePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.15),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# 2. Загрузка сохраненных объектов
device = torch.device("cpu")  # Используем CPU для предсказаний
print(f"Используется устройство: {device}")

with open('scaler_features.pkl', 'rb') as f:
    scaler_features = pickle.load(f)
with open('scaler_price.pkl', 'rb') as f:
    scaler_price = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)
with open('make_model_class_dict.pkl', 'rb') as f:
    make_model_class_dict = pickle.load(f)

# Загрузка input_size
input_size = 20  # Учитываем Price_Class
model = ImprovedCarPricePredictor(input_size=input_size).to(device)
model.load_state_dict(torch.load('car_price_predictor_improved.pth', map_location=device))
model.eval()
print(f"Модель загружена успешно! Input size: {input_size}")

# 3. Функция для предсказания цены
def predict_price(new_data, model, scaler_features, scaler_price, label_encoders, mlb, device, make_model_class_dict):
    # Диапазоны классов цен
    class_ranges = {
        'Very Low': (4000.832, 4997.880),
        'Low': (5001.464, 19999.896),
        'Medium': (20002.568, 49997.940),
        'High': (100036.000, 149956.000),
        'Very High': (150055.000, 299922.000)
    }

    # Преобразование новых данных
    new_data['Age'] = 2025 - new_data['Year']
    new_data['LogMileage'] = np.log1p(new_data['Mileage'])

    # Кодирование категориальных признаков
    make_model_key = (new_data['Car Make'].iloc[0], new_data['Car Model'].iloc[0])
    if make_model_key in make_model_class_dict:
        # Используем числовые значения из словаря
        mapping = make_model_class_dict[make_model_key]
        new_data['Car Make'] = mapping['Car Make (numeric)']
        new_data['Car Model'] = mapping['Car Model (numeric)']
        predicted_class = mapping['Price_Class']
        new_data['Price_Class'] = mapping['Price_Class (numeric)']
        print(f"Найдена марка/модель {make_model_key}, класс: {predicted_class}")
    else:
        # Обработка неизвестных марок/моделей
        print(f"Предупреждение: марка/модель {make_model_key} отсутствует в обучающем датасете.")
        for column in ['Car Make', 'Car Model']:
            try:
                if new_data[column].iloc[0] not in label_encoders[column].classes_:
                    new_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])
                else:
                    new_data[column] = label_encoders[column].transform(new_data[column])
            except ValueError:
                new_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])
        # Логика для Price_Class
        if new_data['Age'].iloc[0] > 15 or new_data['Accident'].iloc[0] == 'Yes':
            predicted_class = 'Low'
            new_data['Price_Class'] = 1
        else:
            predicted_class = 'Medium'
            new_data['Price_Class'] = 2
        print(f"Установлен класс {predicted_class} для неизвестной марки/модели.")

    # Кодирование остальных категориальных признаков
    for column in ['Fuel Type', 'Color', 'Transmission', 'Condition', 'Accident']:
        try:
            if new_data[column].iloc[0] not in label_encoders[column].classes_:
                print(f"Предупреждение: {column} '{new_data[column].iloc[0]}' отсутствует в обучающем датасете.")
                new_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])
            else:
                new_data[column] = label_encoders[column].transform(new_data[column])
        except ValueError:
            new_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])

    # Обработка 'Options/Features'
    options = new_data['Options/Features'].apply(lambda x: [f for f in x.split(', ') if f in mlb.classes_])
    if not options.iloc[0]:
        print("Предупреждение: указанные опции отсутствуют в обучающем датасете.")
    options_encoded = pd.DataFrame(mlb.transform(options), columns=mlb.classes_)
    new_data = pd.concat([new_data, options_encoded], axis=1)
    new_data = new_data.drop('Options/Features', axis=1)

    # Удаление столбца Price, если он есть
    if 'Price' in new_data:
        new_data = new_data.drop('Price', axis=1)

    # Нормализация числовых признаков
    numeric_columns = ['Age', 'LogMileage']
    new_data[numeric_columns] = scaler_features.transform(new_data[numeric_columns])

    # Проверка порядка столбцов
    expected_columns = [
        'Car Make', 'Car Model', 'Year', 'Mileage', 'Fuel Type', 'Color', 'Transmission',
        'Condition', 'Accident', 'Price_Class', 'Age', 'LogMileage'
    ] + list(mlb.classes_)
    new_data = new_data.reindex(columns=expected_columns, fill_value=0)

    # Преобразование в тензор
    new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32).to(device)

    # Предсказание
    with torch.no_grad():
        normalized_prediction = model(new_data_tensor)
        prediction = scaler_price.inverse_transform(normalized_prediction.cpu().numpy())[0][0]

    # Корректировка предсказания на основе класса
    min_price, max_price = class_ranges[predicted_class]
    if prediction < min_price:
        prediction = min_price + (prediction - min_price) * 0.2
    elif prediction > max_price:
        prediction = max_price - (max_price - prediction) * 0.2

    return prediction, predicted_class

# 4. Примеры использования для тестирования
print("\n" + "="*70)
print("ТЕСТ 1: Honda Civic 2024, новый, малый пробег")
print("="*70)
new_car_1 = pd.DataFrame({
    'Car Make': ['Lamborghini'],
    'Car Model': ['Aventador'],
    'Year': [2024],
    'Mileage': [1242],
    'Fuel Type': ['Hybrid'],
    'Color': ['Black'],
    'Transmission': ['Automatic'],
    'Options/Features': ['Heated Seats'],
    'Condition': ['New'],
    'Accident': ['No']
})

predicted_price_1, predicted_class_1 = predict_price(
    new_data=new_car_1.copy(),
    model=model,
    scaler_features=scaler_features,
    scaler_price=scaler_price,
    label_encoders=label_encoders,
    mlb=mlb,
    device=device,
    make_model_class_dict=make_model_class_dict
)
print(f'Predicted Price: ${predicted_price_1:.2f}')
print(f'Price Class: {predicted_class_1}')

print("\n" + "="*70)
print("ТЕСТ 2: Honda Civic 2004, старый, большой пробег")
print("="*70)
new_car_2 = pd.DataFrame({
    'Car Make': ['Lamborghini'],
    'Car Model': ['Aventador'],
    'Year': [2004],
    'Mileage': [150000],
    'Fuel Type': ['Gasoline'],
    'Color': ['Black'],
    'Transmission': ['Automatic'],
    'Options/Features': ['Heated Seats'],
    'Condition': ['Used'],
    'Accident': ['Yes']
})

predicted_price_2, predicted_class_2 = predict_price(
    new_data=new_car_2.copy(),
    model=model,
    scaler_features=scaler_features,
    scaler_price=scaler_price,
    label_encoders=label_encoders,
    mlb=mlb,
    device=device,
    make_model_class_dict=make_model_class_dict
)
print(f'Predicted Price: ${predicted_price_2:.2f}')
print(f'Price Class: {predicted_class_2}')

print("\n" + "="*70)
print("ТЕСТ 3: Honda Civic 2015, средний возраст, средний пробег")
print("="*70)
new_car_3 = pd.DataFrame({
    'Car Make': ['Honda'],
    'Car Model': ['Civic'],
    'Year': [2015],
    'Mileage': [75000],
    'Fuel Type': ['Gasoline'],
    'Color': ['Black'],
    'Transmission': ['Automatic'],
    'Options/Features': ['GPS, Bluetooth'],
    'Condition': ['Used'],
    'Accident': ['No']
})

predicted_price_3, predicted_class_3 = predict_price(
    new_data=new_car_3.copy(),
    model=model,
    scaler_features=scaler_features,
    scaler_price=scaler_price,
    label_encoders=label_encoders,
    mlb=mlb,
    device=device,
    make_model_class_dict=make_model_class_dict
)
print(f'Predicted Price: ${predicted_price_3:.2f}')
print(f'Price Class: {predicted_class_3}')

print("\n" + "="*70)
print(f"Разница между новым (2024) и старым (2004): ${predicted_price_1 - predicted_price_2:.2f}")
print(f"Разница между новым (2024) и средним (2015): ${predicted_price_1 - predicted_price_3:.2f}")
print("="*70)