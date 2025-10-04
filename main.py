import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
data = pd.read_csv('Updated_Car_Sales_Data_with_Classes_prepared.csv')

# 2. Feature Engineering
data['Age'] = 2025 - data['Year']
data['LogMileage'] = np.log1p(data['Mileage'])

# 3. Создание словаря для Car Make, Car Model, Price_Class
make_model_class_dict = {}
label_encoders = {}

# Кодирование категориальных признаков
for column in ['Car Make', 'Car Model', 'Fuel Type', 'Color', 'Transmission', 'Condition', 'Accident', 'Price_Class']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Заполнение словаря
for idx, row in data.iterrows():
    make_str = label_encoders['Car Make'].inverse_transform([row['Car Make']])[0]
    model_str = label_encoders['Car Model'].inverse_transform([row['Car Model']])[0]
    price_class_str = label_encoders['Price_Class'].inverse_transform([row['Price_Class']])[0]
    make_model_class_dict[(make_str, model_str)] = {
        'Price_Class': price_class_str,
        'Car Make (numeric)': int(row['Car Make']),
        'Car Model (numeric)': int(row['Car Model']),
        'Price_Class (numeric)': int(row['Price_Class'])
    }

# Сохранение словаря
with open('make_model_class_dict.pkl', 'wb') as f:
    pickle.dump(make_model_class_dict, f)
print("Словарь make_model_class_dict сохранён в 'make_model_class_dict.pkl'.")

# 4. Обработка 'Options/Features'
mlb = MultiLabelBinarizer()
options = data['Options/Features'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
options_encoded = pd.DataFrame(mlb.fit_transform(options), columns=mlb.classes_, index=data.index)
data = pd.concat([data, options_encoded], axis=1)
data = data.drop('Options/Features', axis=1)

# 5. Нормализация числовых признаков
scaler_features = StandardScaler()
numeric_columns = ['Age', 'LogMileage']
data[numeric_columns] = scaler_features.fit_transform(data[numeric_columns])

# 6. Нормализация целевой переменной (Price)
scaler_price = StandardScaler()
data['Price'] = scaler_price.fit_transform(data[['Price']])

# 7. Разделение данных
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
price_class_test = X_test['Price_Class']  # Для анализа классификации

# 8. Преобразование в тензоры
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 9. Создание DataLoader
class CarDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CarDataset(X_train_tensor, y_train_tensor)
test_dataset = CarDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 10. Определение модели
class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# 11. Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarPricePredictor(input_size=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 12. Обучение модели с сохранением истории потерь и метрик
num_epochs = 200
train_losses = []
test_losses = []
train_maes = []
test_maes = []

for epoch in range(num_epochs):
    # Обучение
    model.train()
    running_loss = 0.0
    train_predictions = []
    train_targets = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_predictions.append(outputs.cpu().detach().numpy())
        train_targets.append(targets.cpu().detach().numpy())

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_predictions = np.concatenate(train_predictions)
    train_targets = np.concatenate(train_targets)
    train_mae = mean_absolute_error(scaler_price.inverse_transform(train_targets),
                                    scaler_price.inverse_transform(train_predictions))
    train_maes.append(train_mae)

    # Тестирование
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    test_mae = mean_absolute_error(scaler_price.inverse_transform(test_targets),
                                   scaler_price.inverse_transform(test_predictions))
    test_maes.append(test_mae)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}')

# 13. Оценка Price_Class
# Предсказываем Price и определяем Price_Class
test_predictions_denorm = scaler_price.inverse_transform(test_predictions)
price_class_pred = np.zeros_like(test_predictions_denorm, dtype=int)
for i, price in enumerate(test_predictions_denorm):
    if price <= 4997.880:
        price_class_pred[i] = 0  # Very Low
    elif price <= 19999.896:
        price_class_pred[i] = 1  # Low
    elif price <= 49997.940:
        price_class_pred[i] = 2  # Medium
    elif price <= 149956.000:
        price_class_pred[i] = 3  # High
    else:
        price_class_pred[i] = 4  # Very High

f1 = f1_score(price_class_test, price_class_pred, average='weighted')
print(f'F1 Score for Price_Class (Test): {f1:.4f}')

# 14. Построение графиков
# График потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Normalized)')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

# График MAE
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_maes, label='Train MAE')
plt.plot(range(1, num_epochs + 1), test_maes, label='Test MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error ($)')
plt.title('Training and Test MAE Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('mae_plot.png')
plt.show()

# Матрица ошибок для Price_Class
cm = confusion_matrix(price_class_test, price_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.xlabel('Predicted Price_Class')
plt.ylabel('True Price_Class')
plt.title('Confusion Matrix for Price_Class')
plt.savefig('confusion_matrix.png')
plt.show()

# Визуализация ошибок предсказания цен
errors = scaler_price.inverse_transform(test_targets) - scaler_price.inverse_transform(test_predictions)
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, edgecolor='black')
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Price Prediction Errors')
plt.grid(True)
plt.savefig('error_distribution.png')
plt.show()

# 15. Сохранение модели и объектов
torch.save(model.state_dict(), 'car_price_predictor_with_classes.pth')
with open('scaler_features.pkl', 'wb') as f:
    pickle.dump(scaler_features, f)
with open('scaler_price.pkl', 'wb') as f:
    pickle.dump(scaler_price, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("Модель, объекты и графики сохранены.")