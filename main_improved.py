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

# УЛУЧШЕНИЕ 1: Уменьшение batch_size для малого датасета
# Для большого датасета (18k+ образцов) используем batch_size 32
batch_size = 32 if len(train_dataset) > 1000 else 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 10. Определение улучшенной модели
class ImprovedCarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(ImprovedCarPricePredictor, self).__init__()
        # УЛУЧШЕНИЕ 2: Уменьшение глубины сети для малого датасета
        # УЛУЧШЕНИЕ 3: Снижение dropout с 0.4 до 0.2-0.3
        # УЛУЧШЕНИЕ 4: Добавление LeakyReLU вместо ReLU для лучших градиентов
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
        
        # УЛУЧШЕНИЕ 5: Инициализация весов Xavier/He
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# 11. Инициализация модели
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Принудительно использовать CPU
print(f"Используется устройство: {device}")

# Очистка памяти GPU если используется CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

model = ImprovedCarPricePredictor(input_size=X_train.shape[1]).to(device)

# УЛУЧШЕНИЕ 6: Использование Huber Loss вместо MSE (более устойчива к выбросам)
criterion = nn.HuberLoss(delta=1.0)

# УЛУЧШЕНИЕ 7: Увеличение learning rate и добавление AdamW для лучшей регуляризации
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

# УЛУЧШЕНИЕ 8: Добавление scheduler для адаптивного learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
)

# УЛУЧШЕНИЕ 9: Early stopping
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=50, min_delta=1e-5)

# 12. Обучение модели с улучшениями
num_epochs = 500  # Увеличиваем максимум, но используем early stopping
train_losses = []
test_losses = []
train_maes = []
test_maes = []
best_test_loss = float('inf')

print(f"Начало обучения модели на {len(X_train)} образцах...")
print(f"Размер батча: {batch_size}, Learning rate: 0.003")
print(f"Количество батчей за эпоху: Train={len(train_loader)}, Test={len(test_loader)}")

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
        
        # УЛУЧШЕНИЕ 10: Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
        train_predictions.append(outputs.cpu().detach().numpy())
        train_targets.append(targets.cpu().detach().numpy())
        
        # Очистка памяти GPU каждые 100 батчей
        if torch.cuda.is_available() and len(train_predictions) % 100 == 0:
            torch.cuda.empty_cache()

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
    
    # Обновление scheduler
    scheduler.step(test_loss)
    
    # Сохранение лучшей модели
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'car_price_predictor_improved_best.pth')

    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}')
    
    # Early stopping
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print(f"Early stopping на эпохе {epoch + 1}")
        break

# Загрузка лучшей модели
model.load_state_dict(torch.load('car_price_predictor_improved_best.pth'))
model.eval()

# 13. Оценка на тестовых данных с лучшей моделью
test_predictions_final = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_predictions_final.append(outputs.cpu().numpy())
test_predictions = np.concatenate(test_predictions_final)

# Оценка Price_Class
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
print(f'\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===')
print(f'Best Test Loss: {best_test_loss:.4f}')
print(f'Final Test MAE: ${test_maes[-1]:.2f}')
print(f'F1 Score for Price_Class (Test): {f1:.4f}')

# 14. Построение графиков
# График потерь
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linewidth=2)
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Huber Loss (Normalized)', fontsize=12)
plt.title('Training and Test Loss Over Epochs (Improved)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# График MAE
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_maes) + 1), train_maes, label='Train MAE', linewidth=2)
plt.plot(range(1, len(test_maes) + 1), test_maes, label='Test MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Absolute Error ($)', fontsize=12)
plt.title('Training and Test MAE Over Epochs (Improved)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_loss_mae_plot.png', dpi=150)
plt.show()

# Матрица ошибок для Price_Class
cm = confusion_matrix(price_class_test, price_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.xlabel('Predicted Price_Class', fontsize=12)
plt.ylabel('True Price_Class', fontsize=12)
plt.title('Confusion Matrix for Price_Class (Improved)', fontsize=14)
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png', dpi=150)
plt.show()

# Визуализация ошибок предсказания цен
test_targets_denorm = scaler_price.inverse_transform(y_test.values.reshape(-1, 1))
errors = test_targets_denorm - test_predictions_denorm
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Price Prediction Errors', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(test_targets_denorm, test_predictions_denorm, alpha=0.6)
plt.plot([test_targets_denorm.min(), test_targets_denorm.max()], 
         [test_targets_denorm.min(), test_targets_denorm.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.title('True vs Predicted Prices', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_error_distribution.png', dpi=150)
plt.show()

# 15. Сохранение модели и объектов
torch.save(model.state_dict(), 'car_price_predictor_improved.pth')
with open('scaler_features.pkl', 'wb') as f:
    pickle.dump(scaler_features, f)
with open('scaler_price.pkl', 'wb') as f:
    pickle.dump(scaler_price, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("\n=== Модель, объекты и графики сохранены ===")
print("Файлы:")
print("  - car_price_predictor_improved.pth (финальная модель)")
print("  - car_price_predictor_improved_best.pth (лучшая модель)")
print("  - improved_loss_mae_plot.png")
print("  - improved_confusion_matrix.png")
print("  - improved_error_distribution.png")
