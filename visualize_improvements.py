"""
Сравнение старой и новой архитектуры модели
Визуализация изменений для улучшения train loss
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.suptitle('Сравнение архитектур: Старая vs Улучшенная', fontsize=18, fontweight='bold')

# Цвета
color_old = '#ff6b6b'
color_new = '#51cf66'
color_layer = '#4dabf7'
color_activation = '#ffd43b'
color_dropout = '#ff8787'

def draw_architecture(ax, title, layers, annotations, color_scheme):
    """Рисует архитектуру нейросети"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    y_pos = len(layers)
    
    for i, (layer_name, layer_info) in enumerate(layers):
        # Рисуем слой
        width = layer_info.get('width', 6)
        height = 0.6
        x_center = 5
        
        # Определяем цвет
        if 'Linear' in layer_name:
            color = color_layer
        elif 'Activation' in layer_name or 'ReLU' in layer_name or 'LeakyReLU' in layer_name:
            color = color_activation
        elif 'Dropout' in layer_name:
            color = color_dropout
        else:
            color = '#dee2e6'
        
        # Рисуем прямоугольник
        rect = FancyBboxPatch(
            (x_center - width/2, y_pos - height/2),
            width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Текст слоя
        ax.text(x_center, y_pos, layer_name, 
                ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Аннотация справа
        if i in annotations:
            ax.text(x_center + width/2 + 0.5, y_pos, annotations[i],
                   ha='left', va='center', fontsize=9, 
                   style='italic', color='#495057')
        
        # Стрелка к следующему слою
        if i < len(layers) - 1:
            arrow = FancyArrowPatch(
                (x_center, y_pos - height/2 - 0.1),
                (x_center, y_pos - height/2 - 0.9),
                arrowstyle='->,head_width=0.4,head_length=0.4',
                color='black',
                linewidth=2
            )
            ax.add_patch(arrow)
        
        y_pos -= 1

# СТАРАЯ АРХИТЕКТУРА
old_layers = [
    ('Input Layer', {'width': 4}),
    ('Linear(input → 256)', {'width': 7}),
    ('ReLU', {'width': 4}),
    ('BatchNorm1d(256)', {'width': 6}),
    ('Dropout(0.4)', {'width': 5}),
    ('Linear(256 → 128)', {'width': 7}),
    ('ReLU', {'width': 4}),
    ('BatchNorm1d(128)', {'width': 6}),
    ('Dropout(0.4)', {'width': 5}),
    ('Linear(128 → 64)', {'width': 6.5}),
    ('ReLU', {'width': 4}),
    ('BatchNorm1d(64)', {'width': 5.5}),
    ('Dropout(0.4)', {'width': 5}),
    ('Linear(64 → 32)', {'width': 6}),
    ('ReLU', {'width': 4}),
    ('BatchNorm1d(32)', {'width': 5.5}),
    ('Dropout(0.4)', {'width': 5}),
    ('Linear(32 → 1)', {'width': 5.5}),
    ('Output', {'width': 4}),
]

old_annotations = {
    4: '❌ Слишком высокий!',
    8: '❌ Слишком высокий!',
    12: '❌ Слишком высокий!',
    16: '❌ Слишком высокий!',
}

# НОВАЯ АРХИТЕКТУРА
new_layers = [
    ('Input Layer', {'width': 4}),
    ('Linear(input → 128)', {'width': 7}),
    ('LeakyReLU(0.1)', {'width': 5}),
    ('BatchNorm1d(128)', {'width': 6}),
    ('Dropout(0.2)', {'width': 5}),
    ('Linear(128 → 64)', {'width': 6.5}),
    ('LeakyReLU(0.1)', {'width': 5}),
    ('BatchNorm1d(64)', {'width': 5.5}),
    ('Dropout(0.2)', {'width': 5}),
    ('Linear(64 → 32)', {'width': 6}),
    ('LeakyReLU(0.1)', {'width': 5}),
    ('BatchNorm1d(32)', {'width': 5.5}),
    ('Dropout(0.15)', {'width': 5}),
    ('Linear(32 → 1)', {'width': 5.5}),
    ('Output', {'width': 4}),
]

new_annotations = {
    2: '✅ Лучшие градиенты',
    4: '✅ Снижен до 0.2',
    8: '✅ Снижен до 0.2',
    12: '✅ Снижен до 0.15',
}

draw_architecture(ax1, 'СТАРАЯ АРХИТЕКТУРА\n(4 скрытых слоя)', old_layers, old_annotations, color_old)
draw_architecture(ax2, 'УЛУЧШЕННАЯ АРХИТЕКТУРА\n(3 скрытых слоя)', new_layers, new_annotations, color_new)

plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
print("✅ График архитектуры сохранен: architecture_comparison.png")

# ВТОРОЙ ГРАФИК: Сравнение гиперпараметров
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
fig.suptitle('Сравнение гиперпараметров и улучшений', fontsize=18, fontweight='bold')

# Данные для сравнения
comparisons = [
    ('Batch Size', '64', '16', '4x меньше → больше обновлений', color_new),
    ('Dropout Rate', '0.4', '0.2-0.15', '2x меньше → лучше обучение', color_new),
    ('Learning Rate', '0.001', '0.003 + scheduler', '3x + адаптация', color_new),
    ('Loss Function', 'MSE', 'Huber Loss', 'Устойчива к выбросам', color_new),
    ('Optimizer', 'Adam', 'AdamW', 'Лучшая регуляризация', color_new),
    ('Weight Decay', '1e-5', '1e-4', '10x сильнее', color_new),
    ('Activation', 'ReLU', 'LeakyReLU', 'Нет dying neurons', color_new),
    ('LR Scheduler', '❌ Нет', '✅ ReduceLROnPlateau', 'Адаптивная скорость', color_new),
    ('Early Stopping', '❌ Нет', '✅ Patience=50', 'Оптимальные эпохи', color_new),
    ('Gradient Clipping', '❌ Нет', '✅ max_norm=1.0', 'Стабильность', color_new),
    ('Weight Init', '❌ Случайная', '✅ Kaiming/He', 'Быстрая сходимость', color_new),
]

y_start = 0.95
y_step = 0.08

# Заголовки колонок
ax.text(0.15, y_start, 'Параметр', fontsize=14, fontweight='bold', ha='center')
ax.text(0.35, y_start, 'Было', fontsize=14, fontweight='bold', ha='center')
ax.text(0.55, y_start, 'Стало', fontsize=14, fontweight='bold', ha='center')
ax.text(0.80, y_start, 'Эффект', fontsize=14, fontweight='bold', ha='center')

# Линия под заголовками
ax.plot([0.05, 0.95], [y_start - 0.02, y_start - 0.02], 'k-', linewidth=2)

y_pos = y_start - 0.05

for i, (param, old_val, new_val, effect, color) in enumerate(comparisons):
    # Фон для строки
    if i % 2 == 0:
        rect = plt.Rectangle((0.05, y_pos - 0.03), 0.9, 0.07, 
                            facecolor='#f8f9fa', edgecolor='none', alpha=0.5)
        ax.add_patch(rect)
    
    # Текст
    ax.text(0.15, y_pos, param, fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(0.35, y_pos, old_val, fontsize=10, ha='center', va='center', 
            color='#dc3545', fontweight='bold')
    ax.text(0.55, y_pos, new_val, fontsize=10, ha='center', va='center', 
            color='#28a745', fontweight='bold')
    ax.text(0.80, y_pos, effect, fontsize=9, ha='center', va='center', style='italic')
    
    y_pos -= y_step

# Итоговая информация
y_bottom = 0.08
ax.text(0.5, y_bottom, '🎯 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: Train Loss снизится в 2-3 раза!', 
        fontsize=14, ha='center', fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_new, alpha=0.3))

plt.tight_layout()
plt.savefig('hyperparameters_comparison.png', dpi=150, bbox_inches='tight')
print("✅ График гиперпараметров сохранен: hyperparameters_comparison.png")

# ТРЕТИЙ ГРАФИК: Ожидаемое поведение loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ожидаемое улучшение Train Loss', fontsize=16, fontweight='bold')

epochs = np.arange(1, 201)

# Старая модель (симуляция)
old_train_loss = 0.18 * np.exp(-0.01 * epochs) + 0.15 + np.random.normal(0, 0.01, len(epochs))
old_test_loss = 0.20 * np.exp(-0.01 * epochs) + 0.16 + np.random.normal(0, 0.015, len(epochs))

# Новая модель (симуляция)
new_train_loss = 0.15 * np.exp(-0.03 * epochs) + 0.05 + np.random.normal(0, 0.005, len(epochs))
new_test_loss = 0.17 * np.exp(-0.03 * epochs) + 0.06 + np.random.normal(0, 0.008, len(epochs))

# График 1: Старая модель
ax1.plot(epochs, old_train_loss, label='Train Loss', color='#ff6b6b', linewidth=2)
ax1.plot(epochs, old_test_loss, label='Test Loss', color='#ff8787', linewidth=2, linestyle='--')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Старая модель\n(медленная сходимость)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.4)
ax1.axhline(y=0.15, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Застревает на 0.15')

# График 2: Новая модель
ax2.plot(epochs, new_train_loss, label='Train Loss', color='#51cf66', linewidth=2)
ax2.plot(epochs, new_test_loss, label='Test Loss', color='#69db7c', linewidth=2, linestyle='--')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (Huber)', fontsize=12)
ax2.set_title('Улучшенная модель\n(быстрая сходимость)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.4)
ax2.axhline(y=0.05, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Достигает 0.05')

# Аннотации
ax1.annotate('Медленное\nулучшение', xy=(100, 0.16), xytext=(130, 0.25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

ax2.annotate('Быстрое\nулучшение', xy=(50, 0.08), xytext=(80, 0.18),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('expected_loss_improvement.png', dpi=150, bbox_inches='tight')
print("✅ График ожидаемого улучшения сохранен: expected_loss_improvement.png")

print("\n" + "="*60)
print("Все графики успешно созданы!")
print("="*60)
print("\nФайлы:")
print("  📊 architecture_comparison.png - Сравнение архитектур")
print("  📊 hyperparameters_comparison.png - Сравнение гиперпараметров")
print("  📊 expected_loss_improvement.png - Ожидаемое улучшение loss")
print("\nТеперь запустите: python main_improved.py")
