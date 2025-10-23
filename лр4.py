import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


class RidePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def generate_sample_data(self, n_samples=200):
        """Генерация синтетических данных о поездках"""
        np.random.seed(42)

        distances = np.random.uniform(1, 50, n_samples)

        base_price = 50
        price_per_km = 15
        noise = np.random.normal(0, 20, n_samples)

        prices = base_price + price_per_km * distances + noise
        prices = np.maximum(prices, 40)

        time_per_km = 2
        time_noise = np.random.normal(0, 5, n_samples)
        durations = time_per_km * distances + time_noise
        durations = np.maximum(durations, 5)  # Минимальное время 5 мин

        self.df = pd.DataFrame({
            'distance': distances,
            'duration': durations,
            'price': prices,
            'time_of_day': np.random.choice(['Утро', 'День', 'Вечер', 'Ночь'], n_samples),
            'day_of_week': np.random.choice(['Будни', 'Выходные'], n_samples)
        })

        return self.df

    def train_model(self):
        """Обучение модели"""
        X = self.df[['distance']].values
        y = self.df['price'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)

        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)

        return self.mae, self.r2

    def predict_price(self, distance):
        """Предсказание цены для заданного расстояния"""
        if not self.is_trained:
            raise Exception("Модель не обучена")

        price = self.model.predict([[distance]])[0]
        return max(price, 40)  # Минимальная цена 40 руб

    def create_comprehensive_visualization(self, test_distance=None):
        """Создание комплексной визуализации"""
        fig = plt.figure(figsize=(20, 12))

        gs = fig.add_gridspec(3, 3)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[2, :])

        colors = self.df['distance'] / self.df['distance'].max()
        scatter = ax1.scatter(self.df['distance'], self.df['price'],
                              c=colors, cmap='viridis', alpha=0.7, s=60)

        x_range = np.linspace(0, 55, 100).reshape(-1, 1)
        y_range = self.model.predict(x_range)
        ax1.plot(x_range, y_range, 'red', linewidth=4,
                 label='Линия регрессии', alpha=0.8)

        if test_distance is not None:
            test_price = self.predict_price(test_distance)
            ax1.scatter([test_distance], [test_price],
                        color='yellow', s=300, marker='*',
                        edgecolors='black', linewidth=3,
                        label=f'Тестовое предсказание: {test_price:.1f} руб',
                        zorder=5)

        ax1.set_xlabel('Расстояние (км)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Стоимость (руб)', fontsize=12, fontweight='bold')
        ax1.set_title('Зависимость стоимости поездки от расстояния\n',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 55)
        plt.colorbar(scatter, ax=ax1, label='Нормализованное расстояние')

        X_test = self.df[['distance']].values
        y_true = self.df['price'].values
        y_pred = self.model.predict(X_test)
        errors = y_true - y_pred

        ax2.hist(errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Ошибка предсказания (руб)', fontweight='bold')
        ax2.set_ylabel('Частота', fontweight='bold')
        ax2.set_title('Распределение ошибок модели', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, f'Средняя ошибка: {np.mean(np.abs(errors)):.1f} руб',
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax3.hist(self.df['price'], bins=20, alpha=0.7, color='lightgreen',
                 edgecolor='black', density=True)
        ax3.axvline(x=self.df['price'].mean(), color='green', linestyle='--',
                    linewidth=2, label=f'Среднее: {self.df["price"].mean():.1f} руб')
        ax3.set_xlabel('Стоимость поездки (руб)', fontweight='bold')
        ax3.set_ylabel('Плотность', fontweight='bold')
        ax3.set_title('Распределение стоимости поездок', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.axis('off')

        metrics_text = (
            "МЕТРИКИ МОДЕЛИ:\n\n"
            f"• Средняя абсолютная ошибка (MAE): {self.mae:.2f} руб\n"
            f"• Средняя квадратичная ошибка (MSE): {self.mse:.2f}\n"
            f"• Коэффициент детерминации (R²): {self.r2:.4f}\n"
            f"• Стандартное отклонение: {np.std(errors):.2f} руб\n\n"
            "ПАРАМЕТРЫ МОДЕЛИ:\n\n"
            f"• Коэффициент (цена за км): {self.model.coef_[0]:.2f} руб/км\n"
            f"• Базовая цена: {self.model.intercept_:.2f} руб\n"
            f"• Минимальная цена поездки: 40.00 руб\n\n"
        )

        if test_distance is not None:
            metrics_text += (
                "ТЕСТОВОЕ ПРЕДСКАЗАНИЕ:\n\n"
                f"• Расстояние: {test_distance} км\n"
                f"• Предсказанная стоимость: {test_price:.1f} руб\n"
                f"• Расчетное время: {test_distance * 2:.0f} мин\n"
            )

        ax4.text(0.02, 0.98, metrics_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=1", facecolor="lightblue",
                           alpha=0.8, edgecolor='navy'))

        plt.tight_layout()
        plt.show()

    def create_additional_visualizations(self):
        """Дополнительные визуализации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # График 1: Стоимость vs Время
        scatter = axes[0, 0].scatter(self.df['duration'], self.df['price'],
                                     c=self.df['distance'], cmap='plasma',
                                     alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Время поездки (мин)', fontweight='bold')
        axes[0, 0].set_ylabel('Стоимость (руб)', fontweight='bold')
        axes[0, 0].set_title('Стоимость vs Время поездки', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Расстояние (км)')

        time_price_data = self.df.groupby('time_of_day')['price'].mean()
        bars = axes[0, 1].bar(time_price_data.index, time_price_data.values,
                              color=['lightblue', 'lightgreen', 'lightcoral', 'lavender'])
        axes[0, 1].set_xlabel('Время суток', fontweight='bold')
        axes[0, 1].set_ylabel('Средняя стоимость (руб)', fontweight='bold')
        axes[0, 1].set_title('Средняя стоимость по времени суток', fontweight='bold')

        for bar, value in zip(bars, time_price_data.values):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        day_price_data = self.df.groupby('day_of_week')['price'].mean()
        bars = axes[1, 0].bar(day_price_data.index, day_price_data.values,
                              color=['skyblue', 'lightpink'])
        axes[1, 0].set_xlabel('День недели', fontweight='bold')
        axes[1, 0].set_ylabel('Средняя стоимость (руб)', fontweight='bold')
        axes[1, 0].set_title('Средняя стоимость по дням недели', fontweight='bold')

        for bar, value in zip(bars, day_price_data.values):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        numeric_df = self.df[['distance', 'duration', 'price']].corr()
        sns.heatmap(numeric_df, annot=True, cmap='coolwarm', center=0,
                    ax=axes[1, 1], square=True, cbar_kws={'shrink': 0.8})
        axes[1, 1].set_title('Корреляционная матрица', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_prediction_range(self, min_dist=1, max_dist=50, step=1):
        """Визуализация предсказаний в диапазоне"""
        distances = np.arange(min_dist, max_dist + step, step)
        prices = [self.predict_price(dist) for dist in distances]

        plt.figure(figsize=(12, 6))
        plt.plot(distances, prices, 'b-', linewidth=3, alpha=0.7,
                 label='Предсказанная стоимость')
        plt.fill_between(distances, prices, alpha=0.2, color='blue')

        plt.xlabel('Расстояние (км)', fontweight='bold')
        plt.ylabel('Стоимость (руб)', fontweight='bold')
        plt.title('Предсказанная стоимость поездки для разных расстояний',
                  fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()

        typical_distances = [3, 5, 10, 15, 20, 30, 40, 50]
        for dist in typical_distances:
            if min_dist <= dist <= max_dist:
                price = self.predict_price(dist)
                plt.plot(dist, price, 'ro', markersize=8)
                plt.annotate(f'{dist} км\n{price:.0f} руб',
                             (dist, price),
                             xytext=(10, 10), textcoords='offset points',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                             fontweight='bold')

        plt.show()


def main():
    predictor = RidePricePredictor()

    print("Ride Price Predictor — Pro+")
    print("=" * 50)

    data = predictor.generate_sample_data(300)
    print(f"Сгенерировано {len(data)} записей о поездках")
    print(f"Диапазон расстояний: {data['distance'].min():.1f} - {data['distance'].max():.1f} км")
    print(f"Диапазон цен: {data['price'].min():.1f} - {data['price'].max():.1f} руб")

    mae, r2 = predictor.train_model()
    print(f"\n Модель обучена")
    print(f" Средняя ошибка: {mae:.1f} руб")
    print(f" Точность (R²): {r2:.3f}")

    print("\n" + "=" * 50)
    print("Quick Predict Demo")

    test_distances = [3.0, 8.5, 15.0, 27.5, 42.0]

    for distance in test_distances:
        price = predictor.predict_price(distance)
        print(f" Расстояние: {distance:5.1f} км → Стоимость: {price:6.1f} руб")

    print(f"\n Основная визуализация для расстояния {test_distances[1]} км:")
    predictor.create_comprehensive_visualization(test_distances[1])

    print("\n Дополнительные визуализации:")
    predictor.create_additional_visualizations()

    print("\n Визуализация предсказаний в диапазоне 1-50 км:")
    predictor.plot_prediction_range(1, 50)

    print("\n" + "=" * 50)
    print("Визуализация завершена!")


if __name__ == "__main__":
    main()