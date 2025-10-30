import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


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

        self.df = pd.DataFrame({
            'distance': distances,
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

        return self.mae, self.r2

    def predict_price(self, distance):
        """Предсказание цены для заданного расстояния"""
        if not self.is_trained:
            raise Exception("Модель не обучена")

        price = self.model.predict([[distance]])[0]
        return max(price, 40)

    def create_visualization(self, test_distance=None):
        """Создание основной визуализации"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.scatter(self.df['distance'], self.df['price'], alpha=0.6, s=50)
        x_range = np.linspace(0, 55, 100).reshape(-1, 1)
        y_range = self.model.predict(x_range)
        ax1.plot(x_range, y_range, 'red', linewidth=2, label='Линия регрессии')

        if test_distance is not None:
            test_price = self.predict_price(test_distance)
            ax1.scatter([test_distance], [test_price], color='yellow', s=200,
                        marker='*', edgecolors='black', label=f'Тест: {test_price:.1f} руб')

        ax1.set_xlabel('Расстояние (км)')
        ax1.set_ylabel('Стоимость (руб)')
        ax1.set_title('Зависимость стоимости от расстояния')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        y_pred = self.model.predict(self.df[['distance']].values)
        errors = self.df['price'].values - y_pred
        ax2.hist(errors, bins=15, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Ошибка предсказания (руб)')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение ошибок')
        ax2.grid(True, alpha=0.3)

        time_price = self.df.groupby('time_of_day')['price'].mean()
        ax3.bar(time_price.index, time_price.values, alpha=0.7)
        ax3.set_xlabel('Время суток')
        ax3.set_ylabel('Средняя стоимость (руб)')
        ax3.set_title('Стоимость по времени суток')

        numeric_df = self.df[['distance', 'price']].corr()
        sns.heatmap(numeric_df, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Корреляционная матрица')

        plt.tight_layout()
        plt.show()

    def plot_predictions_range(self):
        """Визуализация предсказаний в диапазоне"""
        distances = np.arange(1, 51)
        prices = [self.predict_price(dist) for dist in distances]

        plt.figure(figsize=(10, 6))
        plt.plot(distances, prices, 'b-', linewidth=2)
        plt.xlabel('Расстояние (км)')
        plt.ylabel('Стоимость (руб)')
        plt.title('Предсказанная стоимость поездки')
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    predictor = RidePricePredictor()

    print("Ride Price Predictor")
    print("=" * 30)

    data = predictor.generate_sample_data(200)
    print(f"Сгенерировано {len(data)} записей")

    mae, r2 = predictor.train_model()
    print(f"Модель обучена | Ошибка: {mae:.1f} руб | R²: {r2:.3f}")

    print("\nБыстрые предсказания:")
    test_distances = [5, 15, 25, 35, 45]
    for dist in test_distances:
        price = predictor.predict_price(dist)
        print(f" {dist} км → {price:.1f} руб")

    predictor.create_visualization(test_distances[1])
    predictor.plot_predictions_range()


if __name__ == "__main__":
    main()