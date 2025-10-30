import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


class RidePricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.label_encoders = {}

    def generate_data(self, n_samples=1000):
        """Генерация данных с разными факторами"""
        np.random.seed(42)

        distances = np.random.uniform(1, 50, n_samples)
        traffic = np.random.uniform(0.1, 1.0, n_samples)
        time_of_day = np.random.uniform(0, 24, n_samples)
        surge = np.random.choice([1.0, 1.2, 1.5, 2.0], n_samples, p=[0.6, 0.2, 0.15, 0.05])
        vehicle_type = np.random.choice(['Эконом', 'Комфорт', 'Бизнес'], n_samples, p=[0.6, 0.3, 0.1])
        weather = np.random.choice(['Ясно', 'Дождь', 'Снег'], n_samples, p=[0.7, 0.2, 0.1])

        base_cost = 50 + 15 * distances
        time_factor = 1 + 0.2 * np.sin(time_of_day * np.pi / 12)
        traffic_factor = 1 + 0.3 * traffic
        weather_factor = np.where(weather == 'Дождь', 1.2, np.where(weather == 'Снег', 1.4, 1.0))
        vehicle_factor = np.where(vehicle_type == 'Эконом', 1.0, np.where(vehicle_type == 'Комфорт', 1.3, 1.7))

        price = base_cost * time_factor * traffic_factor * weather_factor * vehicle_factor * surge
        price += np.random.normal(0, 15, n_samples)
        price = np.maximum(price, 40)

        self.df = pd.DataFrame({
            'distance': distances, 'traffic': traffic, 'time': time_of_day,
            'surge': surge, 'vehicle': vehicle_type, 'weather': weather,
            'price': price
        })

        return self.df

    def train_model(self):
        """Обучение модели"""
        for col in ['vehicle', 'weather']:
            le = LabelEncoder()
            self.df[col + '_code'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        features = ['distance', 'traffic', 'time', 'surge', 'vehicle_code', 'weather_code']
        X = self.df[features].values
        y = self.df['price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

        return self.mae, self.r2

    def predict_price(self, distance, traffic, time, surge, vehicle, weather):
        """Предсказание цены"""
        if not self.is_trained:
            raise Exception("Модель не обучена")

        vehicle_code = self.label_encoders['vehicle'].transform([vehicle])[0]
        weather_code = self.label_encoders['weather'].transform([weather])[0]

        features = [[distance, traffic, time, surge, vehicle_code, weather_code]]
        price = self.model.predict(features)[0]

        return max(price, 40)

    def show_plots(self):
        """Показать основные графики"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].scatter(self.df['distance'], self.df['price'], alpha=0.6)
        axes[0, 0].set_xlabel('Расстояние (км)')
        axes[0, 0].set_ylabel('Цена (руб)')
        axes[0, 0].set_title('Цена vs Расстояние')

        vehicle_prices = self.df.groupby('vehicle')['price'].mean()
        vehicle_prices.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Цена по типу авто')

        weather_prices = self.df.groupby('weather')['price'].mean()
        weather_prices.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Цена по погоде')

        numeric_df = self.df[['distance', 'traffic', 'time', 'surge', 'price']].corr()
        sns.heatmap(numeric_df, annot=True, ax=axes[1, 1])
        axes[1, 1].set_title('Корреляции')

        plt.tight_layout()
        plt.show()


def main():
    predictor = RidePricePredictor()

    print("Предиктор стоимости поездок")
    print("=" * 30)

    data = predictor.generate_data(500)
    print(f"Данные: {len(data)} записей")

    mae, r2 = predictor.train_model()
    print(f"Модель: MAE={mae:.1f} руб, R²={r2:.3f}")

    print("\nТестовые предсказания:")
    test_rides = [
        (10, 0.3, 8, 1.0, 'Эконом', 'Ясно'),
        (25, 0.8, 18, 1.5, 'Комфорт', 'Дождь')
    ]

    for dist, traffic, time, surge, vehicle, weather in test_rides:
        price = predictor.predict_price(dist, traffic, time, surge, vehicle, weather)
        print(f"{dist} км, {vehicle}, {weather} → {price:.1f} руб")

    predictor.show_plots()


if __name__ == "__main__":
    main()