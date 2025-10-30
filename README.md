# intelligent-systems

# ЛР1 - Анализ датасет поездок 

1. Импорт библиотек и загрузка данных

`import pandas as pd`\
`import numpy as np`\
`df = pd.read_csv('ncr_ride_bookings.csv')`

**pandas** - для работы с табличными данными\
**numpy** - для математических операций\
_Загружается CSV файл с данными о бронированиях поездок_

2. Первичный анализ данных

`print(df.head())`       
`print(df.head())`      
`print(df.info())`      
`print(df.describe())`    
`print(df.shape)`

_Цель: Понять структуру данных, типы переменных, наличие пропусков_

3. Работа с подмножеством столбцов

`selected_columns = ['Booking ID', 'Date', 'Time', 'Booking Status', 'Vehicle Type', 'Payment Method']`\
`subset_df = df[selected_columns]`

+ **Booking ID** - идентификатор бронирования\
+ **Date и Time** - дата и время поездки\
+ **Booking Status** - статус бронирования\
+ **Vehicle Type** - тип транспортного средства\
+ **Payment Method** - способ оплаты\

4. Фильтрация данных
   
+ Отмененные водителем поездки\
`cancelled_by_driver = df[df['Booking Status'] == 'Cancelled by Driver']`
_Анализ: Сколько поездок отменили водители_\
+ Дорогие поездки на Auto\
`auto_high_value = df[(df['Vehicle Type'] == 'Auto') & (df['Booking Value'] > 500)]`
_Анализ: Поездки на авто-рикшах стоимостью более 500 единиц_

5. Работа с датами
`df['booking_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
march_2024_bookings = df[
    (df['booking_datetime'] >= '2024-03-01') &
    (df['booking_datetime'] <= '2024-03-31 23:59:59')
]`

_Создается: Единая колонка с датой и временем_\
_Фильтруются: Бронирования за март 2024 года_

# ЛР2 - Статистический обзор и анализ данных о бронированиях поездок

1. Загрузка данных и заголовок
   
`df = pd.read_csv('ncr_ride_bookings.csv')`

_Загружается CSV файл с данными о бронированиях поездок_

2. Общая информация о датасете

`print(f"Всего записей: {len(df)}")`\
`print(f"Количество столбцов: {len(df.columns)}")`\
`print(f"Период данных: с {df['Date'].min()} по {df['Date'].max()}")`

_Анализирует:_
+ Объем данных (количество строк)
+ Количество характеристик (столбцов)
+ Временной период покрытия данных
  
3. Анализ пропущенных значений

`missing_values = df.isnull().sum()`\
`missing_percentage = (df.isnull().sum() / len(df)) * 100`

_Вычисляет:_
+ Абсолютное количество пропусков по каждому столбцу
+ Процент пропусков от общего числа записей
+ Показывает только столбцы с пропусками

4. Анализ категориальных переменных
+ Статусы бронирования\
`booking_status_counts = df['Booking Status'].value_counts()`\
_Показывает: Распределение по статусам (например: "Completed", "Cancelled", "In Progress")_\
+ Типы транспортных средств\
`vehicle_type_counts = df['Vehicle Type'].value_counts()`\
_Показывает: Популярность разных типов транспорта (Auto, Car, Bike, etc.)_

5. Анализ причин отмен
+ Отмены клиентами\
`cancel_reasons = df['Reason for cancelling by Customer'].value_counts(dropna=False)`
_Анализирует: Почему клиенты отменяют поездки_\
+ Отмены водителями\
`driver_cancel_reasons = df['Driver Cancellation Reason'].value_counts(dropna=False)`
_Анализирует: Почему водители отменяют заказы_

6. Статистика числовых показателей
`numeric_columns = df.select_dtypes(include=[np.number]).columns`\
`df[numeric_columns].describe().round(2)`
_Анализирует числовые столбцы:_
+ **count** - количество непустых значений
+ **mean** - среднее значение
+ **std** - стандартное отклонение
+ **min/max** - минимальное/максимальное значение
+ **25%/50%/75%** - квартили распределения

# ЛР3 -  Создание графиков для анализа бизнес-метрик сервиса такси

1. Импорт библиотек и настройка

`import pandas as pd`\
`import matplotlib.pyplot as plt`\
`import seaborn as sns`\
`import numpy as np`\
`matplotlib.use('Agg')`

2. Загрузка данных

`df = pd.read_csv('ncr_ride_bookings.csv')`

3. Фильтрация данных

`completed_rides = df[(df['Booking Status'] == 'Completed') & (df['Booking Value'].notnull())]`

_Фильтрует только:_
+ Завершенные поездки (Booking Status == 'Completed')
+ С известной стоимостью (Booking Value.notnull())

4. Гистограмма стоимости поездок
`sns.histplot(completed_rides['Booking Value'], bins=50, kde=True)`

_Создает:
+ Гистограмму распределения стоимости поездок
+ 50 интервалов (bins)
+ Линию плотности распределения (KDE)
+ Сохраняет в файл: booking_value_histogram.png

6. Базовая статистика стоимости

`print(f"Средняя стоимость: {completed_rides['Booking Value'].mean():.2f}")`\
`print(f"Медианная стоимость: {completed_rides['Booking Value'].median():.2f}")`

_Вычисляет метрики:_
+ Среднее - общая тенденция
+ Медиана - устойчивая к выбросам
+ Минимум/Максимум - диапазон значений

6. Диаграмма рассеяния

`sns.scatterplot(data=scatter_data, x='Ride Distance', y='Booking Value', alpha=0.6)`

_Анализирует зависимость:_
+ По оси X: расстояние поездки
+ По оси Y: стоимость поездки
+ Сохраняет в файл: scatter_plot.png

7. Корреляционный анализ

`correlation = scatter_data['Ride Distance'].corr(scatter_data['Booking Value'])`

_Вычисляет коэффициент корреляции Пирсона:_
+ 1.0 - сильная прямая зависимость
+ 0.0 - отсутствие зависимости
+ -1.0 - сильная обратная зависимость\

# ЛР4 - Модель машинного обучения
## Структура класса RidePricePredictor
1. Инициализация

`def __init__(self):
    self.model = LinearRegression()
    self.is_trained = False`
    
_Создает модель линейной регрессии_
_Флаг **is_trained** отслеживает состояние обучения_

2. Генерация синтетических данных

`def generate_sample_data(self, n_samples=200):`

_Создает искусственные данные:_
+ distances: случайные расстояния от 1 до 50 км
+ prices: рассчитываются по формуле 50 + 15 * расстояние + шум
+ Добавляет категориальные признаки (время суток, день недели)

3. Обучение модели
   
`def train_model(self):
    X = self.df[['distance']].values  
    y = self.df['price'].values`

+ Разделяет данные на обучающую и тестовую выборки (80%/20%)
+ Обучает линейную регрессию
+ Вычисляет метрики качества:
 MAE (Mean Absolute Error) - средняя абсолютная ошибка
 R² (R-squared) - коэффициент детерминации

4. Предсказание цены

`def predict_price(self, distance):
   rice = self.model.predict([[distance]])[0]
   return max(price, 40)`
   
## Визуализации

1. Основная визуализация
+ График 1: Рассеяние + линия регрессии + тестовое предсказание
+ График 2: Распределение ошибок модели
+ График 3: Средняя стоимость по времени суток
+ График 4: Корреляционная матрица

2. График предсказаний

`def plot_predictions_range(self):`

_Показывает предсказанную стоимость для расстояний от 1 до 50 км_

## Работа программы

`def main():
    predictor = RidePricePredictor()
    data = predictor.generate_sample_data(200)
    mae, r2 = predictor.train_model()`
    
_Выполняет:_
+ Создание предсказателя
+ Генерацию 200 примеров поездок
+ Обучение модели
+ Тестовые предсказания для 5, 15, 25, 35, 45 км
+ Построение графиков

# ЛР5 - Система прогнозирования стоимости поездок такси с использованием машинного обучения и учетом множества факторов

1. Инициализация класса

`def __init__(self):
    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    self.is_trained = False
    self.label_encoders = {}`

+ **RandomForestRegressor** - ансамблевая модель на основе деревьев решений
+ **n_estimators=100** - количество деревьев в лесу
+ **random_state=42** - для воспроизводимости результатов
+ **is_trained** - флаг обучения модели
+ **label_encoders** - словарь для кодирования категориальных признаков

2. Генерация данных

`def generate_data(self, n_samples=1000):`

_Создает реалистичные данные с учетом факторов:_
+ **distance** - расстояние (1-50 км)
+ **traffic** - уровень трафика (0.1-1.0)
+ **time_of_day** - время суток (0-24 часа)
+ **surge** - коэффициент спроса (1.0, 1.2, 1.5, 2.0)
+ **vehicle_type** - тип автомобиля (Эконом/Комфорт/Бизнес)
+ **weather** - погодные условия

_Формула цены:_ 

`price = base_cost * time_factor * traffic_factor * weather_factor * vehicle_factor * surge`

3. Обучение модели

`def train_model(self):`

+ Кодирует категориальные признаки (vehicle, weather) в числовые
+ Разделяет данные на обучающую и тестовую выборки (80/20)
+ Обучает модель Random Forest
+ Вычисляет метрики качества: MAE и R²

4. Предсказание цены

`def predict_price(self, distance, traffic, time, surge, vehicle, weather):`

_Преобразует входные параметры в формат, понятный модели, и возвращает предсказанную цену_

5. Визуализация

`def show_plots(self):`

_Создает 4 графика:_
+ Расстояние vs Цена - scatter plot
+ Средняя цена по типам авто - bar plot
+ Средняя цена по погоде - bar plot
+ Матрица корреляций - heatmap
