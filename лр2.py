import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('ncr_ride_bookings.csv')

print("=" * 60)
print("Статистический обзор данных")
print("=" * 60)

# 1. Общая информация о данных
print("\n1. Общая информация:")
print(f"Всего записей: {len(df)}")
print(f"Количество столбцов: {len(df.columns)}")
print(f"Период данных: с {df['Date'].min()} по {df['Date'].max()}")

# 2. Пропущенные значения
print("\n2. Пропущенные значения по столбцам:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Пропущенные значения': missing_values,
    'Процент пропусков': missing_percentage.round(2)
})
print(missing_df[missing_df['Пропущенные значения'] > 0])

# 3. Уникальные значения в категориальных столбцах
print("\n3. Уникальные значения в категориальных столбцах:")

# Booking Status
print("\nСтатусы бронирования:")
booking_status_counts = df['Booking Status'].value_counts()
print(booking_status_counts)
print(f"Всего уникальных статусов: {df['Booking Status'].nunique()}")

# Vehicle Type
print("\nТипы транспортных средств:")
vehicle_type_counts = df['Vehicle Type'].value_counts()
print(vehicle_type_counts)
print(f"Всего уникальных типов транспорта: {df['Vehicle Type'].nunique()}")

# 4. Дополнительная статистика по категориальным столбцам
print("\n4. Дополнительная статистика:")

# Причины отмены клиентом
print("\nПричины отмены клиентом:")
if 'Reason for cancelling by Customer' in df.columns:
    cancel_reasons = df['Reason for cancelling by Customer'].value_counts(dropna=False)
    print(cancel_reasons)

# Причины отмены водителем
print("\nПричины отмены водителем:")
if 'Driver Cancellation Reason' in df.columns:
    driver_cancel_reasons = df['Driver Cancellation Reason'].value_counts(dropna=False)
    print(driver_cancel_reasons)

# 5. Статистика по числовым столбцам
print("\n5. Статистика по числовым столбцам:")
numeric_columns = df.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 0:
    print("Числовые столбцы:", list(numeric_columns))
    print("\nОсновные статистики:")
    print(df[numeric_columns].describe().round(2))
else:
    print("Числовые столбцы не найдены")

print("\n" + "=" * 60)
print("Анализ завершен")
print("=" * 60)