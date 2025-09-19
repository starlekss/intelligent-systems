import pandas as pd
import numpy as np

df = pd.read_csv('ncr_ride_bookings.csv')

print("Первые 5 строк датасета:")
print(df.head())
print("\n" + "="*80 + "\n")

print("Общая информация о датасете:")
print(df.info())
print("\n" + "="*80 + "\n")

print("Статистическое описание числовых столбцов:")
print(df.describe())
print("\n" + "="*80 + "\n")

print(f"Количество строк и столбцов: {df.shape}")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")
print("\n" + "="*80 + "\n")

selected_columns = ['Booking ID', 'Date', 'Time', 'Booking Status', 'Vehicle Type', 'Payment Method']
subset_df = df[selected_columns]
print("Выбранные столбцы (первые 5 строк):")
print(subset_df.head())
print("\n" + "="*80 + "\n")

cancelled_by_driver = df[df['Booking Status'] == 'Cancelled by Driver']
print(f"Бронирования, отмененные водителем: {len(cancelled_by_driver)} записей")
print("Первые 5 отмененных водителем:")
print(cancelled_by_driver.head())
print("\n" + "="*80 + "\n")

auto_high_value = df[(df['Vehicle Type'] == 'Auto') & (df['Booking Value'] > 500)]
print(f"Бронирования на Auto с стоимостью > 500: {len(auto_high_value)} записей")
print("Первые 5 записей:")
print(auto_high_value.head())
print("\n" + "="*80 + "\n")

df['booking_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

march_2024_bookings = df[
    (df['booking_datetime'] >= '2024-03-01') &
    (df['booking_datetime'] <= '2024-03-31 23:59:59')
]
print(f"Бронирования за март 2024 года: {len(march_2024_bookings)} записей")
print("Первые 5 записей за март 2024:")
print(march_2024_bookings.head())
