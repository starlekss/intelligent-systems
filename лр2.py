import pandas as pd
import numpy as np

df = pd.read_csv('ncr_ride_bookings.csv')

print("=" * 60)
print("СТАТИСТИЧЕСКИЙ ОБЗОР ДАННЫХ")
print("=" * 60)

print("\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"Всего записей: {len(df)}")
print(f"Количество столбцов: {len(df.columns)}")
print(f"Период данных: с {df['Date'].min()} по {df['Date'].max()}")

print("\n2. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ПО СТОЛБЦАМ:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Пропущенные значения': missing_values,
    'Процент пропусков': missing_percentage.round(2)
})
print(missing_df[missing_df['Пропущенные значения'] > 0])

print("\n3. УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В КАТЕГОРИАЛЬНЫХ СТОЛБЦАХ:")

print("\nBooking Status:")
booking_status_counts = df['Booking Status'].value_counts()
print(booking_status_counts)
print(f"Всего уникальных статусов: {df['Booking Status'].nunique()}")

print("\nVehicle Type:")
vehicle_type_counts = df['Vehicle Type'].value_counts()
print(vehicle_type_counts)
print(f"Всего уникальных типов транспорта: {df['Vehicle Type'].nunique()}")

print("\n4. ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")

print("\nПричины отмены клиентом (Reason for cancelling by Customer):")
if 'Reason for cancelling by Customer' in df.columns:
    cancel_reasons = df['Reason for cancelling by Customer'].value_counts(dropna=False)
    print(cancel_reasons)

print("\nПричины отмены водителем (Driver Cancellation Reason):")
if 'Driver Cancellation Reason' in df.columns:
    driver_cancel_reasons = df['Driver Cancellation Reason'].value_counts(dropna=False)
    print(driver_cancel_reasons)

print("\n5. СТАТИСТИКА ПО ЧИСЛОВЫМ СТОЛБЦАМ:")
numeric_columns = df.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 0:
    print("Числовые столбцы:", list(numeric_columns))
    print("\nОсновные статистики:")
    print(df[numeric_columns].describe().round(2))
else:
    print("Числовые столбцы не найдены")

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 60)