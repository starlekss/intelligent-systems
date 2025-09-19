import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.use('Agg')

df = pd.read_csv('ncr_ride_bookings.csv')

completed_rides = df[(df['Booking Status'] == 'Completed') & (df['Booking Value'].notnull())]

plt.figure(figsize=(12, 6))
sns.histplot(completed_rides['Booking Value'], bins=50, kde=True)
plt.title('Распределение стоимости поездок (Booking Value)')
plt.xlabel('Стоимость поездки')
plt.ylabel('Количество поездок')
plt.grid(True, alpha=0.3)
plt.savefig('booking_value_histogram.png')
plt.close()

print("Гистограмма сохранена в файл 'booking_value_histogram.png'")

print(f"Средняя стоимость поездки: {completed_rides['Booking Value'].mean():.2f}")
print(f"Медианная стоимость поездки: {completed_rides['Booking Value'].median():.2f}")
print(f"Минимальная стоимость: {completed_rides['Booking Value'].min():.2f}")
print(f"Максимальная стоимость: {completed_rides['Booking Value'].max():.2f}")

scatter_data = df[(df['Booking Status'] == 'Completed') &
                 (df['Booking Value'].notnull()) &
                 (df['Ride Distance'].notnull())]

plt.figure(figsize=(12, 8))
sns.scatterplot(data=scatter_data, x='Ride Distance', y='Booking Value', alpha=0.6)
plt.title('Зависимость стоимости поездки от расстояния')
plt.xlabel('Расстояние поездки (км)')
plt.ylabel('Стоимость поездки')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png')
plt.close()

print("Диаграмма рассеяния сохранена в файл 'scatter_plot.png'")

correlation = scatter_data['Ride Distance'].corr(scatter_data['Booking Value'])
print(f"Коэффициент корреляции между расстоянием и стоимостью: {correlation:.3f}")