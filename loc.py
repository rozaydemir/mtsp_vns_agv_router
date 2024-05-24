import matplotlib.pyplot as plt

# Verilen koordinat listesi
coor = [
    [0.0, 1.0],
    [55.0, -63.0],
    [98.0, 21.0],
    [66.0, -19.0],
    [53.0, 25.0],
    [41.0, -33.0],
    [97.0, 27.0],
    [52.0, 30.0],
    [74.0, 77.0],
    [11.0, -13.0],
    [30.0, 67.0],
    [12.0, 77.0],
    [65.0, -37.0],
    [14.0, 11.0],
    [27.0, 99.0],
]

# Koordinatları x ve y listelerine ayırma
x = [point[0] for point in coor]
y = [point[1] for point in coor]

# Plotlama işlemi
plt.figure(figsize=(10, 6))

# İlk nokta siyah
plt.scatter(x[0], y[0], color='black', label='Point 1')

# Sonraki 4 nokta kırmızı
plt.scatter(x[1:7], y[1:7], color='red', label='Next 4 Points')

# Son 4 nokta yeşil
plt.scatter(x[7:], y[7:], color='green', label='Last 4 Points')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Coordinate Plot')
plt.legend()
plt.grid(True)
plt.show()