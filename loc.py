import matplotlib.pyplot as plt

# Verilen koordinat listesi
coor = [
    ["C0", 35.0, 35.0],
    ["C1", 41.0, 49.0],
    ["C2", 55.0, 22.0],
    ["C3", 20.0, 13.0],
    ["C4", 40.0, 20.0],
    ["C5", 38.0, 12.0],
    ["C6", 17.0, 10.0],
    ["C7", 25.0, 19.0],
    ["C8", 12.0, 27.0],
    ["C9", 40.0, 14.0],
    ["C10", 36.0, 10.0]
]

# Araç rotaları
vehicle_0_route = ["C0", "C1", "C6", "C3", "C8", "C0"]
vehicle_1_route = ["C0", "C4", "C9", "C5", "C10", "C2", "C7", "C0"]

# Koordinatları ayırma
names = [point[0] for point in coor]
x = [point[1] for point in coor]
y = [point[2] for point in coor]

# Noktaların renkleri
colors = ['black'] + ['red']*5 + ['green']*5

# Plotlama işlemi
plt.figure(figsize=(12, 8))

# Noktaları plot etme
for i in range(len(coor)):
    plt.scatter(x[i], y[i], color=colors[i], label=names[i] if i == 0 else "")
    plt.text(x[i], y[i], names[i], fontsize=12, ha='right')

# Araç 0 rotasını çizme
for i in range(len(vehicle_0_route) - 1):
    start_index = names.index(vehicle_0_route[i])
    end_index = names.index(vehicle_0_route[i + 1])
    plt.plot([x[start_index], x[end_index]], 'r--')

# Araç 1 rotasını çizme
for i in range(len(vehicle_1_route) - 1):
    start_index = names.index(vehicle_1_route[i])
    end_index = names.index(vehicle_1_route[i + 1])
    plt.plot([x[start_index], x[end_index]], 'g--')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Vehicle Routes')
plt.legend()
plt.grid(True)
plt.show()








