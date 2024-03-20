from alns import ALNS, State
import copy
from alns.stop import MaxRuntime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

# Problem verilerinin tanımlanması

vehicle_capacities = {1: 30, 2: 50}  # Her iki aracın da kapasitesi
demands = [0, 20, 20, 20, 20, 0]  # Nokta talepleri, ilk nokta deposu temsil eder ve talebi 0'dır
locations = [(0, 0), (10, 10), (-10, 10), (-10, -10), (10, -10), (0, 20)]  # Noktaların koordinatları
distance_matrix = np.zeros((len(locations), len(locations)))  # Mesafe matrisi
SEED = 1234
pickup_deliveries = [(10, 10), (-10, 10), (-10, -10), (10, -10)]

vehicles = [1, 2]
all_location = [0, 1, 2, 3, 4, 5]
pickup_nodes = [1, 2]
delivery_nodes = [1, 2]
depot_nodes = {1: 0}  # depot node
destination_node = {1: 5}  # destination node

alns = ALNS(rnd.RandomState(SEED))

alpha = 1
beta = 10

# Mesafe matrisinin hesaplanması
for i in range(len(locations)):
    for j in range(len(locations)):
        distance_matrix[i, j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))


class CvrpState:
    def __init__(self, routes):
        self.routes = routes

    def objective(self) -> float:
        """
           Hesaplar toplam maliyeti, verilen rotalar, seyahat maliyetleri, erken ve geç teslimat cezaları için.
           routes: Araç rotaları ve ziyaret edilen düğümler listesi.
           c_ijk: Her bir (i, j) düğüm çifti ve k araç için seyahat maliyetleri.
           E: Her bir i düğümü için erken varış miktarları.
           TA: Her bir i düğümü için geç varış miktarları.
           alpha: Erken varış başına ceza maliyeti.
           beta: Geç varış başına ceza maliyeti.
           """
        # travel_cost = sum(
        #     c_ijk[k][i][j] * x_ijk[k][i][j] for k, route in enumerate(routes) for i, j in zip(route, route[1:]))
        # earliness_cost = sum(alpha * E[i] for i in E)
        # tardiness_cost = sum(beta * TA[i] for i in TA)

        total_cost = 1 + 1 + 1
        return total_cost


def initial_solution(locations, vehicle_capacity, demands):
    """
    Initialize routes for vehicles, ensuring they pick up loads up to their capacity,
    then deliver before returning to the destination.
    locations: List of all locations, including start and destination.
    vehicle_capacity: Maximum load each vehicle can carry.
    demands: List of demands at each pickup location (0 for start, destination, and delivery locations).
    """
    routes = []
    pickup_indices = list(range(1, len(locations) // 2))  # Assuming first half after start are pickups
    delivery_indices = list(
        range(len(locations) // 2, len(locations) - 1))  # Assuming second half before destination are deliveries

    arcs = {k: [(p, d) for p in all_location if p != 5 for d in all_location if p != d] for k in
            vehicles}  # A subset of V_k x V_k, representing all feasible paths for vehicle k between its nodes.

    print(arcs)

    for _ in vehicles:  # Assuming vehicles
        route = [(0, 0)]  # Start from the first location
        current_load = 0
        # Pickup phase
        for pickup in pickup_indices:
            if current_load + demands[pickup] <= vehicle_capacity[_]:
                route.append(pickup)
                current_load += demands[pickup]
        # Delivery phase
        for delivery in delivery_indices:
            route.append(delivery)  # Assume we deliver in order of pickup for simplicity
            current_load = 0  # Assume all load is delivered at each stop for simplicity

        route.append(len(locations) - 1)  # Return to the destination
        routes.append(route)

    return CvrpState(routes)


routes = initial_solution(locations, vehicle_capacities, demands)

print(routes.routes)

# Sonuçların görselleştirilmesi
# Bu kısım, bulunan rota üzerindeki noktaları ve araç geçişlerini gösteren bir grafik çizimi içerir.
# Not: Görselleştirme, ALNS algoritmasının sonuçlarına bağlı olarak dinamik olacaktır.

# Bu örnek, ALNS algoritmasının uygulanması için bir başlangıç noktası sağlar.
# Tam bir çözüm, problem spesifik operatörlerin detaylı tanımlanmasını ve algoritmanın etkili bir şekilde uygulanmasını gerektirir.
