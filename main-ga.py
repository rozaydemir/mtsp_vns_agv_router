import random
import numpy as np
import random, time
import math
import sys

# Depot tanımı
depot = {
    'requestID': 0,
    'demand': 0,  # Depodan alınacak yük yok
    'startTW': 0,  # En erken başlangıç süresi
    'endTW': 730,  # En geç varış süresi
    'servTime': 4,  # Depodaki servis süresi
    'servStartTime': 0,
    'typeLoc': 'depot',  # Node tipi
    'nodeID': 0,
    'stringId': 'D0'
}
distMatrix = None

# Lokasyonlar tanımı
locations = {
    1: {'requestID': 0, 'demand': 45, 'x': 41.0, 'y': 49.0, 'startTW': 12, 'endTW': 296, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 1, 'stringId': 'C1'},
    2: {'requestID': 2, 'demand': -50, 'x': 27.0, 'y': -50.0, 'startTW': 44, 'endTW': 460, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 8, 'stringId': 'C8'},
    3: {'requestID': 4, 'demand': 35, 'x': 12.0, 'y': 35.0, 'startTW': 15, 'endTW': 295, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 5, 'stringId': 'C5'},
    4: {'requestID': 2, 'demand': 50, 'x': 13.0, 'y': 50.0, 'startTW': 17, 'endTW': 290, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 3, 'stringId': 'C3'},
    5: {'requestID': 0, 'demand': 0, 'x': 35.0, 'y': 35.0, 'startTW': 0, 'endTW': 730, 'servTime': 4,
        'servStartTime': 0, 'typeLoc': 'depot',
        'nodeID': 0, 'stringId': 'D0'},
    6: {'requestID': 0, 'demand': -45, 'x': 10.0, 'y': -45.0, 'startTW': 42, 'endTW': 469, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 6, 'stringId': 'C6'},
    7: {'requestID': 3, 'demand': 35, 'x': 40.0, 'y': 20.0, 'startTW': 19, 'endTW': 299, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 4, 'stringId': 'C4'},
    8: {'requestID': 3, 'demand': -35, 'x': 14.0, 'y': -35.0, 'startTW': 45, 'endTW': 461, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 9, 'stringId': 'C9'},
    9: {'requestID': 1, 'demand': 55, 'x': 22.0, 'y': 55.0, 'startTW': 14, 'endTW': 291, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 2, 'stringId': 'C2'},
    10: {'requestID': 1, 'demand': -55, 'x': 25.0, 'y': 19.0, 'startTW': 43, 'endTW': 465, 'servTime': 12,
         'servStartTime': 0,
         'typeLoc': 'delivery', 'nodeID': 7, 'stringId': 'C7'},
    11: {'requestID': 4, 'demand': -35, 'x': 10.0, 'y': -35.0, 'startTW': 46, 'endTW': 463, 'servTime': 14,
         'servStartTime': 0,
         'typeLoc': 'delivery', 'nodeID': 10, 'stringId': 'C10'}
}

pickupNodes = {
    1: {'requestID': 0, 'demand': 45, 'x': 41.0, 'y': 49.0, 'startTW': 12, 'endTW': 296, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 1, 'stringId': 'C1'},
    2: {'requestID': 4, 'demand': 35, 'x': 12.0, 'y': 35.0, 'startTW': 15, 'endTW': 295, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 5, 'stringId': 'C5'},
    3: {'requestID': 2, 'demand': 50, 'x': 13.0, 'y': 50.0, 'startTW': 17, 'endTW': 290, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 3, 'stringId': 'C3'},
    4: {'requestID': 3, 'demand': 35, 'x': 40.0, 'y': 20.0, 'startTW': 19, 'endTW': 299, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 4, 'stringId': 'C4'},
    5: {'requestID': 1, 'demand': 55, 'x': 22.0, 'y': 55.0, 'startTW': 14, 'endTW': 291, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'pickup', 'nodeID': 2, 'stringId': 'C2'},
}

deliveryNodes = {
    1: {'requestID': 2, 'demand': -50, 'x': 27.0, 'y': -50.0, 'startTW': 44, 'endTW': 460, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 8, 'stringId': 'C8'},
    2: {'requestID': 0, 'demand': -45, 'x': 10.0, 'y': -45.0, 'startTW': 42, 'endTW': 469, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 6, 'stringId': 'C6'},
    3: {'requestID': 3, 'demand': -35, 'x': 14.0, 'y': -35.0, 'startTW': 45, 'endTW': 461, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 9, 'stringId': 'C9'},
    4: {'requestID': 1, 'demand': -55, 'x': 25.0, 'y': 19.0, 'startTW': 43, 'endTW': 465, 'servTime': 12,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 7, 'stringId': 'C7'},
    5: {'requestID': 4, 'demand': -35, 'x': 10.0, 'y': -35.0, 'startTW': 46, 'endTW': 463, 'servTime': 14,
        'servStartTime': 0,
        'typeLoc': 'delivery', 'nodeID': 10, 'stringId': 'C10'}
}

# Araç tanımları
vehicles = {
    1: {'capacity': 60, 'start': 0, 'end': 0, 'max_trolley': 3, 'TIR': 1.2},
}

population_size = 12


# Mesafe hesaplama fonksiyonu (Öklid uzaklığı)
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def getDistance(l1, l2):
    """
    Method that computes the rounded euclidian distance between two locations
    """
    dx = l1['x'] - l2['x']
    dy = l1['y'] - l2['y']
    return round(math.sqrt(dx ** 2 + dy ** 2))


def calculateTrolley(loc, vehicle):
    trolley_count_needed = 1
    curDemand = 0
    for l in range(0, len(loc)):
        curNode = loc[l]
        curDemand += curNode['demand']
        calculateTrolley = ((curDemand + vehicle['capacity'] - 1)
                            // vehicle['capacity'])
        if calculateTrolley > trolley_count_needed:
            trolley_count_needed = calculateTrolley

        if trolley_count_needed > vehicle['max_trolley']:
            trolley_count_needed = vehicle['max_trolley']
    return trolley_count_needed


# Fitness fonksiyonu: toplam mesafe ve cezaları içerir
def fitness(solution):
    total_cost = 0
    total_penalty = 0
    total_distance = 0
    global distMatrix

    for vehicle_id, route in solution.items():
        current_time = 0
        vehicle = vehicles[vehicle_id]
        route_with_depots = route

        trolley_count_needed = calculateTrolley(route, vehicle)
        if trolley_count_needed > vehicle['max_trolley']:
            return float('inf')  # Kapasite aşılırsa fitness değeri çok kötü olur

        for i in range(1, len(route_with_depots)):
            current_node = route_with_depots[i]  # Depoyu da dahil et
            prevNode = route_with_depots[i - 1]

            # Mesafe hesapla
            dist = distMatrix[prevNode['nodeID']][current_node['nodeID']];
            total_distance += dist
            current_time += max(0, current_time + prevNode['servTime'] + dist + (trolley_count_needed * vehicle['TIR']))

            # Zaman penceresine göre ceza hesapla
            if current_node['typeLoc'] == "delivery":
                if current_time < current_node['startTW']:
                    total_penalty += (current_node['startTW'] - current_time) * 10

                if current_time > current_node['endTW']:
                    total_penalty += (current_time - current_node['endTW']) * 10

    return total_distance + total_penalty  # Toplam mesafe ve ceza döndürülür


# Popülasyon başlatma: her araca rastgele bir çözüm atıyoruz
# def initialize_population(randomGen):
#     population = []
#     for _ in range(12):  # Popülasyon boyutu
#         solution = {}
#         for v_id in vehicles.keys():  # Her araç için
#             # Rastgele düğümleri seç ve bu düğümlerin tam bilgilerini al
#             random_nodes = random.sample(list(locations.keys()), random.randint(5, len(locations)))
#             solution[v_id] = [locations[node_id] for node_id in random_nodes]  # Düğümlerin bilgilerini ekle
#         population.append(solution)
#     return population
def initialize_population(randomGen):
    population = []
    global distMatrix

    # Her araç için bir çözüm oluştur
    for _ in range(12):  # Popülasyon boyutu
        solution = {}
        for v_id in vehicles.keys():
            route = []

            # Rastgele pickup noktalarını seç
            random_pickups = random.sample(list(pickupNodes.keys()), random.randint(1, len(pickupNodes)))

            # Pickup noktalarının karşılık gelen delivery noktalarını ekle
            for pickup_id in random_pickups:
                pickup_node = pickupNodes[pickup_id]
                delivery_node = next((delivery for delivery in deliveryNodes.values() if
                                      delivery['requestID'] == pickup_node['requestID']), None)

                if delivery_node:
                    # Rota sırasıyla pickup ve delivery noktalarını ekler
                    route.append(pickup_node)
                    route.append(delivery_node)

            # Aracın başlangıç noktası (depot) ile başla ve depot ile bitir
            solution[v_id] = [depot] + route + [depot]

        population.append(solution)

    distMatrix = np.zeros((len(locations), len(locations)))  # init as nxn matrix
    for i in locations:
        for j in locations:
            xLoc = locations.get(i)
            yLoc = locations.get(j)
            distItoJ = getDistance(xLoc, yLoc)
            distMatrix[xLoc['nodeID'], yLoc['nodeID']] = distItoJ

    return population


def selection(population):
    # Popülasyondan rastgele 3 birey seç ve onları uygunluklarına göre sırala
    selected = random.sample(population, 3)
    selected.sort(key=fitness)  # Uygunluk değerine göre sırala
    return selected[0]  # En iyi (en küçük mesafeye sahip) olanı geri döndür


# Çaprazlama (Crossover): İki ebeveynden yeni bireyler oluşturuyoruz
# Çaprazlama, iki çözümün genetik materyallerini karıştırarak yeni bireyler (çocuklar) üretir.
# Bu sayede popülasyona yeni ve daha iyi çözüm adayları eklenir.
def crossover(parent1, parent2):
    child1, child2 = [], []
    # Her araç için çaprazlama yapıyoruz
    for v in range(n_vehicles):
        # Rastgele iki nokta seçip bu aralıktaki genleri (düğümleri) çaprazlıyoruz
        start, end = sorted(random.sample(range(len(parent1[v])), 2))
        # Parent1'den gelen genler çocuk1'e eklenir, geri kalan kısımlar Parent2'den gelir
        child1.append(parent1[v][start:end] + [x for x in parent2[v] if x not in parent1[v][start:end]])
        # Aynı işlemi çocuk2 için de tersine yapıyoruz
        child2.append(parent2[v][start:end] + [x for x in parent1[v] if x not in parent2[v][start:end]])
    return child1, child2  # Oluşturulan iki yeni çocuk çözümünü döndür


# Mutasyon (Mutation): Çözüme rastgele değişiklikler yapıyoruz
# Mutasyon, çözümde küçük rastgele değişiklikler yaparak popülasyondaki çeşitliliği korur.
# Aksi takdirde algoritma, yerel minimuma sıkışabilir.
def mutate(solution):
    vehicle = random.randint(0, n_vehicles - 1)  # Rastgele bir araç seç
    if len(solution[vehicle]) > 1:
        # Araç rotasındaki iki düğümün yerini değiştir
        i, j = random.sample(range(len(solution[vehicle])), 2)
        solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]
    return solution  # Mutasyona uğramış çözümü döndür


# Genetik algoritmayı çalıştır
def genetic_algorithm():
    randomGen = random.Random(1234)
    population = initialize_population(randomGen)
    best_solution = min(population, key=fitness)
    for _ in range(100):
        new_population = []  # Yeni nesil için boş bir popülasyon oluştur
        for _ in range(population_size // 2):
            # Seçilim işlemi: en iyi iki bireyi buluyoruz
            parent1 = selection(population)
            parent2 = selection(population)
            # Çaprazlama ile iki yeni çocuk oluşturuyoruz
            child1, child2 = crossover(parent1, parent2)
            # Mutasyon şansı %10, bu yüzden rastgele mutasyon yapıyoruz
            if random.random() < 0.1:
                child1 = mutate(child1)
            if random.random() < 0.1:
                child2 = mutate(child2)
            # Yeni nesil popülasyona çocukları ekliyoruz
            new_population.extend([child1, child2])
        # Mevcut popülasyon ile yeni popülasyonu birleştirip uygunluklarına göre sıralıyoruz
        population = sorted(population + new_population, key=fitness)[:population_size]
        best_solution = min(population, key=fitness)  # En iyi çözümü güncelliyoruz

    return best_solution, fitness(best_solution)


# Algoritmayı test et
best_solution, best_distance = genetic_algorithm()

print("En iyi çözüm:", best_solution)
print("En iyi mesafe:", best_distance)

# Vehicle 1 - cost = 140.0, demand = 220, trolley count = 3
#  ( D0, Demand : 0, CurrentTime: 272.0, depot, Distance: 0.0, Start: 0, End: 730, ServiceTime: 4, Penalty: 0, cumsum: 0.0 )
#  ( C1, Demand : 45, CurrentTime: 19.0, pickup, Distance: 15.0, Start: 12, End: 296, ServiceTime: 12, Penalty: 0, cumsum: 15.0 )
#  ( C2, Demand : 55, CurrentTime: 61.0, pickup, Distance: 30.0, Start: 14, End: 291, ServiceTime: 12, Penalty: 0, cumsum: 30.0 )
#  ( C4, Demand : 35, CurrentTime: 88.0, pickup, Distance: 15.0, Start: 19, End: 299, ServiceTime: 12, Penalty: 0, cumsum: 15.0 )
#  ( C9, Demand : -35, CurrentTime: 106.0, delivery, Distance: 6.0, Start: 45, End: 461, ServiceTime: 12, Penalty: 0, cumsum: 6.0 )
#  ( C5, Demand : 35, CurrentTime: 121.0, pickup, Distance: 3.0, Start: 15, End: 295, ServiceTime: 12, Penalty: 0, cumsum: 3.0 )
#  ( C10, Demand : -35, CurrentTime: 136.0, delivery, Distance: 3.0, Start: 46, End: 463, ServiceTime: 14, Penalty: 0, cumsum: 3.0 )
#  ( C7, Demand : -55, CurrentTime: 164.0, delivery, Distance: 14.0, Start: 43, End: 465, ServiceTime: 12, Penalty: 0, cumsum: 14.0 )
#  ( C3, Demand : 50, CurrentTime: 184.0, pickup, Distance: 8.0, Start: 17, End: 290, ServiceTime: 14, Penalty: 0, cumsum: 8.0 )
#  ( C6, Demand : -45, CurrentTime: 202.0, delivery, Distance: 4.0, Start: 42, End: 469, ServiceTime: 14, Penalty: 0, cumsum: 4.0 )
#  ( C8, Demand : -50, CurrentTime: 234.0, delivery, Distance: 18.0, Start: 44, End: 460, ServiceTime: 14, Penalty: 0, cumsum: 18.0 )
#  ( D0, Demand : 0, CurrentTime: 272.0, depot, Distance: 24.0, Start: 0, End: 730, ServiceTime: 4, Penalty: 0, cumsum: 24.0 )
