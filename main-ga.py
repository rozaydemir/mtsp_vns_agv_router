import random
import numpy as np
import random, time
import math
import sys

# Depot tanımı ve # Lokasyonlar tanımı
depot = {}
distMatrix = None
locations = {}
pickupNodes = {}
deliveryNodes = {}
requests = list()
vehicles = {}
alpha = 10
beta = 10
TIR = 10

class Vehicles:
    def __init__(self, id, Mmax, capacityOfTrolley):
        self.vehicleId = id
        self.capacity = capacityOfTrolley
        self.max_trolley = Mmax

class Request:
    def __init__(self, pickUpLoc, deliveryLoc, ID):
        self.pickUpLoc = pickUpLoc
        self.deliveryLoc = deliveryLoc
        self.ID = ID

    def __str__(self):
        return "requestID: {}; pickUpLoc: {}; deliveryLoc: {}".format(self.ID, self.pickUpLoc, self.deliveryLoc)

class Location:
    """
    Class that represents either (i) a location where a request should be picked up
    or delivered or (ii) the depot
    Attributes
    ----------
    requestID : int
        id of request.
    xLoc : int
        x-coordinate.
    yLoc : int
        y-coordinate.
    demand : int
        demand quantity, positive if pick-up, negative if delivery
    startTW : int
        start time of time window.
    endTW : int
        end time of time window.
    servTime : int
        service time.
    typeLoc : int
        1 if pick-up, -1 if delivery, 0 if depot
    nodeID : int
        id of the node, used for the distance matrix
    """

    def __init__(self, requestID, xLoc, yLoc, demand, startTW, endTW, servTime, servStartTime, typeLoc, nodeID,
                 stringId):
        self.requestID = requestID
        self.x = xLoc
        self.y = yLoc
        self.demand = demand
        self.startTW = startTW
        self.endTW = endTW
        self.servTime = servTime
        self.servStartTime = servStartTime
        self.typeLoc = typeLoc
        self.nodeID = nodeID
        self.stringId = stringId

    def __str__(self):
        return (
            f"requestID: {self.requestID}; demand: {self.demand}; startTW: {self.startTW}; endTW: {self.endTW}; servTime:{self.servTime}; servStartTime: {self.servStartTime}; typeLoc: {self.typeLoc}, nodeID: {self.nodeID}, stringId: {self.stringId}")


def read_instance(fileName):
    global alpha
    global beta
    global requests
    global vehicles
    global TIR
    global depot
    global locations
    global pickupNodes
    global deliveryNodes
    """
    Method that reads an instance from a file and returns the instancesf
    """
    f = open(fileName)
    unmatchedPickups = dict()
    unmatchedDeliveries = dict()
    nodeCount = 0
    requestCount = 0  # start with 1
    capacityOfTrolley = 60
    Mmax = 3
    vehicleCount = 1

    for infoLine in f.readlines()[-6:]:
        if infoLine.startswith("VehicleCount"):
            vehicleCount = int(infoLine[-3:-1].strip())
        if infoLine.startswith("VehiclePerCapacity"):
            capacityOfTrolley = int(infoLine[-6:-1].strip())
        if infoLine.startswith("VehicleMaxTrolleyCount"):
            Mmax = int(infoLine[-2:-1].strip())
        if infoLine.startswith("TrolleyImpactRate"):
            TIR = float(infoLine[-5:-1].strip())
        if infoLine.startswith("EarlinessPenalty"):
            alpha = float(infoLine[-3:-1].strip())
        if infoLine.startswith("TardinessPenalty"):
            beta = float(infoLine[-3:-1].strip())

    f = open(fileName)
    for line in f.readlines()[1:-7]:
        asList = []
        n = 13  # satırların sondan 13 karakteri boş o yüzden
        for index in range(0, len(line), n):
            asList.append(line[index: index + n].strip())

        lID = asList[0]  # location tipi  D : depot, S: station, C : pickup / delivery point,
        x = int(asList[2][:-2])  # need to remove ".0" from the string
        y = int(asList[3][:-2])
        lType = asList[1]
        demand = int(asList[4][:-2])
        startTW = int(asList[5][:-2])
        endTW = int(asList[6][:-2])
        servTime = int(asList[7][:-2])
        partnerID = asList[8]
        if lID.startswith("D"):  # depot ise
            # it is the depot
            depot = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "depot", nodeCount,
                             lID)  # depot requestID=0
            locations[nodeCount] = depot
            nodeCount += 1

        elif lID.startswith("C"):  # pickup/delivery point ise
            # it is a location

            if lType == "cp":  # cp ise pickup, #cd ise delivery point
                if partnerID in unmatchedDeliveries:
                    deliv = unmatchedDeliveries.pop(
                        partnerID)  # pop listeden siler, sildiği değeri ise bir değişkene atar, burada deliv değişkenine atadı
                    pickup = Location(deliv.requestID, x, y, demand, startTW, endTW, servTime, 0,
                                      "pickup", nodeCount, lID)
                    nodeCount += 1
                    pickupNodes[nodeCount] = pickup
                    locations[nodeCount] = pickup
                else:
                    pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "pickup",
                                      nodeCount, lID)
                    nodeCount += 1
                    requestCount += 1
                    pickupNodes[nodeCount] = pickup
                    locations[nodeCount] = pickup
                    unmatchedPickups[lID] = pickup

            elif lType == "cd":  # cp ise pickup, #cd ise delivery point
                if partnerID in unmatchedPickups:
                    pickup = unmatchedPickups.pop(partnerID)
                    deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, 0,
                                     "delivery", nodeCount, lID)
                    nodeCount += 1
                    req = Request(pickup, deliv, pickup.requestID)
                    requests.append(req)
                    unmatchedDeliveries[lID] = deliv
                    deliveryNodes[nodeCount] = deliv
                    locations[nodeCount] = deliv
                else:
                    deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0,
                                     "delivery", nodeCount, lID)
                    nodeCount += 1
                    requestCount += 1
                    unmatchedDeliveries[lID] = deliv
                    deliveryNodes[nodeCount] = deliv
                    locations[nodeCount] = deliv


    for i in range(vehicleCount):
        vehicles[i + 1] = Vehicles(i, Mmax, capacityOfTrolley)



# Mesafe hesaplama fonksiyonu (Öklid uzaklığı)
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def getDistance(l1, l2):
    """
    Method that computes the rounded euclidian distance between two locations
    """
    dx = l1.x - l2.x
    dy = l1.y - l2.y
    return round(math.sqrt(dx ** 2 + dy ** 2))

def calculateTrolley(loc):
    global vehicles
    global locations
    global pickupNodes
    global deliveryNodes
    trolley_count_needed = 1
    curDemand = 0
    for l in range(0, len(loc)):
        curNode = loc[l]
        curDemand += curNode.demand
        calculateTrolley = ((curDemand + vehicles.get(1).capacity - 1)
                            // vehicles.get(1).capacity)
        if calculateTrolley > trolley_count_needed:
            trolley_count_needed = calculateTrolley

        if trolley_count_needed > vehicles.get(1).max_trolley:
            trolley_count_needed = vehicles.get(1).max_trolley
    return trolley_count_needed

# Fitness fonksiyonu: toplam mesafe ve cezaları içerir
def fitness(solution):
    global alpha
    global beta
    global vehicles
    global distMatrix
    global TIR
    global locations
    global pickupNodes
    global deliveryNodes
    total_penalty = 0
    total_distance = 0


    for vehicle_id, route in solution.items():
        current_time = 0
        vehicle = vehicles.get(vehicle_id)
        route_with_depots = route

        trolley_count_needed = calculateTrolley(route)
        if trolley_count_needed > vehicle.max_trolley:
            return float('inf')  # Kapasite aşılırsa fitness değeri çok kötü olur

        for i in range(1, len(route_with_depots)):
            current_node = route_with_depots[i]  # Depoyu da dahil et
            prevNode = route_with_depots[i - 1]

            # Mesafe hesapla
            dist = distMatrix[prevNode.nodeID][current_node.nodeID]
            total_distance += dist
            current_time = max(0, current_time + prevNode.servTime + dist + (trolley_count_needed * TIR))

            # Zaman penceresine göre ceza hesapla
            if current_node.typeLoc == "delivery":
                if current_time < current_node.startTW:
                    total_penalty += (current_node.startTW - current_time) * alpha

                if current_time > current_node.endTW:
                    total_penalty += (current_time - current_node.endTW) * beta

    return total_distance + total_penalty  # Toplam mesafe ve ceza döndürülür

def computeTimeWindow(loc) -> int:
    global distMatrix
    global alpha
    global beta
    global vehicles
    global TIR
    global locations
    global pickupNodes
    global deliveryNodes
    curTime = 0
    totalTimeWindowPenaly = 0
    trolley_count_needed = calculateTrolley(loc)

    for i in range(1, len(loc)):
        prevNode = loc[i - 1]
        curNode = loc[i]
        dist = distMatrix[prevNode.nodeID][curNode.nodeID]
        curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * TIR) )

        ETPenalty = 0
        if curNode.typeLoc == "delivery":
            if curTime < curNode.startTW:
                ETPenalty += (curNode.startTW - curTime) * alpha

            if curTime > curNode.endTW:
                ETPenalty += (curTime - curNode.endTW) * beta
        totalTimeWindowPenaly += ETPenalty
    return totalTimeWindowPenaly

# Mesafe hesaplama fonksiyonu (örneğin, Öklid mesafesi)
def compute_cost_add_request(route, pickup_idx, delivery_idx, pickup_node, delivery_node):
    # Pickup ve delivery noktalarının mevcut rotaya eklenmesinin maliyetini hesapla
    # route = mevcut rota, pickup_idx = pickup'ın ekleneceği yer, delivery_idx = delivery'nin ekleneceği yer
    global distMatrix
    global locations
    global pickupNodes
    global deliveryNodes

    # Pickup eklendikten sonra rotada değişen mesafeyi hesapla
    cost_before_pickup = distMatrix[route[pickup_idx - 1].nodeID][pickup_node.nodeID]
    cost_after_pickup = distMatrix[pickup_node.nodeID][route[pickup_idx].nodeID]

    # Delivery eklendikten sonra rotada değişen mesafeyi hesapla
    cost_before_delivery = distMatrix[route[delivery_idx - 1].nodeID][delivery_node.nodeID]
    cost_after_delivery = distMatrix[delivery_node.nodeID][route[delivery_idx].nodeID]

    cost3 = computeTimeWindow(route)

    total_cost = (cost_before_pickup + cost_after_pickup + cost_before_delivery + cost_after_delivery + cost3)

    return total_cost

def greedy_insert(solution, vehicle_id, pickup_node, delivery_node):
    best_cost = sys.maxsize
    best_route = None
    global distMatrix
    global vehicles
    global depot
    global locations
    global pickupNodes
    global deliveryNodes

    # Eğer sadece depo varsa, ilk ekleme özel bir duruma tabi
    if len(solution.get(vehicle_id)) == 2:  # Yalnızca depo varsa
        # Depo dışındaki ilk eklemeyi yap (depo ile depo arasına pickup ve delivery eklenir)
        temp_route = [depot] + [pickup_node] + [delivery_node] + [depot]
        # 3. Şart: Maliyeti hesapla ve en iyi maliyeti bul
        insertion_cost = compute_cost_add_request(temp_route, 1, 2, pickup_node, delivery_node)

        # En düşük maliyetli yerleşimi bul
        if insertion_cost < best_cost:
            best_cost = insertion_cost
            best_route = temp_route

        return best_route, best_cost
    else:
        # Mevcut rotada pickup ve delivery eklemek için tüm olası pozisyonları dene
        for i in range(1, len(solution[vehicle_id]) - 1):  # Pickup pozisyonu için depo sonrası ve depo öncesi hariç
            for j in range(i + 1, len(solution[vehicle_id])):  # Delivery pozisyonu (pickup'tan sonra olmalı)
                # Geçici olarak pickup ve delivery eklenmiş rotayı oluştur (depoyu sabit tutarak)
                temp_route = solution[vehicle_id][:i] + [pickup_node] + solution[vehicle_id][i:j - 1] + [delivery_node] + solution[vehicle_id][j - 1:]

                # 1. Şart: Trolley kapasitesi kontrolü
                trolley_needed = calculateTrolley(temp_route)
                if trolley_needed > vehicles.get(vehicle_id).max_trolley:
                    continue  # Kapasite aşımı varsa geçersiz, diğer pozisyonlara bak

                # 2. Şart: Pickup önce, delivery sonra olmalı
                if pickup_node.nodeID > delivery_node.nodeID:
                    continue  # Eğer pickup, delivery'den büyükse bu araca ekleme yapılmaz

                # 3. Şart: Maliyeti hesapla ve en iyi maliyeti bul
                insertion_cost = compute_cost_add_request(temp_route, i, j, pickup_node, delivery_node)

                # En düşük maliyetli yerleşimi bul
                if insertion_cost < best_cost:
                    best_cost = insertion_cost
                    best_route = temp_route

    return best_route, best_cost

# Popülasyon başlatma: her araca rastgele bir çözüm atıyoruz
def initialize_population():
    population = []
    global distMatrix
    global vehicles
    global depot
    global locations
    global pickupNodes
    global deliveryNodes

    # Her araç için bir çözüm oluştur
    for i in range(population_size):  # Popülasyon boyutu
        solution = {}

        # Pickup ve delivery noktalarının kopyalarını oluştur
        available_pickups = pickupNodes.copy()
        available_deliveries = deliveryNodes.copy()

        for v_id in vehicles.keys():
            solution[v_id] = [depot] + [depot]  # Başlangıç olarak her aracın rotasını depodan başlatıyoruz

        # Pickup noktalarını greedy ekleme işlemi
        while available_pickups:
            pickup_id = random.choice(list(available_pickups.keys()))  # Rastgele bir pickup noktası seç
            pickup_node = available_pickups.get(pickup_id)
            delivery_node = next((delivery for delivery in available_deliveries.values() if
                                  delivery.requestID == pickup_node.requestID), None)

            if delivery_node is None:
                continue  # Eğer uygun bir delivery noktası yoksa sıradaki pickup noktasına geç

            assigned = False  # Pickup ve delivery noktalarının başarılı bir şekilde atanıp atanmadığını takip ederiz

            while not assigned:
                # Rastgele Greedy ekleme yöntemi: her araca en uygun pozisyona ekleme
                v_id = random.choice(list(vehicles.keys()))
                best_route, best_cost = greedy_insert(solution, v_id, pickup_node, delivery_node)
                if best_route is not None:
                    # Eğer geçerli bir ekleme varsa rota güncellenir
                    solution[v_id] = best_route
                    del available_pickups[pickup_id]  # Eklenen pickup noktasını kaldır
                    delivery_node_key = next(key for key, val in available_deliveries.items() if val == delivery_node)
                    del available_deliveries[delivery_node_key]  # Eklenen delivery noktasını da kaldır
                    assigned = True

        population.append(solution)

    return population

def selection(population):
    # Popülasyondan rastgele 3 birey seç ve onları uygunluklarına göre sırala
    selected = random.sample(population, selection_size)
    selected.sort(key=fitness)  # Uygunluk değerine göre sırala
    return selected[0]  # En iyi (en küçük mesafeye sahip) olanı geri döndür

# Çaprazlama (Crossover): İki ebeveynden yeni bireyler oluşturuyoruz
# Çaprazlama, iki çözümün genetik materyallerini karıştırarak yeni bireyler (çocuklar) üretir.
# Bu sayede popülasyona yeni ve daha iyi çözüm adayları eklenir.
def crossover(parent1, parent2):
    global vehicles
    global depot
    global locations
    global pickupNodes
    global deliveryNodes
    child1, child2 = {1: [], 2: []}, {1: [], 2: []}

    valid_child = False
    while not valid_child:
        for v in range(1, len(vehicles) + 1):
            non_depot_parent1, non_depot_parent2, selected_pickups_parent1, selected_pickups_parent2, selected_deliveries_1 = [], [], [], [], []
            selected_deliveries_2, child1_route, child2_route = [], [], []
            # Depo dışındaki noktaları seçmek için filtreleme
            non_depot_parent1 = [loc for loc in parent1[v] if loc.typeLoc != 'depot']
            non_depot_parent2 = [loc for loc in parent2[v] if loc.typeLoc != 'depot']

            # Rastgele iki noktayı seçmek için pickup ve delivery noktalarını ayırıyoruz
            selected_pickups_parent1 = [loc for loc in non_depot_parent1 if loc.typeLoc == 'pickup']
            selected_pickups_parent2 = [loc for loc in non_depot_parent2 if loc.typeLoc == 'pickup']

            # İlk önce boş olma durumunu kontrol ediyoruz
            if len(selected_pickups_parent1) == 0 or len(selected_pickups_parent2) == 0:
                # Eğer bir pickup listesi boşsa, num_pickups değerini 0 yap veya işlemi atla
                num_pickups = 0  # ya da burada farklı bir strateji uygulayabilirsiniz
            else:
                # Eğer her iki tarafta da pickup varsa, minimum uzunluğu baz alarak num_pickups belirliyoruz
                num_pickups = random.randint(1, min(len(selected_pickups_parent1), len(selected_pickups_parent2)))

            # num_pickups = random.randint(1, min(len(selected_pickups_parent1), len(selected_pickups_parent2)))

            selected_pickups_1 = random.sample(selected_pickups_parent1, num_pickups)
            selected_deliveries_1 = [next(loc for loc in non_depot_parent1 if
                                          loc.requestID == pickup.requestID and loc.typeLoc == 'delivery') for
                                     pickup in selected_pickups_1]

            selected_pickups_2 = random.sample(selected_pickups_parent2, num_pickups)
            selected_deliveries_2 = [next(loc for loc in non_depot_parent2 if
                                          loc.requestID == pickup.requestID and loc.typeLoc == 'delivery') for
                                     pickup in selected_pickups_2]

            child1_route = selected_pickups_1 + selected_deliveries_1 + [x for x in non_depot_parent2 if x not in (
                    selected_pickups_1 + selected_deliveries_1)]
            child2_route = selected_pickups_2 + selected_deliveries_2 + [x for x in non_depot_parent1 if x not in (
                    selected_pickups_2 + selected_deliveries_2)]

            # Depoları başa ve sona ekliyoruz
            child1[v] = [depot] + child1_route + [depot]
            child2[v] = [depot] + child2_route + [depot]

        # Trolley kapasitesini kontrol et
        for c in range(1, len(child1) + 1):
            trolley_needed_child1 = calculateTrolley(child1[c])
            if trolley_needed_child1 <= vehicles.get(c).max_trolley:
                valid_child = True
            else:
                print("Trolley kapasitesi aşıldı, yeni çocuk yaratılıyor." + child1[c])

        for c2 in range(1, len(child2) + 1):
            trolley_needed_child2 = calculateTrolley(child2[c2])
            if trolley_needed_child2 <= vehicles.get(c2).max_trolley:
                valid_child = True
            else:
                print("Trolley kapasitesi aşıldı, yeni çocuk yaratılıyor." + child2[c2])

    return child1, child2

# Mutasyon (Mutation): Çözüme rastgele değişiklikler yapıyoruz
# Mutasyon, çözümde küçük rastgele değişiklikler yaparak popülasyondaki çeşitliliği korur.
# Aksi takdirde algoritma, yerel minimuma sıkışabilir.
def mutate(solution):
    global vehicles
    global locations
    global pickupNodes
    global deliveryNodes
    vehicle = random.randint(1, len(vehicles))  # Rastgele bir araç seç

    if len(solution[vehicle]) > 2:  # En az 2 düğüm varsa mutasyon yapılabilir
        valid_range = range(1, len(solution[vehicle]) - 1)  # Depolar hariç

        # Rastgele iki düğümün yerini değiştir
        i, j = random.sample(valid_range, 2)

        # Swap öncesi validasyon kontrolü
        if is_valid_swap(solution[vehicle], i, j):
            # Swap işlemini uygula
            solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]

            # Swap sonrası tüm rotayı kontrol et, geçerli mi?
            if is_route_valid(solution[vehicle]) == False:
                # Swap geçerli değilse çözümü geri çevir
                solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]

    return solution

# Swap işleminin geçerli olup olmadığını kontrol eden fonksiyon
def is_valid_swap(route, i, j):
    node_i = route[i]
    node_j = route[j]

    # Aynı talebin pickup'ı delivery'den önce olmalı
    if node_i.requestID == node_j.requestID:
        if (node_i.typeLoc == 'pickup' and node_j.typeLoc == 'delivery'):
            return True
        elif (node_i.typeLoc == 'delivery' and node_j.typeLoc == 'pickup'):
            return False

    # Diğer durumlarda swap işlemine izin verebiliriz
    return True

# Tüm rotanın geçerli olup olmadığını kontrol eden fonksiyon
def is_route_valid(route):
    # Her talep için önce pickup, sonra delivery olup olmadığını kontrol etmeliyiz
    pickup_seen = {}

    for node in route:
        if node.typeLoc == 'pickup':
            # Pickup noktasını görüyoruz, bunu kaydediyoruz
            pickup_seen[node.requestID] = True
        elif node.typeLoc == 'delivery':
            # Delivery noktasına gidiyoruz, bu noktadan önce pickup görülmüş olmalı
            if pickup_seen.get(node.requestID, False) is not True:
                return False  # Delivery, pickup'tan önce geldi, geçersiz

    # Tüm kontrol geçtiyse rota geçerlidir
    return True

# Araç rotası sonuçlarını belirttiğin formatta yazdırmak için fonksiyon
def print_solution(solution):
    global alpha
    global beta
    global vehicles
    global TIR
    global locations
    global pickupNodes
    global deliveryNodes
    totalResultDistance, totalResultPenalty, totalCumSum = 0, 0, 0
    for vehicle_id, route in solution.items():
        trolley_count = calculateTrolley(route)  # Troley sayısını hesapla
        print(f"# Vehicle {vehicle_id} (trolley count = {trolley_count}):")
        total_distance = 0
        total_penalty = 0
        cumsum = 0
        current_time = 0

        for i in range(len(route)):
            node = route[i]
            penalty = 0
            distance = 0

            if i > 0:
                previous_node = route[i - 1]
                distance = distMatrix[previous_node.nodeID][node.nodeID]
                total_distance += distance
                current_time = max(0, current_time + previous_node.servTime + distance + (
                        trolley_count * TIR))

                if node.typeLoc == 'delivery':
                    if current_time < node.startTW:
                        penalty = (node.startTW - current_time) * alpha
                    elif current_time > node.endTW:
                        penalty = (current_time - node.endTW) * beta

                total_penalty += penalty
                cumsum += distance + penalty

            print(f"#  ({node.stringId}, Demand: {node.demand}, "
                  f"CurrentTime: {current_time}, {node.typeLoc}, Distance: {distance}, "
                  f"Start: {node.startTW}, End: {node.endTW}, "
                  f"ServiceTime: {node.servTime}, Penalty: {penalty}, cumsum: {distance + penalty})")

        totalResultDistance += total_distance
        totalResultPenalty += total_penalty
        totalCumSum += cumsum
        print(f"# Total Distance: {total_distance}, Total Penalty: {total_penalty}, Cumulative Sum: {cumsum}\n")

    print(
        f"# All Total Distance: {totalResultDistance}, All Total Penalty: {totalResultPenalty}, All Cumulative Sum: {totalCumSum}\n")
    return totalCumSum

def compute_distMatrix():
    # Mesafe matrisini başlat
    global distMatrix
    global locations
    global locations
    global pickupNodes
    global deliveryNodes

    distMatrix = np.zeros((len(locations), len(locations)))  # init as nxn matrix
    for i in locations:
        for j in locations:
            xLoc = locations.get(i)
            yLoc = locations.get(j)
            distItoJ = getDistance(xLoc, yLoc)
            distMatrix[xLoc.nodeID, yLoc.nodeID] = distItoJ

# Genetik algoritmayı çalıştır
def genetic_algorithm():
    global locations
    global pickupNodes
    global deliveryNodes
    starttime = time.time()  # get the start time
    compute_distMatrix()
    population = initialize_population()
    best_global_solution = min(population, key=fitness)
    best_global_solution_cost = fitness(best_global_solution)
    print(f"# Initialize Global Result: {best_global_solution_cost}\n")
    for i in range(iteration_size):
        _ = i
        new_population = []  # Yeni nesil için boş bir popülasyon oluştur
        for _ in range(population_size // 2):
            # Seçilim işlemi: en iyi iki bireyi buluyoruz
            parent1 = selection(population)
            parent2 = selection(population)
            # Çaprazlama ile iki yeni çocuk oluşturuyoruz
            child1, child2 = crossover(parent1, parent2)
            # Mutasyon şansı %10, bu yüzden rastgele mutasyon yapıyoruz
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            # Yeni nesil popülasyona çocukları ekliyoruz
            new_population.extend([child1, child2])
        # Mevcut popülasyon ile yeni popülasyonu birleştirip uygunluklarına göre sıralıyoruz
        population = sorted(population + new_population, key=fitness)[:population_size]
        best_solution = min(population, key=fitness)  # En iyi çözümü güncelliyoruz
        best_solution_cost = fitness(best_solution)
        if best_solution_cost < best_global_solution_cost:
            print(
                f'New best global solution found: cost :{best_solution_cost}, iteration : {i}')
            best_global_solution_cost = best_solution_cost
            best_global_solution = best_solution

    totalCumSum = print_solution(best_global_solution)

    endtime = time.time()  # get the end time
    cpuTime = round(endtime - starttime, 3)

    print("Terminated. Final cost: " + str(totalCumSum) + ", cpuTime: " + str(cpuTime) + " seconds")

    return best_global_solution, fitness(best_global_solution)

population_size = 50
selection_size = 10
iteration_size = 6000
mutation_rate = 0.1

data = "Instances/lrc11.txt"
problem = read_instance(data)

# Algoritmayı test et
best_solution, best_distance = genetic_algorithm()


