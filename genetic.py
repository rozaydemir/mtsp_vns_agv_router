import random
import numpy as np
import random, time
import math
import sys

class Vehicles:
    def __init__(self, id, Mmax, capacityOfTrolley):
        self.vehicleId = id
        self.max_trolley = Mmax
        self.capacity = capacityOfTrolley

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

class Genetic:
    def __init__(self, fileName, vehicleCount, VehiclePerCapacity, VehicleMaxTrolleyCount, TrolleyImpactRate,
                 EarlinessPenalty, TardinessPenalty, iterationCount, populationSize, selectionSize):
        self.fileName = fileName
        self.nIterations = iterationCount
        self.vehicleCount = vehicleCount
        self.VehiclePerCapacity = VehiclePerCapacity
        self.VehicleMaxTrolleyCount = VehicleMaxTrolleyCount
        self.TIR = TrolleyImpactRate
        self.alpha = EarlinessPenalty
        self.beta = TardinessPenalty

        self.population_size = populationSize
        self.selection_size = selectionSize
        self.mutation_rate = 0.10

        self.distMatrix = []
        self.vehicles = {}
        self.depot = {}
        self.locations = {}
        self.pickupNodes = {}
        self.deliveryNodes = {}

    def execute(self):
        self.read_instance(self.fileName,
                      self.vehicleCount,
                      self.VehicleMaxTrolleyCount,
                      self.VehiclePerCapacity)

        starttime = time.time()  # get the start time
        # Algoritmayı test et
        best_solution, best_distance = self.genetic_algorithm(self.nIterations)

        endtime = time.time()  # get the end time
        cpuTime = round(endtime - starttime, 3)

        totalCumSum, resultArray = self.print_solution(best_solution)

        print("GA   - Final cost: " + str(totalCumSum) + ", cpuTime: " + str(cpuTime) + " seconds")
        return round(best_distance), cpuTime, resultArray

    def read_instance(self, fileName, vehicleCount, VehicleMaxTrolleyCount, VehiclePerCapacity):
        """
        Method that reads an instance from a file and returns the instancesf
        """
        # f = open(fileName)
        unmatchedPickups = dict()
        unmatchedDeliveries = dict()
        nodeCount = 0
        requestCount = 0  # start with 1

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
                dpt = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "depot", nodeCount,
                                 lID)  # depot requestID=0
                self.locations[nodeCount] = dpt
                self.depot = dpt
                nodeCount += 1
                requestCount += 1

            elif lID.startswith("C"):  # pickup/delivery point ise
                # it is a location

                if lType == "cp":  # cp ise pickup, #cd ise delivery point
                    if partnerID in unmatchedDeliveries:
                        deliv = unmatchedDeliveries.pop(
                            partnerID)  # pop listeden siler, sildiği değeri ise bir değişkene atar, burada deliv değişkenine atadı
                        pickup = Location(deliv.requestID, x, y, demand, startTW, endTW, servTime, 0,
                                          "pickup", nodeCount, lID)
                        nodeCount += 1
                        self.pickupNodes[nodeCount] = pickup
                        self.locations[nodeCount] = pickup
                    else:
                        pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "pickup",
                                          nodeCount, lID)
                        nodeCount += 1
                        requestCount += 1
                        self.pickupNodes[nodeCount] = pickup
                        self.locations[nodeCount] = pickup
                        unmatchedPickups[lID] = pickup

                elif lType == "cd":  # cp ise pickup, #cd ise delivery point
                    if partnerID in unmatchedPickups:
                        pickup = unmatchedPickups.pop(partnerID)
                        deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, 0,
                                         "delivery", nodeCount, lID)
                        nodeCount += 1
                        unmatchedDeliveries[lID] = deliv
                        self.deliveryNodes[nodeCount] = deliv
                        self.locations[nodeCount] = deliv
                    else:
                        deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0,
                                         "delivery", nodeCount, lID)
                        nodeCount += 1
                        requestCount += 1
                        unmatchedDeliveries[lID] = deliv
                        self.deliveryNodes[nodeCount] = deliv
                        self.locations[nodeCount] = deliv

        for i in range(vehicleCount):
            self.vehicles[i + 1] = Vehicles(i + 1, VehicleMaxTrolleyCount, VehiclePerCapacity)

    def sort_random_request(self):
        items = list(self.pickupNodes.items())
        random.shuffle(items)
        shuffled_data = {i + 1: value for i, (_, value) in enumerate(items)}
        self.pickupNodes = shuffled_data

    # Genetik algoritmayı çalıştır
    def genetic_algorithm(self, nIterations):
        self.compute_distMatrix()
        self.sort_random_request()
        population = self.initialize_population()
        best_global_solution = min(population, key=self.fitness)
        best_global_solution_cost = self.fitness(best_global_solution)
        print(f"# Initialize Global Result: {best_global_solution_cost}\n")
        for i in range(nIterations):
            _ = i
            new_population = []  # Yeni nesil için boş bir popülasyon oluştur
            for _ in range(self.population_size // 2):
                # Seçilim işlemi: en iyi iki bireyi buluyoruz
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                # Çaprazlama ile iki yeni çocuk oluşturuyoruz
                child1, child2 = self.crossover(parent1, parent2)
                # Mutasyon şansı %10, bu yüzden rastgele mutasyon yapıyoruz
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)
                # Yeni nesil popülasyona çocukları ekliyoruz
                new_population.extend([child1, child2])
            # Mevcut popülasyon ile yeni popülasyonu birleştirip uygunluklarına göre sıralıyoruz
            population = sorted(population + new_population, key=self.fitness)[:self.population_size]
            best_solution = min(population, key=self.fitness)  # En iyi çözümü güncelliyoruz
            best_solution_cost = self.fitness(best_solution)
            if best_solution_cost < best_global_solution_cost:
                print(
                    f'New best global solution found: cost :{best_solution_cost}, iteration : {i}')
                best_global_solution_cost = best_solution_cost
                best_global_solution = best_solution

        return best_global_solution, self.fitness(best_global_solution)

    # Mesafe hesaplama fonksiyonu (Öklid uzaklığı)
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def getDistance(self, l1, l2):
        """
        Method that computes the rounded euclidian distance between two locations
        """
        dx = l1.x - l2.x
        dy = l1.y - l2.y
        return round(math.sqrt(dx ** 2 + dy ** 2))

    def calculateTrolley(self, loc):
        trolley_count_needed = 1
        curDemand = 0
        for l in range(0, len(loc)):
            curNode = loc[l]
            curDemand += curNode.demand
            calculateTrolley = ((curDemand + self.vehicles.get(1).capacity - 1)
                                // self.vehicles.get(1).capacity)

            if calculateTrolley > trolley_count_needed:
                trolley_count_needed = calculateTrolley

        return trolley_count_needed

    # Fitness fonksiyonu: toplam mesafe ve cezaları içerir
    def fitness(self, solution):
        total_penalty = 0
        total_distance = 0

        for vehicle_id, route in solution.items():
            current_time = 0
            vehicle = self.vehicles.get(vehicle_id)
            route_with_depots = route

            trolley_count_needed = self.calculateTrolley(route)
            if trolley_count_needed > vehicle.max_trolley:
                return sys.maxsize  # Kapasite aşılırsa fitness değeri çok kötü olur

            for i in range(1, len(route_with_depots)):
                current_node = route_with_depots[i]  # Depoyu da dahil et
                prevNode = route_with_depots[i - 1]

                # Mesafe hesapla
                dist = self.distMatrix[prevNode.nodeID][current_node.nodeID]
                total_distance += dist
                current_time = max(0, current_time + prevNode.servTime + dist + (trolley_count_needed * self.TIR))

                # Zaman penceresine göre ceza hesapla
                if current_node.typeLoc == "delivery":
                    if current_time < current_node.startTW:
                        total_penalty += (current_node.startTW - current_time) * self.alpha

                    if current_time > current_node.endTW:
                        total_penalty += (current_time - current_node.endTW) * self.beta

        return total_distance + total_penalty  # Toplam mesafe ve ceza döndürülür

    def computeTimeWindow(self, loc) -> int:
        curTime = 0
        totalTimeWindowPenaly = 0
        trolley_count_needed = self.calculateTrolley(loc)

        if trolley_count_needed > self.vehicles.get(1).max_trolley:
            return sys.maxsize  # Kapasite aşılırsa fitness değeri çok kötü olur

        for i in range(1, len(loc)):
            prevNode = loc[i - 1]
            curNode = loc[i]
            dist = self.distMatrix[prevNode.nodeID][curNode.nodeID]
            curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.TIR))

            ETPenalty = 0
            if curNode.typeLoc == "delivery":
                if curTime < curNode.startTW:
                    ETPenalty = (curNode.startTW - curTime) * self.alpha

                if curTime > curNode.endTW:
                    ETPenalty = (curTime - curNode.endTW) * self.beta
            totalTimeWindowPenaly += ETPenalty
        return totalTimeWindowPenaly

    # Mesafe hesaplama fonksiyonu (örneğin, Öklid mesafesi)
    def compute_cost_add_request(self, route, pickup_idx, delivery_idx, pickup_node, delivery_node):
        # Pickup ve delivery noktalarının mevcut rotaya eklenmesinin maliyetini hesapla
        # route = mevcut rota, pickup_idx = pickup'ın ekleneceği yer, delivery_idx = delivery'nin ekleneceği yer

        # Pickup eklendikten sonra rotada değişen mesafeyi hesapla
        cost_before_pickup = self.distMatrix[route[pickup_idx - 1].nodeID][pickup_node.nodeID]
        cost_after_pickup = self.distMatrix[pickup_node.nodeID][route[pickup_idx].nodeID]

        # Delivery eklendikten sonra rotada değişen mesafeyi hesapla
        cost_before_delivery = self.distMatrix[route[delivery_idx - 1].nodeID][delivery_node.nodeID]
        cost_after_delivery = self.distMatrix[delivery_node.nodeID][route[delivery_idx].nodeID]

        cost3 = self.computeTimeWindow(route)

        total_cost = (cost_before_pickup + cost_after_pickup + cost_before_delivery + cost_after_delivery + cost3)

        return total_cost

    def greedy_insert(self, solution, vehicle_id, pickup_node, delivery_node):
        best_cost = sys.maxsize
        best_route = None

        # Eğer sadece depo varsa, ilk ekleme özel bir duruma tabi
        if len(solution.get(vehicle_id)) == 2:  # Yalnızca depo varsa
            # Depo dışındaki ilk eklemeyi yap (depo ile depo arasına pickup ve delivery eklenir)
            temp_route = [self.depot] + [pickup_node] + [delivery_node] + [self.depot]
            # 3. Şart: Maliyeti hesapla ve en iyi maliyeti bul
            insertion_cost = self.compute_cost_add_request(temp_route, 1, 2, pickup_node, delivery_node)

            # En düşük maliyetli yerleşimi bul
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_route = temp_route

            return best_route, best_cost
        else:
            # Mevcut rotada pickup ve delivery eklemek için tüm olası pozisyonları dene
            for i in range(1, len(solution[vehicle_id])):  # Pickup pozisyonu için depo sonrası ve depo öncesi hariç
                for j in range(i + 1, len(solution[vehicle_id]) + 1):  # Delivery pozisyonu (pickup'tan sonra olmalı)
                    # Geçici olarak pickup ve delivery eklenmiş rotayı oluştur (depoyu sabit tutarak)
                    temp_route = solution[vehicle_id][:i] + [pickup_node] + solution[vehicle_id][i:j - 1] + [
                        delivery_node] + solution[vehicle_id][j - 1:]

                    # 1. Şart: Trolley kapasitesi kontrolü
                    trolley_needed = self.calculateTrolley(temp_route)
                    if trolley_needed > self.vehicles.get(vehicle_id).max_trolley:
                        break  # Kapasite aşımı varsa geçersiz, diğer pozisyonlara bak

                    # # 2. Şart: Pickup önce, delivery sonra olmalı
                    # if pickup_node.nodeID > delivery_node.nodeID:
                    #     break  # Eğer pickup, delivery'den büyükse bu araca ekleme yapılmaz

                    # 3. Şart: Maliyeti hesapla ve en iyi maliyeti bul
                    insertion_cost = self.compute_cost_add_request(temp_route, i, j, pickup_node, delivery_node)

                    # En düşük maliyetli yerleşimi bul
                    if insertion_cost < best_cost:
                        best_cost = insertion_cost
                        best_route = temp_route

        return best_route, best_cost

    # Popülasyon başlatma: her araca rastgele bir çözüm atıyoruz
    def initialize_population(self):
        population = []

        # Her araç için bir çözüm oluştur
        for i in range(self.population_size):  # Popülasyon boyutu
            solution = {}
            self.sort_random_request()
            # Pickup ve delivery noktalarının kopyalarını oluştur
            available_pickups = self.pickupNodes.copy()
            available_deliveries = self.deliveryNodes.copy()

            for v_id in self.vehicles.keys():
                solution[v_id] = [self.depot] + [self.depot]  # Başlangıç olarak her aracın rotasını depodan başlatıyoruz

            while available_pickups:
                pickup_id = random.choice(list(available_pickups.keys()))  # Rastgele bir pickup noktası seç
                pickup_node = available_pickups.get(pickup_id)
                delivery_node = next((delivery for delivery in available_deliveries.values() if
                                      delivery.requestID == pickup_node.requestID), None)

                if delivery_node is None:
                    continue  # Eğer uygun bir delivery noktası yoksa sıradaki pickup noktasına geç

                min_cost = sys.maxsize
                best_vehicle_id = None
                best_route = None

                # En düşük maliyetli aracı bulmak için her aracı dene
                for v_id in self.vehicles.keys():
                    temp_route, temp_cost = self.greedy_insert(solution, v_id, pickup_node, delivery_node)

                    if temp_route is not None and temp_cost < min_cost:
                        min_cost = temp_cost
                        best_vehicle_id = v_id
                        best_route = temp_route

                # En düşük maliyetli araca atama yap
                if best_vehicle_id is not None:
                    solution[best_vehicle_id] = best_route

                    # Eklenen pickup ve delivery noktalarını kaldır
                    delivery_node_key = next(key for key, val in available_deliveries.items() if val == delivery_node)
                    del available_deliveries[delivery_node_key]
                    del available_pickups[pickup_id]  # Eklenen pickup noktasını kaldır

            population.append(solution)

        return population

    def selection(self, population):
        # Popülasyondan rastgele 3 birey seç ve onları uygunluklarına göre sırala
        selected = random.sample(population, self.selection_size)
        selected.sort(key=self.fitness)  # Uygunluk değerine göre sırala
        return selected[0]  # En iyi (en küçük mesafeye sahip) olanı geri döndür

    # Çaprazlama (Crossover): İki ebeveynden yeni bireyler oluşturuyoruz
    # Çaprazlama, iki çözümün genetik materyallerini karıştırarak yeni bireyler (çocuklar) üretir.
    # Bu sayede popülasyona yeni ve daha iyi çözüm adayları eklenir.
    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}

        for v in range(1, len(self.vehicles) + 1):
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
            child1[v] = [self.depot] + child1_route + [self.depot]
            child2[v] = [self.depot] + child2_route + [self.depot]


        return child1, child2

    # Mutasyon (Mutation): Çözüme rastgele değişiklikler yapıyoruz
    # Mutasyon, çözümde küçük rastgele değişiklikler yaparak popülasyondaki çeşitliliği korur.
    # Aksi takdirde algoritma, yerel minimuma sıkışabilir.
    def mutate(self, solution):
        vehicle = random.randint(1, len(self.vehicles))  # Rastgele bir araç seç

        if len(solution[vehicle]) > 2:  # En az 2 düğüm varsa mutasyon yapılabilir
            valid_range = range(1, len(solution[vehicle]) - 1)  # Depolar hariç

            # Rastgele iki düğümün yerini değiştir
            i, j = random.sample(valid_range, 2)

            solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]

            if self.is_route_valid(solution[vehicle]) == False:
                # Swap geçerli değilse çözümü geri çevir
                solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]


        return solution

    # Swap işleminin geçerli olup olmadığını kontrol eden fonksiyon
    def is_valid_swap(self, route, i, j):
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
    def is_route_valid(self, route):
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
    def print_solution(self, solution):
        resultArray = list()
        totalResultDistance, totalResultPenalty, totalCumSum = 0, 0, 0
        for vehicle_id, route in solution.items():
            if len(route) > 2:
                result = dict()
                trolley_count = self.calculateTrolley(route)  # Troley sayısını hesapla
                # print(f"# Vehicle {vehicle_id} (trolley count = {trolley_count}):")
                result["trolleyCount"] = trolley_count
                result["vehicleId"] = vehicle_id
                total_distance = 0
                total_penalty = 0
                cumsum = 0
                current_time = 0

                routeDetail = list()
                routeStringIds = list()
                costDetail = list()
                for i in range(len(route)):

                    node = route[i]
                    Epenalty = 0
                    Tpenalty = 0
                    distance = 0

                    routeStringIds.append(node.stringId)
                    if i > 0:
                        previous_node = route[i - 1]
                        distance = self.distMatrix[previous_node.nodeID][node.nodeID]
                        total_distance += distance
                        current_time = max(0, current_time + previous_node.servTime + distance + (
                                trolley_count * self.TIR))

                        if node.typeLoc == 'delivery':
                            if current_time < node.startTW:
                                Epenalty = (node.startTW - current_time) * self.alpha
                            elif current_time > node.endTW:
                                Tpenalty = (current_time - node.endTW) * self.beta

                        total_penalty += Epenalty + Tpenalty
                        cumsum += distance + Epenalty + Tpenalty

                    routeDetail.append(f" ({node.stringId}, Demand: {node.demand}, "
                        f"CurrentTime: {current_time}, {node.typeLoc}, Distance: {distance}, "
                        f"Start: {node.startTW}, End: {node.endTW}, "
                        f"ServiceTime: {node.servTime}, EarlinessPenalty: {Epenalty}, TardinessPen: {Tpenalty}, cumsum: {distance + Epenalty + Tpenalty})")

                    costDetail.append([distance, Epenalty, Tpenalty])
                    # print(f" ({node.stringId}, Demand: {node.demand}, "
                    #     f"CurrentTime: {current_time}, {node.typeLoc}, Distance: {distance}, "
                    #     f"Start: {node.startTW}, End: {node.endTW}, "
                    #     f"ServiceTime: {node.servTime}, EarlinessPenalty: {Epenalty}, TardinessPen: {Tpenalty}, cumsum: {distance + Epenalty + Tpenalty})")

                result["route"] = routeStringIds
                result["routeDetail"] = routeDetail
                result["costDetail"] = costDetail
                totalResultDistance += total_distance
                totalResultPenalty += total_penalty
                totalCumSum += cumsum
                resultArray.append(result)
                # print(f"# Total Distance: {total_distance}, Total Penalty: {total_penalty}, Cumulative Sum: {cumsum}\n")

        # print(
        #     f"# All Total Distance: {totalResultDistance}, All Total Penalty: {totalResultPenalty}, All Cumulative Sum: {totalCumSum}\n")
        return totalCumSum, resultArray

    def compute_distMatrix(self):
        self.distMatrix = np.zeros((len(self.locations), len(self.locations)))  # init as nxn matrix
        for i in self.locations:
            for j in self.locations:
                xLoc = self.locations.get(i)
                yLoc = self.locations.get(j)
                distItoJ = self.getDistance(xLoc, yLoc)
                self.distMatrix[xLoc.nodeID, yLoc.nodeID] = distItoJ

