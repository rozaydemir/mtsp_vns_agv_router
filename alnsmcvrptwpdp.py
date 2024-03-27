import copy
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

# MCVRPTW contains an implementation of the ALNS with Ruin-and-Recreate


import vrplib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from haversine import haversine
from alns import ALNS, State
from alns.accept import *
from alns.select import *
from alns.stop import *

# from read_data import read_vrp_data

SEED = 1234
# speed = 60/3600

# f = open()

# -----------------
# Initialize Data
# -----------------
# vrp_data, vehicle_data = read_vrp_data(batch=6)
fileName = "lrc104.txt"
customers = list()
num_customers = 0
corr_xy = list()
demands = list()
start_time = list()
end_time = list()
serve_time = list()
stationType = list()

vehicles = list([0,1,2])
num_vehicle = len(list(vehicles))
capacitys_type = list([100,100,100])
capacitys = list([300,300,300])
vehicle_lens_capacity_mapping = {
    "4.2": [12, 13, 14, 14.4, 15.23, 15.45, 15.53, 15.55, 15.96, 16, 16.01, 16.2, 16.22, 16.35, 16.55, 16.88, 16, 18, 20, 22],
    "3": [],
    "2.8": []
}


def readInstance(fileName):
    """
    Method that reads an instance from a file and returns the instancesf
    """
    global corr_xy
    global customers
    global demands
    global start_time
    global end_time
    global serve_time
    global num_customers
    global stationType
    servStartTime = 0  # serviceTime
    f = open(fileName)
    requests = list()
    # stations = list()
    unmatchedPickups = dict()
    unmatchedDeliveries = dict()
    nodeCount = 0
    requestCount = 1  # start with 1
    for line in f.readlines()[1:-6]:
        asList = []
        n = 13  # satırların sondan 13 karakteri booş o yüzden
        for index in range(0, len(line), n):
            asList.append(line[index: index + n].strip())

        lID = asList[0]  # location tipi  D : depot, S: station, C : pickup / delivery point,
        x = int(asList[2][:-2])  # need to remove ".0" from the string
        y = int(asList[3][:-2])
        if lID.startswith("D"):  # depot ise
            # it is the depot
            # depot = Location(0, x, y, 0, 0, 0, 0, servStartTime, "depot", nodeCount, "D0")  # depot requestID=0
            corr_xy.append((x, y))
            nodeCount += 1

        elif lID.startswith("C"):  # pickup/delivery point ise
            # it is a location

            lType = asList[1]
            demand = int(asList[4][:-2])
            startTW = int(asList[5][:-2])
            endTW = int(asList[6][:-2])
            servTime = int(asList[7][:-2])
            # partnerID = asList[8]
            corr_xy.append((x, y))
            customers.append(lID)
            demands.append(demand)
            start_time.append(startTW)
            end_time.append(endTW)
            serve_time.append(servTime)
            stationType.append(lType)
    num_customers = len(customers)
    print(corr_xy)

def nearest_neighbor():
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity and TW limit.
    """

    routes = []
    unvisited = list(range(1, num_customers))
    time_remaining = [0] * num_customers

    while unvisited:
        route = [0]  # Start at the depot
        route_demands = 0
        vehicle_id = 0
        time_elapsed = start_time[0]  # Depotun başlangıç zamanı
        # Depot start time

        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(current) if nb in unvisited][0]

            # ------------------
            # Kapasite kısıtı
            # Capacity constraint
            # ------------------
            # Check if adding the nearest customer violates the capacity constraint of the current vehicle
            if route_demands + demands[nearest] > capacitys[vehicle_id]:
                # If it does, try the next vehicle
                vehicle_id += 1
                if vehicle_id >= num_vehicle:
                    break
                continue

            # # ------------------
            # # Zaman penceresi kısıtı
            # # Time window constraint
            # # ------------------
            if capacitys[vehicle_id] in vehicle_lens_capacity_mapping.get("4.2"):
                time_elapsed += 1.5 * 60 * 60
            else:
                time_elapsed += 1.0 * 60 * 60

            # if time_elapsed + distance[current][nearest] / serve_time[nearest] > end_time[nearest]:
            #     break
            if time_elapsed + distance[current][nearest] / serve_time[nearest] < start_time[nearest]:
                time_elapsed = start_time[nearest]

            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += demands[nearest]
            time_elapsed += distance[current][nearest] / serve_time[nearest]
            time_remaining[nearest] = end_time[nearest] - time_elapsed


        customers = route[1:]  # Remove the depot
        print("customers",customers)
        routes.append(customers)
    print("routes:>>>>>>>>>>",routes) # Başlangıç çözümü
    # Initial solution

    return CvrpState(routes)


def get_distance():
    # Calculate the distance matrix
    distance_matrix = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(i + 1, num_customers):
            dis = haversine(corr_xy[i], corr_xy[j])
            distance_matrix[i][j] = distance_matrix[j][i] = dis
    return distance_matrix



# ------------------------
# Solution state
# ------------------------
class CvrpState(State):
    """
    Solution state for CVRP. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Unassigned is a list of integers,
    each integer representing an unassigned customer.
    """

    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        #TODO Minimum number of vehicles
        return sum(route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")

def route_cost(route):
    tour = [0] + route + [0]
    return sum(distance[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))

# ------------------------
# Repair operators
# ------------------------
def greedy_repair(state, rnd_state, vehicle_id):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    rnd_state.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state, vehicle_id)

        if route is not None:
            route.insert(idx, customer)
        else:
            state.routes.append([customer])

    return state

def best_insert(customer, state, vehicle_id):
    """
    Finds the best feasible route and insertion index for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        for idx in range(len(route) + 1):

            if can_insert(customer, route, vehicle_id):
                cost = insert_cost(customer, route, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


# TODO kapasite farklı, diğer kısıtlamalar da burada dikkate alınmalı
# TODO capacity is different, other constraints should also be considered here
def can_insert(customer, route, vehicle_id):
    """
    Checks if inserting customer does not exceed vehicle capacity.
    """
    # TODO talep kapasitesi
    # TODO demand capacity
    total = sum(demands[cust] for cust in route) + demands[customer]
    return total <= capacitys[vehicle_id]


def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Müşteri eklenerek maliyet artışı
    # Increase in cost by adding the customer
    cost = distance[pred][customer] + distance[customer][succ]
    cost -= distance[pred][succ]
    return cost


def repair(state, rnd_state):
    """
    Applies a set of repair operators to the solution state until all
    constraints are satisfied.
    """
    for vehicle_type in range(num_vehicle):
        state = greedy_repair(state, rnd_state, vehicle_type)

    return state


def intra_relocate(state: CvrpState) -> CvrpState:
    """
    Perform intra-route relocation operator. This operator removes a customer from one route and inserts it
    into another position in the same route.
    """
    # Bir rota rastgele seç
    # Randomly select a route
    route_idx = np.random.choice(len(state.routes))
    route = state.routes[route_idx]

    # En az iki müşteri olduğundan emin ol
    # Ensure there are at least two customers
    if len(route) < 3:
        return state

    # Rastgele bir müşteri seç
    # Randomly select a customer
    customer_idx = np.random.choice(range(1, len(route) - 1))
    customer = route[customer_idx]

    # Ekleme yapılacak pozisyonu rastgele seç
    # Randomly select an insertion position
    insert_idx = np.random.choice(range(1, len(route)))

    # Ekleme pozisyonu ve müşterinin bulunduğu pozisyon aynıysa işlem yapma
    # If the insertion position and the customer's current position are the same, do not proceed
    if insert_idx == customer_idx:
        return state

    # Yeni rotayı hesapla
    # Calculate the new route
    new_route = route[:customer_idx] + route[customer_idx + 1:]
    new_route = new_route[:insert_idx] + [customer] + new_route[insert_idx:]

    # Değişiklik sonrası maliyeti hesapla
    # Calculate the cost after modification
    old_cost = sum(route_cost(_route) for _route in route)
    new_cost = sum(route_cost(_new_route) for _new_route in new_route)

    # Eğer yeni maliyet daha iyiyse, değişikliği kabul et
    # If the new cost is better, accept the modification
    if new_cost < old_cost:
        state.routes[route_idx] = new_route

    return state


def inter_relocate(state: CvrpState) -> CvrpState:
    """
    Perform inter-route relocation operator. This operator removes a customer from one route and inserts it
    into another route at a different position.
    """
    # En az iki rota olduğundan emin ol
    # Ensure there are at least two routes
    if len(state.routes) < 2:
        return state

    # İki farklı rota rastgele seç
    # Randomly select two different routes
    route_idxs = np.random.choice(len(state.routes), size=2, replace=False)
    route1, route2 = state.routes[route_idxs[0]], state.routes[route_idxs[1]]
    # Her iki rotada da en az bir müşteri olduğundan emin ol
    # Ensure both routes have at least one customer
    if len(route1) < 2 or len(route2) < 2:
        return state

    # Bir müşteri rastgele seç
    # Randomly select a customer
    customer_idx = np.random.choice(range(1, len(route1) - 1))
    customer = route1[customer_idx]

    # Ekleme yapılacak pozisyonu rastgele seç
    # Randomly select an insertion position
    insert_idx = np.random.choice(range(1, len(route2)))

    # Yeni rotaları hesapla
    # Calculate the new routes
    new_route1 = route1[:customer_idx] + route1[customer_idx + 1:]
    new_route2 = route2[:insert_idx] + [customer] + route2[insert_idx:]

    # Maliyeti hesapla ve karşılaştır
    # Calculate and compare the costs
    old_cost = sum(route_cost(_route1) for _route1 in route1) + sum(route_cost(_route2) for _route2 in route2)
    new_cost = sum(route_cost(_new_route1) for _new_route1 in new_route1)+sum(route_cost(_new_route2) for _new_route2 in new_route2)

    if new_cost < old_cost:
        state.routes[route_idxs[0]] = new_route1
        state.routes[route_idxs[1]] = new_route2

    return state

def exchange(state):
    """
    Exchange the customers between two positions in two different routes.
    """
    # Check if the exchange is valid
    # İki farklı rota rastgele seç
    # Randomly select two different routes
    route_idxs = np.random.choice(len(state.routes), size=2, replace=False)
    route1, route2 = state.routes[route_idxs[0]], state.routes[route_idxs[1]]
    pos1 = np.random.choice(range(0, len(route1)))
    pos2 = np.random.choice(range(0, len(route2)))

    if route1 == route2 or pos1 == 0 or pos2 == 0:
        return None

    route1_demand = sum(demands[i] for i in route1)
    route2_demand = sum(demands[i] for i in route2)

    if route1_demand - demands[route1[pos1]] + demands[route2[pos2]] > capacitys[route_idxs[0]] or \
            route2_demand - demands[route2[pos2]] + demands[route1[pos1]] > capacitys[route_idxs[1]]:
        return None

    # Perform the exchange
    state.routes[route_idxs[0]][pos1], state.routes[route_idxs[1]][pos2] = state.routes[route_idxs[1]][pos2], state.routes[route_idxs[0]][pos1]
    return state


def neighbors(customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(distance[customer])
    return locations[locations != 0]




# ------------------------
# Slack nedeniyle alt dizi kaldırma
# Slack-induced substring removal
# ------------------------
MAX_STRING_REMOVALS = 3
MAX_STRING_SIZE = 10

def string_removal(state, rnd_state):
    """
    Remove partial routes around a randomly chosen customer.
    """
    destroyed = state.copy()

    avg_route_size = int(np.mean([len(route) for route in state.routes]))
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

    destroyed_routes = []
    center = rnd_state.randint(1, num_customers) # Yapılacak

    for customer in neighbors(center):
        if len(destroyed_routes) >= max_string_removals:
            break

        if customer in destroyed.unassigned:
            continue

        route = destroyed.find_route(customer)
        if route in destroyed_routes:
            continue

        customers = remove_string(route, customer, max_string_size, rnd_state)
        destroyed.unassigned.extend(customers)
        destroyed_routes.append(route)

    return destroyed


def remove_string(route, cust, max_string_size, rnd_state):
    """
    Remove a string that contains the passed-in customer.
    """
    # Müşteriyi içeren ardışık indeksleri kaldır
    # Find consecutive indices to remove that contain the customer
    size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
    start = route.index(cust) - rnd_state.randint(size)
    idcs = [idx % len(route) for idx in range(start, start + size)]

    # İndeksleri azalan sırayla kaldır
    # Remove indices in descending order
    removed_customers = []
    for idx in sorted(idcs, reverse=True):
        removed_customers.append(route.pop(idx))

    return removed_customers


# ------------------------
# Sezgisel çözüm
# Heuristic solution
# ------------------------

readInstance(fileName)
distance = get_distance()

alns = ALNS(rnd.RandomState(SEED))
alns.add_destroy_operator(string_removal)
#alns.add_repair_operator(greedy_repair)
alns.add_repair_operator(repair)

init = nearest_neighbor()
select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 100)
stop = MaxRuntime(60*15)

result = alns.iterate(init, select, accept, stop)

solution = result.best_state
objective = solution.objective()

print(f"Müşteri sayısı {num_customers}")
print(f"En iyi sezgisel objektif değer {objective}.")
print(f"çözüm: {solution.routes}")

routes_solutions = [i for i in solution.routes if len(i) != 0]
print(f"çözüm: {routes_solutions}")
# Her rotanın yükünü hesapla
# Calculate the load for each route
routes_loads = [sum(demands[cus] for cus in route) for route in routes_solutions]
print("routes_loads",routes_loads)
# Her rotanın yüküne en uygun aracı bul, kriter: araç kapasitesi içinde mümkün olduğunca fazla yük taşıması
# Match the best vehicle type for each route's loads, criterion: the more a vehicle can carry within its capacity, the better
# def find_match_vehicle_type(route_load,capacitys_type):
#     best_vehicle_type = 10000
#     for ty in capacitys_type:
#         if ty >= route_load:
#             best_vehicle_type = min(best_vehicle_type,ty-route_load)
#     return best_vehicle_type+route_load
#
# routes_vehicle_type = [find_match_vehicle_type(route_load,capacitys_type) for route_load in routes_loads]
# print(f"rotaların tipi: {routes_vehicle_type}")

# Araç id'sini rastgele seç, daha fazla araç tipi varsa daha büyük araçları eşleştir
# Randomly select vehicle id, match with larger vehicles if there are multiple vehicle types
# from collections import Counter
# print(f"Tüm araç tiplerinin sayımı: {Counter(routes_vehicle_type)}")

print(f"Tüm düğümlerin kapsandığını doğrula {len(sum(solution.routes,[]))} ,{num_customers}")
print(f"Minimum araç sayısı: {len(routes_solutions)}")


