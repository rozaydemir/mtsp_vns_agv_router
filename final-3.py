import random
import math
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

vehiclesList = [0, 1]  # vehicle değişkeni.
vehicleCount = 2  # bu 2 aracım olduğu anlamına gelir.
vehicle_capacities = {0: 30, 1: 50}  # Her iki aracın da kapasitesi
location = [
    (5, 10, "p", 95, 10), (10, 15, "p", 110, 10), (10, 15, "p", 82, 15), (25, -20, "p", 95, 5),
            (-38, 20, "p", 85, 5), (45, 10, "d", -30, 12), (15, 10, "d", -22, 25), (-35, 30, "d", -10, 20),
            (-15, 15, "d", -35, 25), (55, 25, "d", -25, 20)
]
# location : pickup ve delivery noktalarımın bilgileri, tupple'ın ilk 2 parametresi x ve y koordinant noktaları,
# tupple'ın 3 parametresi "p" ise "pickup" noktası, "d" ise delivery noktasıdır.
# tupple'ın 4. parametresi demand bilgisi (location "p" ise araca aktarılan yük , location "d" ise araçtan alınan yük),
# tupple'ın 5. parametresi yük yüklenirken ve o noktaya giderkenki toplam servis süresi
depotLocation = (0, 0)  # depo konumu koordinatı, araçların siparişi başlattığı ve bitirdiği yer

class Vehicles:
    def __init__(self, id):
        self.vehiclesId = id
        self.trolleyCount = 1
        self.trolleyCapacity = 100
        self.routes = []
        self.maxTrolleyCount = 3
        self.vehicleCurrentDemand = 0

    def increaseTrolleyCapacity(self):
        self.trolleyCount += 1

    def getVehicleTotalCapacity(self):
        return self.trolleyCount * self.trolleyCapacity

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

    def __init__(self, requestID, xLoc, yLoc, demand, servTime, servStartTime, typeLoc, stringId):
        self.requestID = requestID
        self.xLoc = xLoc
        self.yLoc = yLoc
        self.demand = demand
        # self.startTW = startTW
        # self.endTW = endTW
        self.servTime = servTime
        self.servStartTime = servStartTime
        self.typeLoc = typeLoc
        self.stringId = stringId

    def __str__(self):
        return (f"requestID: {self.requestID}; demand: {self.demand}; servTime:{self.servTime}; servStartTime: {self.servStartTime}; typeLoc: {self.typeLoc}, stringId: {self.stringId}")

    def print(self):
        """
        Method that prints the location
        """
        print(f" ( StringId: {self.stringId}, LocType: {self.typeLoc}, demand: {self.demand}; servTime:{self.servTime} ) ", end='')

    def printOnlyRoute(self):
        """
        Method that prints the location
        """
        print(f" ( {self.stringId}, {self.demand}, {self.servStartTime}, {self.typeLoc}, {self.demand} ) ", end='')

    def getDistance(l1, l2):
        """
        Method that computes the rounded euclidian distance between two locations
        """
        dx = l1.xLoc - l2.xLoc
        dy = l1.yLoc - l2.yLoc
        return round(math.sqrt(dx ** 2 + dy ** 2))

class MCVRPPDTWState(State):
    """
    Solution state for CVRP. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Unassigned is a list of integers,
    each integer representing an unassigned customer.
    """

    def __init__(self, routes, depot, vehicles):
        self.routes = routes
        self.depot = depot
        self.vehicles = vehicles



    def copy(self):
        return MCVRPPDTWState(copy.deepcopy(self.routes))

    def objective(self):
        """
        Computes the total route costs.
        """
        #TODO Minimum number of vehicles
        # return sum(route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        # return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        # for route in self.routes:
        #     if customer in route:
        #         return route
        #
        # raise ValueError(f"Solution does not contain customer {customer}.")


def euclidean_distance(x, y):
    return math.sqrt(x**2 + y**2)

def neighbors(current, unvisitedLocation):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    if current.typeLoc == 'depot':
        # 'depot' için, 'pickup' olanları seç
        return filter(lambda u: u.typeLoc == 'pickup', unvisitedLocation)
    elif current.typeLoc == 'pickup':
        # 'pickup' için, 'delivery' olanları seç (örnekte 'delivery' yok, varsayım)
        return filter(lambda u: u.typeLoc == 'delivery', unvisitedLocation)
    elif current.typeLoc == 'delivery':
        # 'delivery' için, 'pickup' olanları seç
        return filter(lambda u: u.typeLoc == 'pickup', unvisitedLocation)


def executeRandomInsertion(self, randomGen, routeCnt=0):
    """
    Method that randomly inserts the unserved requests in the solution

    This is repair method number 1 in the ALNS

    Parameters
    ----------
    randomGen : Random
        Used to generate random numbers

    """
    # iterate over the list with unserved requests
    while len(self.notServed) > 0:
        # pick a random request
        req = randomGen.choice(self.notServed)

        # keep track of routes in which req could be inserted
        potentialRoutes = self.routes.copy()
        inserted = False
        while len(potentialRoutes) > 0:
            # pick a random route
            randomRoute = randomGen.choice(potentialRoutes)

            afterInsertion, cost = randomRoute.greedyInsert(req, routeCnt)
            if afterInsertion == None:
                # insertion not feasible, remove route from potential routes
                potentialRoutes.remove(randomRoute)
            else:
                # insertion feasible, update routes and break from while loop
                inserted = True
                # print("Possible")
                self.routes.remove(randomRoute)
                self.routes.append(afterInsertion)
                self.distance += cost
                # afterInsertion.print()
                break

        # if we were not able to insert, create a new route
        if not inserted:
            # create a new route with the request
            locList = [self.problem.depot, req.pickUpLoc, req.deliveryLoc, self.problem.depot]
            newRoute = Route(locList, {req}, self.problem, routeCnt)
            self.routes.append(newRoute)
            self.distance += newRoute.distance
            newRoute.print()
        # update the lists with served and notServed requests
        self.served.append(req)
        self.notServed.remove(req)
def init_solution(fileName):

    global vehiclesList
    global vehicleCount
    global vehicle_capacities
    global location
    global depotLocation

    f = open(fileName)
    requests = list()
    unmatchedPickups = dict()
    unmatchedDeliveries = dict()
    nodeCount = 0
    requestCount = 1  # start with 1
    depot = Location(0, 0, 0, 0, 0, servStartTime, "depot", "D0")  # depot requestID=0
    # locations.append(depot)

    # for line in f.readlines()[1:-6]:
    #     asList = []
    #     n = 13  # satırların sondan 13 karakteri booş o yüzden
    #     for index in range(0, len(line), n):
    #         asList.append(line[index: index + n].strip())
    #
    #     lID = asList[0]  # location tipi  D : depot, S: station, C : pickup / delivery point,
    #     x = int(asList[2][:-2])  # need to remove ".0" from the string
    #     y = int(asList[3][:-2])
    #     if lID.startswith("D"):  # depot ise
    #         # it is the depot
    #         depot = Location(0, x, y, 0, 0, 0, 0, servStartTime, "depot", nodeCount, "D0")  # depot requestID=0
    #         nodeCount += 1
    #
    #     elif lID.startswith("C"):  # pickup/delivery point ise
    #         # it is a location
    #         lType = asList[1]
    #         demand = int(asList[4][:-2])
    #         startTW = int(asList[5][:-2])
    #         endTW = int(asList[6][:-2])
    #         servTime = int(asList[7][:-2])
    #         partnerID = asList[8]
    #         if lType == "cp":  # cp ise pickup, #cd ise delivery point
    #             if partnerID in unmatchedDeliveries:
    #                 deliv = unmatchedDeliveries.pop(
    #                     partnerID)  # pop listeden siler, sildiği değeri ise bir değişkene atar, burada deliv değişkenine atadı
    #                 pickup = Location(deliv.requestID, x, y, demand, startTW, endTW, servTime, servStartTime,
    #                                   "pickup", nodeCount, lID)
    #                 nodeCount += 1
    #                 req = Request(pickup, deliv, deliv.requestID)
    #                 requests.append(req)
    #             else:
    #                 pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, servStartTime, "pickup",
    #                                   nodeCount, lID)
    #                 nodeCount += 1
    #                 requestCount += 1
    #                 # lID -> partnerID
    #                 unmatchedPickups[lID] = pickup
    #         elif lType == "cd":  # cp ise pickup, #cd ise delivery point
    #             if partnerID in unmatchedPickups:
    #                 pickup = unmatchedPickups.pop(partnerID)
    #                 deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, servStartTime,
    #                                  "delivery", nodeCount, lID)
    #                 nodeCount += 1
    #                 req = Request(pickup, deliv, pickup.requestID)
    #                 requests.append(req)
    #             else:
    #                 deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, servStartTime,
    #                                  "delivery", nodeCount, lID)
    #                 nodeCount += 1
    #                 requestCount += 1
    #                 unmatchedDeliveries[lID] = deliv
    for line in location:
        x, y, type, demand, service_time = line
        requestCount += 1
        if type == "p":  # cp ise pickup, #cd ise delivery point
            lID = "P" + str(requestCount)
            pickup = Location(requestCount, x, y, demand, service_time, servStartTime,
                              "pickup", lID)


            locations.append(pickup)
        elif type == "d":  # cp ise pickup, #cd ise delivery point
            lID = "D" + str(requestCount)
            deliv = Location(requestCount, x, y, demand, service_time, servStartTime,
                             "delivery", lID)

            locations.append(deliv)

    routes = []
    vehicles = list()
    unvisitedLocation = locations.copy()
    for i in range(vehicleCount):
        vehicles.append(Vehicles(i))

    while unvisitedLocation:
        route = [depot]
        route_demands = depot.demand
        vehicle_id = 0
        while unvisitedLocation:
            current = route[-1]
            nearests = [nb for nb in neighbors(current, unvisitedLocation) if nb in unvisitedLocation]
            nearest = random.choice(nearests)

            currentDemand = route_demands + nearest.demand
            totalCapacity = vehicles[vehicle_id].getVehicleTotalCapacity()

            if currentDemand >= totalCapacity:
                if vehicles[vehicle_id].trolleyCount >= vehicles[vehicle_id].maxTrolleyCount:
                    vehicle_id += 1
                    if vehicle_id >= vehicleCount:
                        break
                else:
                    vehicles[vehicle_id].vehicleCurrentDemand = currentDemand
                    if vehicles[vehicle_id].vehicleCurrentDemand >= vehicles[vehicle_id].trolleyCount * vehicles[vehicle_id].trolleyCapacity:
                        vehicles[vehicle_id].increaseTrolleyCapacity()
            else:
                vehicles[vehicle_id].vehicleCurrentDemand = currentDemand
                if vehicles[vehicle_id].vehicleCurrentDemand >= vehicles[vehicle_id].trolleyCount * vehicles[vehicle_id].trolleyCapacity:
                    vehicles[vehicle_id].increaseTrolleyCapacity()




            route.append(nearest)
            unvisitedLocation.remove(nearest)
            route_demands += nearest.demand

        print(route_demands)
        customers = route[1:]  # Remove the depot
        routes.append(customers)
        print(routes)

    for vehicle in vehicles:
        finalTrolleyCnt = 0
        if vehicle.vehicleCurrentDemand <= vehicle.trolleyCapacity:
            finalTrolleyCnt = 1
        elif vehicle.vehicleCurrentDemand >= vehicle.maxTrolleyCount * vehicle.trolleyCapacity:
            finalTrolleyCnt = vehicle.maxTrolleyCount
        else:
            finalTrolleyCnt = vehicle.vehicleCurrentDemand // vehicle.trolleyCapacity

        vehicle.trolleyCount = finalTrolleyCnt

    # read the vehicle capacity
    return MCVRPPDTWState(locations, depot, vehicles)




# alns = ALNS(rnd.RandomState(SEED))
# alns.add_destroy_operator(string_removal)
# #alns.add_repair_operator(greedy_repair)
# alns.add_repair_operator(repair)

init = init_solution()
# select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
# accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 6000)
# stop = MaxRuntime(60*15)
#
# result = alns.iterate(init, select, accept, stop)
#
# solution = result.best_state
# objective = solution.objective()
#
# print(f"Müşteri sayısı {num_customers}")
# print(f"En iyi sezgisel objektif değer {objective}.")
# print(f"çözüm: {solution.routes}")
#
# routes_solutions = [i for i in solution.routes if len(i) != 0]
# print(f"çözüm: {routes_solutions}")
# # Her rotanın yükünü hesapla
# # Calculate the load for each route
# routes_loads = [sum(demands[cus] for cus in route) for route in routes_solutions]
# print("routes_loads",routes_loads)
# # Her rotanın yüküne en uygun aracı bul, kriter: araç kapasitesi içinde mümkün olduğunca fazla yük taşıması
# # Match the best vehicle type for each route's loads, criterion: the more a vehicle can carry within its capacity, the better
#
#
#
# print(f"Tüm düğümlerin kapsandığını doğrula {len(sum(solution.routes,[]))} ,{num_customers}")
# print(f"Minimum araç sayısı: {len(routes_solutions)}")