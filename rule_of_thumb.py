# import random, time
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys
# from numpy import log as ln
#
# distMatrix = None
# prevNode = 0
# alpha = 15
# beta = 90
#
# class Vehicles:
#     def __init__(self, id, Mmax, capacityOfTrolley):
#         self.vehiclesId = id
#         self.trolleyCapacity = capacityOfTrolley
#         self.routes = []
#         self.maxTrolleyCount = Mmax
#         self.totalDistance = 0
#         self.totalDemand = 0
#         self.trolleyCount = 1
#
#     def print(self):
#         for route in self.routes:
#             print("Vehicle " + str(self.vehiclesId + 1) + " - cost = " + str(self.totalDistance) + ", demand = " + str(
#                 self.totalDemand) + ", trolley count = " + str(route["trolleyCount"]))
#             route["route"].calculateServiceStartTime()
#             route["route"].print()
# class Request:
#     def __init__(self, pickUpLoc, deliveryLoc, ID):
#         self.pickUpLoc = pickUpLoc
#         self.deliveryLoc = deliveryLoc
#         self.ID = ID
#
#     def __str__(self):
#         return "requestID: {}; pickUpLoc: {}; deliveryLoc: {}".format(self.ID, self.pickUpLoc, self.deliveryLoc)
# class Location:
#     """
#     Class that represents either (i) a location where a request should be picked up
#     or delivered or (ii) the depot
#     Attributes
#     ----------
#     requestID : int
#         id of request.
#     xLoc : int
#         x-coordinate.
#     yLoc : int
#         y-coordinate.
#     demand : int
#         demand quantity, positive if pick-up, negative if delivery
#     startTW : int
#         start time of time window.
#     endTW : int
#         end time of time window.
#     servTime : int
#         service time.
#     typeLoc : int
#         1 if pick-up, -1 if delivery, 0 if depot
#     nodeID : int
#         id of the node, used for the distance matrix
#     """
#
#     def __init__(self, requestID, xLoc, yLoc, demand, startTW, endTW, servTime, servStartTime, typeLoc, nodeID, stringId):
#         global distMatrix
#         global beta
#         global alpha
#         self.requestID = requestID
#         self.xLoc = xLoc
#         self.yLoc = yLoc
#         self.demand = demand
#         self.startTW = startTW
#         self.endTW = endTW
#         self.servTime = servTime
#         self.servStartTime = servStartTime
#         self.typeLoc = typeLoc
#         self.nodeID = nodeID
#         self.stringId = stringId
#
#     def __str__(self):
#         return (f"requestID: {self.requestID}; demand: {self.demand}; startTW: {self.startTW}; endTW: {self.endTW}; servTime:{self.servTime}; servStartTime: {self.servStartTime}; typeLoc: {self.typeLoc}, nodeID: {self.nodeID}, stringId: {self.stringId}")
#
#     def print(self):
#         """
#         Method that prints the location
#         """
#         print(f" ( StringId: {self.stringId}, LocType: {self.typeLoc}, demand: {self.demand}; startTW: {self.startTW}; endTW: {self.endTW}; servTime:{self.servTime} ) ", end='')
#
#     def printOnlyRoute(self):
#         """
#         Method that prints the location
#         """
#         global prevNode
#         global beta
#         global alpha
#         dist = distMatrix[prevNode][self.nodeID]
#         penalty = 0
#         if self.typeLoc == "delivery":
#             if self.servStartTime > self.endTW:
#                 penalty += (self.servStartTime - self.endTW) * beta
#
#             if self.servStartTime < self.startTW:
#                 penalty += (self.startTW - self.servStartTime) * alpha
#
#         print(f" ( {self.stringId}, Demand : {self.demand}, CurrentTime: {self.servStartTime}, {self.typeLoc}, Distance: {dist}, Start: {self.startTW}, "
#               f"End: {self.endTW}, ServiceTime: {self.servTime}, Penalty: {penalty}, cumsum: {penalty + dist} ) ", end='\n')
#         prevNode = self.nodeID
#
#     def getDistance(l1, l2):
#         """
#         Method that computes the rounded euclidian distance between two locations
#         """
#         dx = l1.xLoc - l2.xLoc
#         dy = l1.yLoc - l2.yLoc
#         return round(math.sqrt(dx ** 2 + dy ** 2))
#
# class Route:
#
#     """
#     Class used to represent a route
#
#     Parameters
#     ----------
#     locations : list of locations
#         the route sequence of locations.
#     requests : set of requests
#         the requests served by the route
#     problem : PDPTW
#         the problem instance, used to compute distances.
#     feasible : boolean
#         true if route respects time windows, capacity and precedence # delivery after pickup
#     distance : int
#         total distance driven, extremely large number if infeasible
#     """
#
#     def __init__(self, locations, requests, problem):
#         global distMatrix
#         self.locations = locations
#         self.requests = requests
#         self.problem = problem
#         # check the feasibility and compute the distance
#         self.feasible = self.isFeasible()
#         self.calculateServiceStartTime()
#         if self.feasible:
#             self.distance, self.demand = self.computeDistanceRoute()
#         else:
#             self.distance, self.demand = sys.maxsize, sys.maxsize  # extremely large number
#
#     def calculateServiceStartTime(self):
#         curTime = 0
#         trolley_count_needed = self.calculateTrolley()
#
#         for i in range(1, len(self.locations)):
#             prevNode = self.locations[i - 1]
#             curNode = self.locations[i]
#             dist = distMatrix[prevNode.nodeID][curNode.nodeID]
#             curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.problem.TIR))
#             self.locations[i].servStartTime = curTime
#
#     def computeDistanceRoute(self) -> int:
#         """
#         Method that computes and returns the distance of the route
#         """
#         global alpha
#         global beta
#         totDist = 0
#         totDemand = 0
#         curTime = 0
#         trolley_count_needed = self.calculateTrolley(self.locations)
#         for i in range(1, len(self.locations)):
#             prevNode = self.locations[i - 1]
#             curNode = self.locations[i]
#             dist = distMatrix[prevNode.nodeID][curNode.nodeID]
#             curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.problem.TIR))
#
#             if curNode.typeLoc == 'pickup':
#                 totDemand += curNode.demand
#
#             ETPenalty = 0
#             if curNode.typeLoc == "delivery":
#                 if curTime < curNode.startTW:
#                     ETPenalty += (curNode.startTW - curTime) * alpha
#
#                 if curTime > curNode.endTW:
#                     ETPenalty += (curTime - curNode.endTW) * beta
#             totDist += dist + ETPenalty
#
#         return totDist, totDemand
#
#     def computeDiff(self, preNode, afterNode, insertNode) -> int:
#         '''
#         Method that calculates the cost of inserting a new node
#         Parameters
#         ----------
#         preNode: Location
#         afterNode: Location
#         insertNode: Location
#         '''
#
#         return distMatrix[preNode.nodeID][insertNode.nodeID] + distMatrix[afterNode.nodeID][
#             insertNode.nodeID] - distMatrix[preNode.nodeID][afterNode.nodeID]
#
#     def computeTimeWindow(self, loc) -> int:
#         global alpha
#         global beta
#         curTime = 0
#         totalTimeWindowPenaly = 0
#         trolley_count_needed = self.calculateTrolley(loc)
#
#         for i in range(1, len(loc)):
#             prevNode = loc[i - 1]
#             curNode = loc[i]
#             dist = distMatrix[prevNode.nodeID][curNode.nodeID]
#             curTime = max(0,curTime + prevNode.servTime + dist
#                           + (trolley_count_needed * self.problem.TIR)
#                           )
#
#             ETPenalty = 0
#             if curNode.typeLoc == "delivery":
#                 if curTime < curNode.startTW:
#                     ETPenalty += (curNode.startTW - curTime) * alpha
#
#                 if curTime > curNode.endTW:
#                     ETPenalty += (curTime - curNode.endTW) * beta
#             totalTimeWindowPenaly += ETPenalty
#         return totalTimeWindowPenaly
#
#
#     # add this method
#     def compute_cost_add_one_request(self, preNode_index, afterNode_index, request) -> int:
#         locationsCopy = self.locations.copy()
#         locationsCopy.insert(preNode_index, request.pickUpLoc)
#         # calculate the cost after inserting pickup location
#         cost1 = self.computeDiff(locationsCopy[preNode_index - 1],
#                                  locationsCopy[preNode_index + 1],
#                                  request.pickUpLoc)
#
#         locationsCopy.insert(afterNode_index, request.deliveryLoc)  # depot at the end
#         # calculte the cost after inserting delivery location
#         cost2 = self.computeDiff(locationsCopy[afterNode_index - 1],
#                                  locationsCopy[afterNode_index + 1],
#                                  request.deliveryLoc)
#
#         cost3 = self.computeTimeWindow(locationsCopy)
#
#         return cost2 + cost1 + cost3
#
#     def print(self):
#         """
#         Method that prints the route
#         """
#         for loc in self.locations:
#             loc.printOnlyRoute()
#
#     def calculateTrolley(self, loc = list()) -> int:
#         trolley_count_needed = 1
#         curDemand = 0
#         calculatedLoc = loc if len(loc) > 0 else self.locations
#         for l in range(0, len(calculatedLoc)):
#             curNode = calculatedLoc[l]
#             curDemand += curNode.demand
#             calculateTrolley = ((curDemand + self.problem.vehicles[0].trolleyCapacity - 1)
#                                 // self.problem.vehicles[0].trolleyCapacity)
#             if calculateTrolley > trolley_count_needed:
#                 trolley_count_needed = calculateTrolley
#         return trolley_count_needed
#
#     def isFeasible(self) -> bool:
#         global distMatrix
#         """
#         Method that checks feasbility. Returns True if feasible, else False
#         """
#         # route should start and end at the depot
#         if self.locations[0] != self.problem.depot or self.locations[-1] != self.problem.depot:
#             return False
#
#         if len(self.locations) <= 2 and self.locations[0] == self.problem.depot and self.locations[1] == self.problem.depot:
#             return False
#
#         pickedUp = set()  # set with all requests that we picked up, used to check precedence
#         trolley_count_needed = self.calculateTrolley()
#         if trolley_count_needed > self.problem.vehicles[0].maxTrolleyCount:
#             return False
#
#         # iterate over route and check feasibility of time windows, capacity and precedence
#         for i in range(1, len(self.locations) - 1):
#             curNode = self.locations[i]
#
#             if curNode.typeLoc == "pickup":
#                 # it is a pickup
#                 pickedUp.add(curNode.requestID)
#             else:
#                 # it is a delivery
#                 # check if we picked up the request
#                 if curNode.requestID not in pickedUp:
#                     return False
#                 pickedUp.remove(curNode.requestID)
#
#         # finally, check if all pickups have been delivered
#         if len(pickedUp) > 0:
#             return False
#         return True
#
#     def removeRequest(self, request):
#         """
#         Method that removes a request from the route.
#         """
#         # remove the request, the pickup and the delivery
#         self.requests.remove(request)
#         self.locations.remove(request.pickUpLoc)
#         self.locations.remove(request.deliveryLoc)
#         # the distance changes, so update
#         self.calculateServiceStartTime()
#         self.computeDistanceRoute()
#         # *** add this method
#
#     def copy(self):
#         """
#         Method that returns a copy of the route
#         """
#         locationsCopy = self.locations.copy()
#         requestsCopy = self.requests.copy()
#         return Route(locationsCopy, requestsCopy, self.problem)
#
#     def greedyInsert(self, request, convinent = True):
#         """
#         Method that inserts the pickup and delivery of a request at the positions
#         that give the shortest total distance. Returns best route.
#
#         Parameters
#         ----------
#         request : Request
#             the request that should be inserted.
#
#         Returns
#         -------
#         bestInsert : Route
#             Route with the best insertion.
#
#         """
#         requestsCopy = self.requests.copy()
#         requestsCopy.add(request)
#
#         bestInsert = None  # if infeasible the bestInsert will be None
#         minCost = sys.maxsize
#         minDemand = sys.maxsize
#         # iterate over all possible insertion positions for pickup and delivery
#
#         for i in range(1, len(self.locations)):
#             for j in range(i + 1, len(self.locations) + 1):  # delivery after pickup
#                 locationsCopy = self.locations.copy()
#                 locationsCopy.insert(i, request.pickUpLoc)
#                 locationsCopy.insert(j, request.deliveryLoc)  # depot at the end
#                 afterInsertion = Route(locationsCopy, requestsCopy, self.problem)
#                 # check if insertion is feasible
#                 if afterInsertion.feasible:
#                     # check if cheapest
#                     # revise. only calculate the cost
#                     cost = self.compute_cost_add_one_request(i, j, request)
#                     if cost < minCost:
#                         bestInsert = afterInsertion
#                         minCost = cost
#                         minDemand = afterInsertion.demand
#
#         # diff = ( ( int(self.problem.bestDistanceProblem) - int(self.problem.convProblem) )  /  int(self.problem.bestDistanceProblem) ) * 100
#         if convinent and self.problem.globalIterationNumber < 100:
#             convinent = False
#
#         if convinent:
#             if bestInsert != None:
#                 routeCost = bestInsert.distance
#                 if routeCost > self.problem.TValue:
#                     self.problem.TValue = self.problem.TValue + (self.problem.TValue * 0.3)
#                     bestInsert = None
#                     minCost = sys.maxsize
#                     minDemand = sys.maxsize
#                 else:
#                     self.problem.TValue = self.problem.TValue - (self.problem.TValue * 0.1)
#                     if self.problem.TValue < self.problem.TValueMin - (self.problem.TValueMin * 0.2):
#                         self.problem.TValue = self.problem.TValueMin
#
#         return bestInsert, minCost, minDemand
# class Repair:
#
#     '''
#     Class that represents repair methods
#
#     Parameters
#     ----------
#     problem : PDPTW
#         The problem instance that we want to solve.
#     currentSolution : Solution
#         The current solution in the ALNS algorithm
#     randomGen : Random
#         random number generator
#
#     '''
#
#     def __init__(self, problem, solution):
#         global distMatrix
#         self.problem = problem
#         self.solution = solution
#
#     def computeDiff(self, preNode, afterNode, insertNode) -> int:
#         '''
#         Method that calculates the cost of inserting a new node
#         Parameters
#         ----------
#         preNode: Location
#         afterNode: Location
#         insertNode: Location
#         '''
#
#         return distMatrix[preNode.nodeID][insertNode.nodeID] + distMatrix[afterNode.nodeID][
#             insertNode.nodeID] - distMatrix[preNode.nodeID][afterNode.nodeID]
#
#     def executeGreedyInsertion(self):
#         """
#         Method that greedily inserts the unserved requests in the solution
#
#         This is repair method number 1 in the ALNS
#
#         """
#         random.shuffle(self.solution.served)
#
#         while len(self.solution.notServed) > 0:
#             for req in self.solution.notServed:
#                 inserted = False
#                 minCost = sys.maxsize  # initialize as extremely large number
#                 minDemand = sys.maxsize  # initialize as extremely large number
#                 bestInsert = None  # if infeasible the bestInsert will be None
#                 removedRoute = None  # if infeasible the bestInsert will be None
#                 for route in self.solution.routes:
#                     afterInsertion, cost, demand = route.greedyInsert(req)
#
#                     if afterInsertion == None:
#                         continue
#
#                     if cost < minCost:
#                         inserted = True
#                         removedRoute = route
#                         bestInsert = afterInsertion
#                         minDemand = demand
#                         minCost = cost
#
#
#                 # if we were not able to insert, create a new route
#                 if not inserted:
#                     self.solution.manage_routes(req)
#                 else:
#                     self.solution.updateRoute(removedRoute, bestInsert, minCost, minDemand)
#
#                 # update the lists with served and notServed requests
#                 self.solution.served.append(req)
#                 self.solution.notServed.remove(req)
# class Solution:
#
#     """
#     Method that represents a solution to the PDPTW
#
#     Attributes
#     ----------
#     problem : PDPTW
#         the problem that corresponds to this solution
#     routes : List of Routes
#          Routes in the current solution
#     served : List of Requests
#         Requests served in the current solution
#     notServed : List of Requests
#          Requests not served in the current solution
#     distance : int
#         total distance of the current solution
#     """
#
#     def __init__(self, problem, routes, served, notServed):
#         global distMatrix
#         self.problem = problem
#         self.routes = routes
#         self.served = served
#         self.notServed = notServed
#         self.distance, self.demand = self.computeDistance()
#
#     def computeDistance(self):
#         """
#         Method that computes the distance of the solution
#         """
#         self.distance = 0
#         self.demand = 0
#         for route in self.routes:
#             self.distance += route.distance
#             self.demand += route.demand
#         return self.distance, self.demand
#
#     def computeDistanceWithNoise(self, max_arc_dist, noise, randomGen):
#         """
#         Method that computes the distance of the solution and implements noise
#         """
#         self.noise_succesful = 1
#         self.no_noise_succesful = 1
#         self.normal_distance = 0
#         self.normal_demand = 0
#         for route in self.routes:
#             self.normal_distance += route.distance
#             self.normal_demand += route.demand
#         maxN = noise * max_arc_dist
#         random_noise = randomGen.uniform(-maxN, maxN)
#         self.noise_distance = max(0, self.distance + random_noise)
#         self.noise_demand = max(0, self.demand + random_noise)
#         summation = self.noise_succesful + self.no_noise_succesful
#         rand_number = randomGen.random()
#         if rand_number < self.no_noise_succesful / summation:
#             self.distance = self.normal_distance
#             self.demand = self.normal_demand
#             self.distanceType = 0  # No noise is used in the objective solution
#         else:
#             self.distance = self.noise_distance
#             self.demand = self.noise_demand
#             self.distanceType = 1  # Noise is used in the objective solution
#         return self.distance, self.demand
#
#     # en uzun rotayı bul
#     def calculateMaxArc(self):
#         max_arc_length = 0
#         for route in self.routes:  # tüm routeları al
#             for i in range(1, len(route.locations)):
#                 first_node_ID = route.locations[i - 1].nodeID
#                 second_node_ID = route.locations[i].nodeID
#                 arc_length = distMatrix[first_node_ID][second_node_ID]
#                 if arc_length > max_arc_length:
#                     max_arc_length = arc_length
#         return max_arc_length
#
#     def print(self):
#         """
#         Method that prints the solution
#         """
#         nRoutes = len(self.routes)
#         nNotServed = len(self.notServed)
#         nServed = len(self.served)
#         print('total distance ' + str(self.distance) + " Solution with " + str(nRoutes) + " routes and " + str(
#             nNotServed) + " unserved requests and " + str(nServed) + " served request: ")
#         for route in self.routes:
#             route.calculateServiceStartTime()
#             route.print()
#
#         print("\n\n")
#
#     def printWithVehicle(self):
#         """
#         Method that prints the solution
#         """
#         nRoutes = len(self.routes)
#         nNotServed = len(self.notServed)
#         nServed = len(self.served)
#         print('total cost ' + str(self.distance) + " Solution with " + str(nRoutes) + " routes and " + str(
#             nNotServed) + " unserved requests and " + str(nServed) + " served request: \n")
#
#         for vehicles in self.problem.vehicles:
#             vehicles.print()
#
#
#     def removeRequest(self, request):
#         """
#         Method that removes a request from the solution
#         """
#         # iterate over routes to find in which route the request is served
#         for route in self.routes:
#             if request in route.requests:
#                 # remove the request from the route and break from loop
#                 route.removeRequest(request)
#                 break
#         self.served.remove(request)
#         self.notServed.append(request)
#         self.computeDistance()
#
#     def manage_routes(self, req):
#         if len(self.routes) < len(self.problem.vehicles):
#             # Yeni rota oluşturabiliriz
#             self.createNewRoute(req)
#         else:
#             # Mevcut rotalara ekleme yapmaya çalış
#             return self.try_add_to_existing_routes(req)
#
#     def try_add_to_existing_routes(self, req):
#         feasibleInsert = None
#         removedRoute = None
#         minC = sys.maxsize
#         minD = 0
#         for route in self.routes:
#             bestInsert, minCost, minDemand = route.greedyInsert(req, False)
#             if bestInsert != None:
#                 if minCost < minC:
#                     feasibleInsert = bestInsert
#                     removedRoute = route
#                     minC = minCost
#                     minD = minDemand
#
#         if feasibleInsert != None:
#             self.updateRoute(removedRoute, feasibleInsert, minC, minD)
#         else:
#             self.removeRequest(req)
#
#     def createNewRoute(self, req):
#         locList = [self.problem.depot, req.pickUpLoc, req.deliveryLoc, self.problem.depot]
#         newRoute = Route(locList, {req}, self.problem)
#         self.routes.append(newRoute)
#         self.distance += newRoute.distance
#         self.demand += newRoute.demand
#
#     def updateRoute(self, removedRoute, bestInsert, minCost, minDemand):
#         self.routes.remove(removedRoute)
#         self.routes.append(bestInsert)
#         self.distance += minCost
#         self.demand += minDemand
#
#     def copy(self):
#         """
#         Method that creates a copy of the solution and returns it
#         """
#         # need a deep copy of routes because routes are modifiable
#         routesCopy = list()
#         for route in self.routes:
#             routesCopy.append(route.copy())
#         copy = Solution(self.problem, routesCopy, self.served.copy(), self.notServed.copy())
#         copy.computeDistance()
#         return copy
#
#     def initialInsert(self, problem, solution, randomGen):
#         repairSol = Repair(problem, solution)
#         repairSol.executeGreedyInsertion()
#
#     def setVehicle(self, vehicles):
#         vehicleID = 0
#         for route in self.routes:
#             trolley_count_needed = route.calculateTrolley()
#
#             # Aracın kapasitesi ve trolley sayısını kontrol et
#             vehicles[vehicleID].routes.append({
#                 "trolleyCount": trolley_count_needed,
#                 "distance": route.distance,
#                 "demand": route.demand,
#                 "locations": route.locations,
#                 "route": route
#             })
#
#             # Araç toplamlarını güncelle
#             vehicles[vehicleID].totalDistance += route.distance
#             vehicles[vehicleID].totalDemand += route.demand
#
#             vehicleID += 1
#             if vehicleID >= len(vehicles):
#                 vehicleID = 0
# class PDPTW:
#
#     """
#     Class that represents a pick-up and delivery problem with time windows
#     Attributes
#     ----------
#     name : string
#         name of the instance.
#     requests : List of Requests
#         The set containing all requests.
#     depot : Location
#         the depot where all vehicles must start and end.
#     locations : Set of Locations
#         The set containing all locations
#      distMatrix : 2D array
#          matrix with all distances between station
#     capacity : int
#         capacity of the vehicles
#
#     """
#     def __init__(self, name, requests, depot, vehicles, TIR):
#         global distMatrix
#         global alpha
#         global beta
#         self.name = name
#         self.requests = requests
#         self.depot = depot
#         self.TIR = TIR
#         self.TValue = sys.maxsize
#         self.TValueMin = 0
#         self.bestDistanceProblem = 1
#         self.globalIterationNumber = 0
#         self.convProblem = 1
#         self.vehicles = vehicles
#         # for vehicle in vehicles:
#         #     self.capacity += vehicle.maxTrolleyCount * vehicle.trolleyCapacity
#         # self.capacity = self.capacity / len(self.vehicles)
#         ##construct the set with all locations
#         self.locations = set()
#         self.locations.add(depot)
#         for r in self.requests:
#             self.locations.add(r.pickUpLoc)
#             self.locations.add(r.deliveryLoc)
#
#         # compute the distance matrix
#         distMatrix = np.zeros((len(self.locations), len(self.locations)))  # init as nxn matrix
#         for i in self.locations:
#             for j in self.locations:
#                 distItoJ = Location.getDistance(i, j)
#                 distMatrix[i.nodeID, j.nodeID] = distItoJ
#     def print(self):
#         print(" MCVRPPDPTW problem " + self.name + " with " + str(
#             len(self.requests)) + " requests and a vehicle capacity of " + str(self.capacity))
#         for i in self.requests:
#             print(i)
# class ALNS:
#     """
#     Class that models the ALNS algorithm.
#
#     Parameters
#     ----------
#     problem : PDPTW
#         The problem instance that we want to solve.
#     nDestroyOps : int
#         number of destroy operators.
#     nRepairOps : int
#         number of repair operators.
#     randomGen : Random
#         random number generator
#     currentSolution : Solution
#         The current solution in the ALNS algorithm
#     bestSolution : Solution
#         The best solution currently found
#     bestDistance : int
#         Distance of the best solution
#     """
#
#     def __init__(self, problem, nDestroyOps, nRepairOps, nIterations, minSizeNBH, maxPercentageNHB, decayParameter,
#                  noise):
#         self.bestDemand = -1
#         self.bestDistance = -1
#         self.bestSolution = None
#         self.problem = problem
#         self.nDestroyOps = nDestroyOps
#         self.nRepairOps = nRepairOps
#         self.randomGen = random.Random(1234)  # used for reproducibility
#
#         self.wDestroy = [1 for i in range(nDestroyOps)]  # weights of the destroy operators
#         self.wRepair = [1 for i in range(nRepairOps)]  # weights of the repair operators
#         self.destroyUseTimes = [0 for i in range(nDestroyOps)]  # The number of times the destroy operator has been used
#         self.repairUseTimes = [0 for i in range(nRepairOps)]  # The number of times the repair operator has been used
#         self.destroyScore = [1 for i in range(nDestroyOps)]  # the score of destroy operators
#         self.repairScore = [1 for i in range(nRepairOps)]  # the score of repair operators
#
#         # Parameters for tuning
#         self.nIterations = nIterations
#         self.minSizeNBH = minSizeNBH
#         self.maxPercentageNHB = maxPercentageNHB
#         self.decayParameter = decayParameter
#         self.noise = noise
#
#         self.time_best_objective_found = 0
#         self.optimal_iteration_number = 0
#         # Presenting results
#         self.register_weights_over_time = False
#         self.removal_weights_per_iteration = []
#         self.insertion_weights_per_iteration = []
#         self.insertion_weights_per_iteration = []
#
#         self.register_objective_value_over_time = False
#         self.list_objective_values = []
#         self.list_objective_values_demand = []
#         self.destroyList = {
#             0: "Random request removal",
#         }
#         self.repairList = {
#             0: "Greedy Insert",
#             1: "Random Insert"
#         }
#     def constructInitialSolution(self):
#         """
#         Method that constructs an initial solution using random insertion
#         """
#         self.currentSolution = Solution(self.problem, list(), list(), list(self.problem.requests.copy()))
#         self.currentSolution.initialInsert(self.currentSolution.problem, self.currentSolution, self.randomGen)
#         self.currentSolution.computeDistance()
#         self.bestSolution = self.currentSolution.copy()
#         self.bestDistance = self.currentSolution.distance
#         self.bestDemand = self.currentSolution.demand
#         self.problem.TValue = self.bestDistance + (self.bestDistance * 0.4)
#         self.problem.TValueMin = (self.bestDistance - (self.problem.TValue - self.bestDistance)) // len(self.problem.vehicles)
#         self.problem.bestDistanceProblem = self.problem.TValueMin
#
#         print(f"Rule Of temp Solution Value : {self.bestDistance}")
#
#         # Print initial solution
#         # self.maxSizeNBH = len(self.problem.requests)
#         number_of_request = len(self.problem.requests)
#         self.maxSizeNBH = number_of_request
#
#     def execute(self):
#         """
#         Method that executes the ALNS
#         """
#         starttime = time.time()  # get the start time
#         self.starttime_best_objective = time.time()
#         self.real_dist = np.inf
#         self.real_demand = np.inf
#
#         self.constructInitialSolution()
#
#         for i in range(self.nIterations):  # Iteration count
#             self.max_arc_length = self.currentSolution.calculateMaxArc()
#             # Simulated annealing
#             self.iteration_number = i
#             self.problem.globalIterationNumber = i
#             self.checkIfAcceptNewSol()
#             # Print solution per iteration
#             objective_value = self.tempSolution.distance
#             objective_value_demand = self.tempSolution.demand
#
#             # To plot weights of the operators over time
#             if self.register_weights_over_time:
#                 self.removal_weights_per_iteration.append(self.wDestroyPlot)
#                 self.insertion_weights_per_iteration.append(self.wDestroyPlot)
#
#             # To plot objective values over time
#             if self.register_objective_value_over_time:
#                 self.list_objective_values.append(objective_value)
#                 self.list_objective_values_demand.append(objective_value_demand)
#
#
#         # set vehicle in route
#         self.bestSolution.setVehicle(self.problem.vehicles)
#         endtime = time.time()  # get the end time
#
#
#         self.bestSolution.printWithVehicle()
#     def checkIfAcceptNewSol(self):
#         """
#         Method that checks if we accept the newly found solution
#         """
#         # Copy the current solution
#         self.tempSolution = self.currentSolution.copy()
#         # decide on the size of the neighbourhood
#         sizeNBH = self.randomGen.randint(self.minSizeNBH, self.maxSizeNBH)
#         destroyOpNr = self.determineDestroyOpNr()  # çeşitlilik sağlanmak istenirse 9 a çıkar
#         repairOpNr = self.determineRepairOpNr()  # çeşitlilik sağlanmak istenirse yorum satırından kaldır
#
#         self.destroyAndRepair(destroyOpNr, repairOpNr, sizeNBH)
#
#         self.tempSolution.computeDistanceWithNoise(self.max_arc_length, self.noise, self.randomGen)
#
#
#         if self.tempSolution.distance < self.currentSolution.distance:
#             if self.tempSolution.distanceType == 0:
#                 self.tempSolution.no_noise_succesful += 1
#             elif self.tempSolution.distanceType == 1:
#                 self.tempSolution.noise_succesful += 1
#             self.currentSolution = self.tempSolution.copy()
#             # we found a global best solution
#             if self.tempSolution.distance < self.bestDistance:  # update best solution
#                 self.bestDistance = self.tempSolution.distance
#                 self.bestDemand = self.tempSolution.demand
#                 self.bestSolution = self.tempSolution.copy()
#                 new_real_dist, new_real_demand = self.tempSolution.computeDistance()
#                 if new_real_dist < self.real_dist:
#                     self.real_dist = new_real_dist
#                     self.problem.bestDistanceProblem = new_real_dist
#                     self.real_demand = new_real_demand
#                     print(f'New best global solution found: distance :{self.real_dist}, demand : {self.real_demand}, iteration : {self.iteration_number}')
#                     print(f'Destroy operator: {self.destroyList[destroyOpNr]}, Repair Operator : {self.repairList[repairOpNr]}')
#                     self.time_best_objective_found = time.time()
#                     self.optimal_iteration_number = self.iteration_number
#         else:
#             if self.randomGen.random() < np.exp(
#                     - (self.tempSolution.distance - self.currentSolution.distance) / 1):
#                 self.currentSolution = self.tempSolution.copy()
#
#         # Update the ALNS weights
#         self.updateWeights(destroyOpNr, repairOpNr)
#
#     def updateWeights(self, destroyOpNr, repairOpNr):
#         """
#         Method that updates the weights of the destroy and repair operators
#         """
#         self.destroyUseTimes[destroyOpNr] += 1
#         self.repairUseTimes[repairOpNr] += 1
#
#         self.wDestroy[destroyOpNr] = self.wDestroy[destroyOpNr] * (1 - self.decayParameter) + self.decayParameter * (
#                 self.destroyScore[destroyOpNr] / self.destroyUseTimes[destroyOpNr])
#
#         self.wDestroyPlot = self.wDestroy.copy()
#
#         self.wRepair[repairOpNr] = self.wRepair[repairOpNr] * (1 - self.decayParameter) + self.decayParameter * (
#                 self.repairScore[repairOpNr] / self.repairUseTimes[repairOpNr])
#         self.wRepairPlot = self.wDestroy.copy()
#
#     def determineDestroyOpNr(self):
#         """
#         Method that determines the destroy operator that will be applied.
#         Currently we just pick a random one with equal probabilities.
#         Could be extended with weights
#         """
#         destroyOperator = -1
#         destroyRoulette = np.array(self.wDestroy).cumsum()
#         r = self.randomGen.uniform(0, max(destroyRoulette))  # uniform distribution
#         for i in range(len(self.wDestroy)):
#             if destroyRoulette[i] >= r:
#                 destroyOperator = i
#                 break
#         return destroyOperator
#
#     def determineRepairOpNr(self):
#         """
#         Method that determines the repair operator that will be applied.
#         Currently we just pick a random one with equal probabilities.
#         Could be extended with weights
#         """
#         repairOperator = -1
#         repairRoulette = np.array(self.wRepair).cumsum()
#         r = self.randomGen.uniform(0, max(repairRoulette))
#         for i in range(len(self.wRepair)):
#             if repairRoulette[i] >= r:
#                 repairOperator = i
#                 break
#         return repairOperator
#
#     def destroyAndRepair(self, destroyHeuristicNr, repairHeuristicNr, sizeNBH):
#         """
#         Method that performs the destroy and repair. More destroy and/or
#         repair methods can be added
#
#         Parameters
#         ----------
#         destroyHeuristicNr : int
#             number of the destroy operator.
#         repairHeuristicNr : int
#             number of the repair operator.
#         sizeNBH : int
#             size of the neighborhood.
#         """
#         # destroySolution = Destroy(self.problem, self.tempSolution)
#         # if destroyHeuristicNr == 0:
#         #     destroySolution.executeWorstCostRemoval(sizeNBH)
#
#
#
#         # Perform the repair
#         repairSolution = Repair(self.problem, self.tempSolution)
#         if repairHeuristicNr == 0:
#             repairSolution.executeGreedyInsertion()
#         # elif repairHeuristicNr == 1:
#         #     repairSolution.executeRandomInsertion(self.randomGen)
#
#
# def read_instance(fileName) -> PDPTW:
#     global alpha
#     global beta
#     """
#     Method that reads an instance from a file and returns the instancesf
#     """
#     f = open(fileName)
#     requests = list()
#     unmatchedPickups = dict()
#     unmatchedDeliveries = dict()
#     nodeCount = 0
#     requestCount = 0  # start with 1
#     capacityOfTrolley = 50
#     TIR = 1.2
#
#     for infoLine in f.readlines()[-6:]:
#         if infoLine.startswith("VehicleCount"):
#             vehicleCount = int(infoLine[-3:-1].strip())
#         if infoLine.startswith("VehiclePerCapacity"):
#             capacityOfTrolley = int(infoLine[-6:-1].strip())
#         if infoLine.startswith("VehicleMaxTrolleyCount"):
#             Mmax = int(infoLine[-2:-1].strip())
#         if infoLine.startswith("TrolleyImpactRate"):
#             TIR = float(infoLine[-5:-1].strip())
#         if infoLine.startswith("EarlinessPenalty"):
#             alpha = float(infoLine[-3:-1].strip())
#         if infoLine.startswith("TardinessPenalty"):
#             beta = float(infoLine[-3:-1].strip())
#
#     f = open(fileName)
#     for line in f.readlines()[1:-7]:
#         asList = []
#         n = 13  # satırların sondan 13 karakteri boş o yüzden
#         for index in range(0, len(line), n):
#             asList.append(line[index: index + n].strip())
#
#         lID = asList[0]  # location tipi  D : depot, S: station, C : pickup / delivery point,
#         x = int(asList[2][:-2])  # need to remove ".0" from the string
#         y = int(asList[3][:-2])
#         lType = asList[1]
#         demand = int(asList[4][:-2])
#         startTW = int(asList[5][:-2])
#         endTW = int(asList[6][:-2])
#         servTime = int(asList[7][:-2])
#         partnerID = asList[8]
#         if lID.startswith("D"):  # depot ise
#             # it is the depot
#             depot = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "depot", nodeCount,
#                              lID)  # depot requestID=0
#             nodeCount += 1
#
#         elif lID.startswith("C"):  # pickup/delivery point ise
#             # it is a location
#
#             if lType == "cp":  # cp ise pickup, #cd ise delivery point
#                 if partnerID in unmatchedDeliveries:
#                     deliv = unmatchedDeliveries.pop(
#                         partnerID)  # pop listeden siler, sildiği değeri ise bir değişkene atar, burada deliv değişkenine atadı
#                     pickup = Location(deliv.requestID, x, y, demand, startTW, endTW, servTime, 0,
#                                       "pickup", nodeCount, lID)
#                     nodeCount += 1
#                     req = Request(pickup, deliv, deliv.requestID)
#                     requests.append(req)
#                 else:
#                     pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "pickup",
#                                       nodeCount, lID)
#                     nodeCount += 1
#                     requestCount += 1
#                     # lID -> partnerID
#                     unmatchedPickups[lID] = pickup
#             elif lType == "cd":  # cp ise pickup, #cd ise delivery point
#                 if partnerID in unmatchedPickups:
#                     pickup = unmatchedPickups.pop(partnerID)
#                     deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, 0,
#                                      "delivery", nodeCount, lID)
#                     nodeCount += 1
#                     req = Request(pickup, deliv, pickup.requestID)
#                     requests.append(req)
#                 else:
#                     deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0,
#                                      "delivery", nodeCount, lID)
#                     nodeCount += 1
#                     requestCount += 1
#                     unmatchedDeliveries[lID] = deliv
#
#     # Constraints 2
#     if len(unmatchedDeliveries) + len(unmatchedPickups) > 0:
#         raise Exception("Not all matched")
#
#     vehicles = list()
#     for i in range(vehicleCount):
#         vehicles.append(Vehicles(i, Mmax, capacityOfTrolley))
#
#     return PDPTW(fileName, requests, depot, vehicles, TIR)
#
# data = "Instances/lrc9.txt"
# problem = read_instance(data)
#
# # Static parameters
# nDestroyOps = 1  #number of destroy operations, çeşitlilik sağlanmak istenirse 9 a çıkar
# nRepairOps = 1  # number of repair operations # çeşitlilik sağlanmak istenirse 3 e çıkar
# minSizeNBH = 1  #Minimum size of neighborhood
# nIterations = 0  #Algoritma 100 kez tekrarlanacak(100 kez destroy ve rerair işlemlerini tekrarlayacak)
#
# # Parameters to tune:
# maxPercentageNHB = 1  #Maximum Percentage for Neighborhood
# decayParameter = 0.15
# noise = 0.015  #gürültü ekleme, çözüm uzayında daha çeşitli noktaları keşfetmeye yardımcı olur.
#
# alns = ALNS(problem, nDestroyOps, nRepairOps, nIterations, minSizeNBH, maxPercentageNHB, decayParameter, noise)
#
# alns.execute()
