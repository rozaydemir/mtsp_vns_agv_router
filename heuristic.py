import random, time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from numpy import log as ln

prevNode = 0

class Vehicles:
    def __init__(self, id, Mmax, capacityOfTrolley):
        self.vehiclesId = id
        self.trolleyCapacity = capacityOfTrolley
        self.routes = []
        self.maxTrolleyCount = Mmax
        self.totalDistance = 0
        self.totalDemand = 0
        self.trolleyCount = 1

    def print(self):
        vehicleResult = dict()
        for route in self.routes:
            route["route"].calculateServiceStartTime()
            vehicleResult["trolleyCount"] = route["trolleyCount"]
            vehicleResult["vehicleId"] = self.vehiclesId + 1
            res, routeDetail = route["route"].print()
            vehicleResult["route"] = res
            vehicleResult["routeDetail"] = routeDetail
        return vehicleResult


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
        # global distMatrix
        # global beta
        # global alpha
        self.requestID = requestID
        self.xLoc = xLoc
        self.yLoc = yLoc
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

    def print(self):
        """
        Method that prints the location
        """
        # print(
        #     f" ( StringId: {self.stringId}, LocType: {self.typeLoc}, demand: {self.demand}; startTW: {self.startTW}; endTW: {self.endTW}; servTime:{self.servTime} ) ",
        #     end='')

    def printOnlyRoute(self, distMatrix, beta, alpha):
        """
        Method that prints the location
        """
        global prevNode
        dist = distMatrix[prevNode][self.nodeID]
        penalty = 0
        if self.typeLoc == "delivery":
            if self.servStartTime > self.endTW:
                penalty += (self.servStartTime - self.endTW) * beta

            if self.servStartTime < self.startTW:
                penalty += (self.startTW - self.servStartTime) * alpha
        detailResult = f" ( {self.stringId}, Demand : {self.demand}, CurrentTime: {self.servStartTime}, {self.typeLoc}, Distance: {dist}, Start: {self.startTW}, End: {self.endTW}, ServiceTime: {self.servTime}, Penalty: {penalty}, cumsum: {penalty + dist} ) "
        prevNode = self.nodeID
        return self.stringId, detailResult

    def getDistance(l1, l2):
        """
        Method that computes the rounded euclidian distance between two locations
        """
        dx = l1.xLoc - l2.xLoc
        dy = l1.yLoc - l2.yLoc
        return round(math.sqrt(dx ** 2 + dy ** 2))


class Destroy:
    '''
    Class that represents destroy methods

    Parameters
    ----------
    problem : PDPTW
        The problem instance that we want to solve.
    currentSolution : Solution
        The current solution in the ALNS algorithm
    randomGen : Random
        random number generator
    '''

    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution

    '''Helper function method 2'''

    def findWorstCostRequest(self):
        cost = []
        # global alpha
        # global beta
        # global distMatrix
        # Making list with request ID's and their corresponding cost
        for route in self.solution.routes:
            trolley_count_needed = route.calculateTrolley()
            curTime = 0
            for i in range(1, len(route.locations)):
                first_node = route.locations[i - 1]
                middle_node = route.locations[i]
                request_ID = route.locations[i].requestID
                dist = self.problem.distMatrix[first_node.nodeID][middle_node.nodeID]
                curTime = max(0, curTime + first_node.servTime + dist
                              + (trolley_count_needed * self.problem.TIR)
                              )

                ETPenalty = 0
                if middle_node.typeLoc == "delivery":
                    if curTime < middle_node.startTW:
                        ETPenalty += (middle_node.startTW - curTime) * self.problem.alpha

                    if curTime > middle_node.endTW:
                        ETPenalty += (curTime - middle_node.endTW) * self.problem.beta

                cost.append([request_ID, dist + ETPenalty])
        # Sort cost
        cost = sorted(cost, key=lambda d: d[1], reverse=True)
        chosen_request = None
        # Get request object that corresponds to worst cost
        worst_cost_request_ID = cost[0][0]
        for req in self.solution.served:
            if req.ID == worst_cost_request_ID:
                chosen_request = req
                break
        return chosen_request

    '''Helper function method 3'''

    def findWorstTimeRequest(self):
        service_time_difference = []
        # Making list with request ID's and difference between serivce start time and the start of time window
        for route in self.solution.routes:
            route.calculateServiceStartTime()
            for location in route.locations:
                if location.typeLoc != "depot":
                    difference = location.servStartTime - location.startTW
                    service_time_difference.append([location.requestID, difference])
        # Sort list with time differences
        service_time_difference = sorted(service_time_difference, key=lambda d: d[1], reverse=True)
        # Get request object that corresponds to worst cost
        if service_time_difference == []:
            return None

        worst_time_request_ID = service_time_difference[0][0]
        chosen_request = None
        for req in self.solution.served:
            if req.ID == worst_time_request_ID:
                chosen_request = req
                break
        return chosen_request

    '''Helper functions method 4 - 5'''

    def findStartingLocationShaw(self, randomGen):
        # Choose random route
        potentialRoutes = self.solution.routes
        location = None
        # Revise.
        while potentialRoutes:
            route = randomGen.choice(potentialRoutes)
            # Might exist the situation of [depot, depot]
            if len(route.locations) > 2:
                # From this route, choose a random location which is not the depot
                location = randomGen.choice(
                    [location for location in route.locations if location.typeLoc != "depot"])
                break
            else:
                potentialRoutes.remove(route)
        return location

    '''Helper function method 4'''

    def findNextTimeBasedRequest(self):
        # Initialize location that is currently selected
        loc_i = self.last_time_based_location
        smallest_diff = np.inf
        chosen_location = None
        # Find most related location in terms of start time window
        for route in self.solution.routes:
            route.calculateServiceStartTime()
            for loc_j in route.locations:
                # Only consider locations which are not depots
                tw_diff = abs(loc_i.startTW - loc_j.startTW)
                if tw_diff < smallest_diff:
                    chosen_location = loc_j
                    smallest_diff = tw_diff

        self.last_time_based_location = chosen_location
        chosen_request = None
        # Determine request that is related to the closest location
        for req in self.solution.served:
            if req.ID == self.last_time_based_location.requestID:
                chosen_request = req
                break
        return chosen_request

    '''Helper function method 5'''

    def findNextDemandBasedRequest(self):
        # Initialize location that is currently selected
        loc_i = self.last_demand_based_location
        smallest_diff = np.inf
        chosen_location = loc_i
        # Find most related location in terms of demand
        for route in self.solution.routes:
            for loc_j in route.locations:
                # Only consider locations which are not depots
                if loc_j.typeLoc != "depot":
                    demand_diff = abs(loc_i.demand - loc_j.demand)
                    if demand_diff < smallest_diff:
                        chosen_location = loc_j
                        smallest_diff = demand_diff
        self.last_demand_based_location = chosen_location
        chosen_request = None
        # Determine request that is related to the closest location
        for req in self.solution.served:
            if req.ID == self.last_demand_based_location.requestID:
                chosen_request = req
                break
        return chosen_request

    '''Helper function method 6'''

    def findRandomRoute(self, randomGen):
        # First make a copy of the routes in the solution
        set_of_routes = self.solution.routes.copy()
        # Find select a random route that contains at least one request
        while True:
            if not set_of_routes:
                return None
            route = randomGen.choice(set_of_routes)
            # If no request can be removed from this route
            if len(route.locations) < 4:
                set_of_routes.remove(route)
            else:
                return route

    def findWorstCostRequestRandomRoute(self, randomGen):
        # global alpha
        # global beta
        route = self.findRandomRoute(randomGen)
        cost = []
        trolley_count_needed = route.calculateTrolley()
        # Making list with request ID's and their corresponding cost
        curTime = 0
        for i in range(1, len(route.locations)):
            prevNode = route.locations[i - 1]
            curNode = route.locations[i]
            # request_ID = route.locations[i].requestID
            dist = self.problem.distMatrix[prevNode.nodeID][curNode.nodeID]
            curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.problem.TIR))

            ETPenalty = 0
            if curNode.typeLoc == "delivery":
                if curTime < curNode.startTW:
                    ETPenalty += (curNode.startTW - curTime) * self.problem.alpha

                if curTime > curNode.endTW:
                    ETPenalty += (curTime - curNode.endTW) * self.problem.beta

            cost.append([curNode.requestID, dist + ETPenalty])
        # Sort cost
        cost = sorted(cost, key=lambda d: d[1], reverse=True)
        # Get request object that corresponds to worst cost
        worst_cost_request_ID = cost[0][0]
        chosen_request = None
        for req in self.solution.served:
            if req.ID == worst_cost_request_ID:
                chosen_request = req
                break
        return chosen_request

    def findWorstNeighborhoodRequest(self):
        cost = []
        # Making list with request ID's and their corresponding cost
        for route in self.solution.routes:
            requests, distance = [], []
            total_dist = 0
            for i in range(2, len(route.locations)):
                first_node_ID = route.locations[i - 2].nodeID
                middle_note_ID = route.locations[i - 1].nodeID
                last_node_ID = route.locations[i].nodeID
                request_ID = route.locations[i - 1].requestID
                dist = self.problem.distMatrix[first_node_ID][middle_note_ID] + self.problem.distMatrix[middle_note_ID][
                    last_node_ID]
                requests.append(request_ID)
                distance.append(dist)
                total_dist += dist
            for index, request in enumerate(requests):
                cost.append([request, distance[index] / total_dist])
        # Sort cost
        cost = sorted(cost, key=lambda d: d[1], reverse=True)
        # print(cost)
        # Get request object that corresponds to worst cost
        worst_average_cost_request_ID = cost[0][0]
        chosen_request = None
        for req in self.solution.served:
            if req.ID == worst_average_cost_request_ID:
                chosen_request = req
                break
        return chosen_request

    '''Destroy method number 1'''

    def executeRandomRemoval(self, nRemove, randomGen):
        for i in range(nRemove):
            # terminate if no more requests are served
            if len(self.solution.served) == 0:
                break
            # pick a random request and remove it from the solution
            req = randomGen.choice(self.solution.served)
            if req != None:
                self.solution.removeRequest(req)

    '''Destroy method number 2'''

    def executeWorstCostRemoval(self, nRemove):
        for i in range(nRemove):
            if len(self.solution.served) == 0:
                break
            chosen_req = self.findWorstCostRequest()
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)

    '''Destroy method number 3'''

    def executeWorstTimeRemoval(self, nRemove):
        for i in range(nRemove):
            if len(self.solution.served) == 0:
                break

            chosen_req = self.findWorstTimeRequest()
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)

    '''Destroy method number 4'''

    def executeTimeBasedRemoval(self, nRemove, randomGen):
        if len(self.solution.served) == 0:
            return
        # Initialize starting location
        location = self.findStartingLocationShaw(randomGen)
        self.last_time_based_location = location

        # Select corresponding request
        request_ID = location.requestID
        for req in self.solution.served:
            if req.ID == request_ID:
                self.solution.removeRequest(req)
                break

        # Remove next requests based on relatedness in terms of start time windows between locations
        for _ in range(nRemove - 1):
            if len(self.solution.served) == 0:
                break
            chosen_req = self.findNextTimeBasedRequest()
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)

    '''Destroy method number 5'''

    def executeDemandBasedRemoval(self, nRemove, randomGen):
        if len(self.solution.served) == 0:
            return
        # Initialize starting location
        location = self.findStartingLocationShaw(randomGen)
        self.last_demand_based_location = location

        # Select corresponding request
        request_ID = location.requestID
        for req in self.solution.served:
            if req.ID == request_ID:
                self.solution.removeRequest(req)
                break

        # Remove next requests based on relatedness in terms of start time windows between locations
        for _ in range(nRemove - 1):
            if len(self.solution.served) == 0:
                break
            chosen_req = self.findNextDemandBasedRequest()
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)

    '''Destroy method number 6'''

    def executeRandomRouteRemoval(self, nRemove, randomGen):
        for _ in range(nRemove):
            if len(self.solution.served) == 0:
                break
            chosen_req = self.findWorstCostRequestRandomRoute(randomGen)
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)

    def executeWorstNeighborhoodRemoval(self, nRemove):
        for _ in range(nRemove):
            if len(self.solution.served) == 0:
                break
            chosen_req = self.findWorstNeighborhoodRequest()
            if chosen_req != None:
                self.solution.removeRequest(chosen_req)


class Route:
    """
    Class used to represent a route

    Parameters
    ----------
    locations : list of locations
        the route sequence of locations.
    requests : set of requests
        the requests served by the route
    problem : PDPTW
        the problem instance, used to compute distances.
    feasible : boolean
        true if route respects time windows, capacity and precedence # delivery after pickup
    distance : int
        total distance driven, extremely large number if infeasible
    """

    def __init__(self, locations, requests, problem):
        self.locations = locations
        self.requests = requests
        self.problem = problem
        # check the feasibility and compute the distance
        self.feasible = self.isFeasible()
        self.calculateServiceStartTime()
        if self.feasible:
            self.distance, self.demand = self.computeDistanceRoute()
        else:
            self.distance, self.demand = sys.maxsize, sys.maxsize  # extremely large number

    def calculateServiceStartTime(self):
        curTime = 0
        trolley_count_needed = self.calculateTrolley()

        for i in range(1, len(self.locations)):
            prevNode = self.locations[i - 1]
            curNode = self.locations[i]
            dist = self.problem.distMatrix[prevNode.nodeID][curNode.nodeID]
            curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.problem.TIR))
            self.locations[i].servStartTime = curTime

    def computeDistanceRoute(self) -> int:
        """
        Method that computes and returns the distance of the route
        """
        # global alpha
        # global beta
        totDist = 0
        totDemand = 0
        curTime = 0
        trolley_count_needed = self.calculateTrolley()
        for i in range(1, len(self.locations)):
            prevNode = self.locations[i - 1]
            curNode = self.locations[i]
            dist = self.problem.distMatrix[prevNode.nodeID][curNode.nodeID]
            curTime = max(0, curTime + prevNode.servTime + dist + (trolley_count_needed * self.problem.TIR))

            if curNode.typeLoc == 'pickup':
                totDemand += curNode.demand

            ETPenalty = 0
            if curNode.typeLoc == "delivery":
                if curTime < curNode.startTW:
                    ETPenalty += (curNode.startTW - curTime) * self.problem.alpha

                if curTime > curNode.endTW:
                    ETPenalty += (curTime - curNode.endTW) * self.problem.beta
            totDist += dist + ETPenalty

        return totDist, totDemand

    def computeDiff(self, preNode, afterNode, insertNode) -> int:
        '''
        Method that calculates the cost of inserting a new node
        Parameters
        ----------
        preNode: Location
        afterNode: Location
        insertNode: Location
        '''

        return self.problem.distMatrix[preNode.nodeID][insertNode.nodeID] + self.problem.distMatrix[afterNode.nodeID][
            insertNode.nodeID] - self.problem.distMatrix[preNode.nodeID][afterNode.nodeID]

    def computeTimeWindow(self, loc) -> int:
        # global alpha
        # global beta
        curTime = 0
        totalTimeWindowPenaly = 0
        trolley_count_needed = self.calculateTrolley()

        for i in range(1, len(loc)):
            prevNode = loc[i - 1]
            curNode = loc[i]
            dist = self.problem.distMatrix[prevNode.nodeID][curNode.nodeID]
            curTime = max(0, curTime + prevNode.servTime + dist
                          + (trolley_count_needed * self.problem.TIR)
                          )

            ETPenalty = 0
            if curNode.typeLoc == "delivery":
                if curTime < curNode.startTW:
                    ETPenalty += (curNode.startTW - curTime) * self.problem.alpha

                if curTime > curNode.endTW:
                    ETPenalty += (curTime - curNode.endTW) * self.problem.beta
            totalTimeWindowPenaly += ETPenalty
        return totalTimeWindowPenaly

    # add this method
    def compute_cost_add_one_request(self, preNode_index, afterNode_index, request) -> int:
        locationsCopy = self.locations.copy()
        locationsCopy.insert(preNode_index, request.pickUpLoc)
        # calculate the cost after inserting pickup location
        cost1 = self.computeDiff(locationsCopy[preNode_index - 1],
                                 locationsCopy[preNode_index + 1],
                                 request.pickUpLoc)

        locationsCopy.insert(afterNode_index, request.deliveryLoc)  # depot at the end
        # calculte the cost after inserting delivery location
        cost2 = self.computeDiff(locationsCopy[afterNode_index - 1],
                                 locationsCopy[afterNode_index + 1],
                                 request.deliveryLoc)

        cost3 = self.computeTimeWindow(locationsCopy)

        return cost2 + cost1 + cost3

    def print(self):
        """
        Method that prints the route
        """
        locationRes = list()
        locationDetailRes = list()
        for loc in self.locations:
            stringId, routeDetail = loc.printOnlyRoute(self.problem.distMatrix, self.problem.alpha, self.problem.beta)
            locationRes.append(stringId)
            locationDetailRes.append(routeDetail)
        return locationRes, locationDetailRes


    def calculateTrolley(self) -> int:
        trolley_count_needed = 1
        curDemand = 0
        for l in range(0, len(self.locations)):
            curNode = self.locations[l]
            curDemand += curNode.demand
            calculateTrolley = ((curDemand + self.problem.vehicles[0].trolleyCapacity - 1)
                                // self.problem.vehicles[0].trolleyCapacity)
            if calculateTrolley > trolley_count_needed:
                trolley_count_needed = calculateTrolley
        return trolley_count_needed

    def isFeasible(self) -> bool:
        """
        Method that checks feasbility. Returns True if feasible, else False
        """
        # route should start and end at the depot
        if self.locations[0] != self.problem.depot or self.locations[-1] != self.problem.depot:
            return False

        if len(self.locations) <= 2 and self.locations[0] == self.problem.depot and self.locations[
            1] == self.problem.depot:
            return False

        pickedUp = set()  # set with all requests that we picked up, used to check precedence
        trolley_count_needed = self.calculateTrolley()
        if trolley_count_needed > self.problem.vehicles[0].maxTrolleyCount:
            return False

        # iterate over route and check feasibility of time windows, capacity and precedence
        for i in range(1, len(self.locations) - 1):
            curNode = self.locations[i]

            if curNode.typeLoc == "pickup":
                # it is a pickup
                pickedUp.add(curNode.requestID)
            else:
                # it is a delivery
                # check if we picked up the request
                if curNode.requestID not in pickedUp:
                    return False
                pickedUp.remove(curNode.requestID)

        # finally, check if all pickups have been delivered
        if len(pickedUp) > 0:
            return False
        return True

    def removeRequest(self, request):
        """
        Method that removes a request from the route.
        """
        # remove the request, the pickup and the delivery
        self.requests.remove(request)
        self.locations.remove(request.pickUpLoc)
        self.locations.remove(request.deliveryLoc)
        # the distance changes, so update
        self.calculateServiceStartTime()
        self.computeDistanceRoute()
        # *** add this method

    def copy(self):
        """
        Method that returns a copy of the route
        """
        locationsCopy = self.locations.copy()
        requestsCopy = self.requests.copy()
        return Route(locationsCopy, requestsCopy, self.problem)

    def greedyInsert(self, request, convinent=True):
        """
        Method that inserts the pickup and delivery of a request at the positions
        that give the shortest total distance. Returns best route.

        Parameters
        ----------
        request : Request
            the request that should be inserted.

        Returns
        -------
        bestInsert : Route
            Route with the best insertion.

        """
        requestsCopy = self.requests.copy()
        requestsCopy.add(request)

        bestInsert = None  # if infeasible the bestInsert will be None
        minCost = sys.maxsize
        minDemand = sys.maxsize
        # iterate over all possible insertion positions for pickup and delivery

        for i in range(1, len(self.locations)):
            for j in range(i + 1, len(self.locations) + 1):  # delivery after pickup
                locationsCopy = self.locations.copy()
                locationsCopy.insert(i, request.pickUpLoc)
                locationsCopy.insert(j, request.deliveryLoc)  # depot at the end
                afterInsertion = Route(locationsCopy, requestsCopy, self.problem)
                # check if insertion is feasible
                if afterInsertion.feasible:
                    # check if cheapest
                    # revise. only calculate the cost
                    cost = self.compute_cost_add_one_request(i, j, request)
                    if cost < minCost:
                        bestInsert = afterInsertion
                        minCost = cost
                        minDemand = afterInsertion.demand

        if self.problem.globalIteration < 100 and convinent:
            convinent = False

        if convinent:
            if bestInsert != None:
                routeCost = bestInsert.distance
                if routeCost > self.problem.TValue:
                    self.problem.TValue = self.problem.TValue + (self.problem.TValue * 0.3)
                    bestInsert = None
                    minCost = sys.maxsize
                    minDemand = sys.maxsize
                else:
                    self.problem.TValue = self.problem.TValue - (self.problem.TValue * 0.1)
                    if self.problem.TValue < self.problem.TValueMin - (self.problem.TValueMin * 0.2):
                        self.problem.TValue = self.problem.TValueMin

        return bestInsert, minCost, minDemand


class Repair:
    '''
    Class that represents repair methods

    Parameters
    ----------
    problem : PDPTW
        The problem instance that we want to solve.
    currentSolution : Solution
        The current solution in the ALNS algorithm
    randomGen : Random
        random number generator

    '''

    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution

    def computeDiff(self, preNode, afterNode, insertNode) -> int:
        '''
        Method that calculates the cost of inserting a new node
        Parameters
        ----------
        preNode: Location
        afterNode: Location
        insertNode: Location
        '''

        return self.problem.distMatrix[preNode.nodeID][insertNode.nodeID] + self.problem.distMatrix[afterNode.nodeID][
            insertNode.nodeID] - self.problem.distMatrix[preNode.nodeID][afterNode.nodeID]

    def executeGreedyInsertion(self):
        """
        Method that greedily inserts the unserved requests in the solution

        This is repair method number 1 in the ALNS

        """

        while len(self.solution.notServed) > 0:
            for req in self.solution.notServed:
                inserted = False
                minCost = sys.maxsize  # initialize as extremely large number
                minDemand = sys.maxsize  # initialize as extremely large number
                bestInsert = None  # if infeasible the bestInsert will be None
                removedRoute = None  # if infeasible the bestInsert will be None
                for route in self.solution.routes:
                    afterInsertion, cost, demand = route.greedyInsert(req)

                    if afterInsertion == None:
                        continue

                    if cost < minCost:
                        inserted = True
                        removedRoute = route
                        bestInsert = afterInsertion
                        minDemand = demand
                        minCost = cost

                # if we were not able to insert, create a new route
                if not inserted:
                    self.solution.manage_routes(req)
                else:
                    self.solution.updateRoute(removedRoute, bestInsert, minCost, minDemand)

                # update the lists with served and notServed requests
                self.solution.served.append(req)
                self.solution.notServed.remove(req)

    def executeRandomInsertion(self, randomGen):
        """
        Method that randomly inserts the unserved requests in the solution

        This is repair method number 0 in the ALNS

        Parameters
        ----------
        randomGen : Random
            Used to generate random numbers

        """
        # iterate over the list with unserved requests
        while len(self.solution.notServed) > 0:
            # pick a random request
            req = randomGen.choice(self.solution.notServed)

            # keep track of routes in which req could be inserted
            potentialRoutes = self.solution.routes.copy()
            inserted = False
            while len(potentialRoutes) > 0:
                # pick a random route
                randomRoute = randomGen.choice(potentialRoutes)
                afterInsertion, cost, demand = randomRoute.greedyInsert(req)

                if afterInsertion == None:
                    # insertion not feasible, remove route from potential routes
                    potentialRoutes.remove(randomRoute)
                else:
                    # insertion feasible, update routes and break from while loop
                    inserted = True
                    self.solution.updateRoute(randomRoute, afterInsertion, cost, demand)
                    break

            # if we were not able to insert, create a new route
            if not inserted:
                self.solution.manage_routes(req)

            # update the lists with served and notServed requests
            self.solution.served.append(req)
            self.solution.notServed.remove(req)


class Parameters:
    randomSeed = 1234  # value of the random seed
    w1 = 1.5  # if the new solution is a new global best
    w2 = 1.2  # if the new solution is better than the current one
    w3 = 0.8  # if the new solution is accepted
    w4 = 0.6  # if the new solution is rejected


class Solution:
    """
    Method that represents a solution to the PDPTW

    Attributes
    ----------
    problem : PDPTW
        the problem that corresponds to this solution
    routes : List of Routes
         Routes in the current solution
    served : List of Requests
        Requests served in the current solution
    notServed : List of Requests
         Requests not served in the current solution
    distance : int
        total distance of the current solution
    """

    def __init__(self, problem, routes, served, notServed):
        self.problem = problem
        self.routes = routes
        self.served = served
        self.notServed = notServed
        self.distance, self.demand = self.computeDistance()

    def computeDistance(self):
        """
        Method that computes the distance of the solution
        """
        self.distance = 0
        self.demand = 0
        for route in self.routes:
            self.distance += route.distance
            self.demand += route.demand
        return self.distance, self.demand

    def computeDistanceWithNoise(self, max_arc_dist, noise, randomGen):
        """
        Method that computes the distance of the solution and implements noise
        """
        self.noise_succesful = 1
        self.no_noise_succesful = 1
        self.normal_distance = 0
        self.normal_demand = 0
        for route in self.routes:
            self.normal_distance += route.distance
            self.normal_demand += route.demand
        maxN = noise * max_arc_dist
        random_noise = randomGen.uniform(-maxN, maxN)
        self.noise_distance = max(0, self.distance + random_noise)
        self.noise_demand = max(0, self.demand + random_noise)
        summation = self.noise_succesful + self.no_noise_succesful
        rand_number = randomGen.random()
        if rand_number < self.no_noise_succesful / summation:
            self.distance = self.normal_distance
            self.demand = self.normal_demand
            self.distanceType = 0  # No noise is used in the objective solution
        else:
            self.distance = self.noise_distance
            self.demand = self.noise_demand
            self.distanceType = 1  # Noise is used in the objective solution
        return self.distance, self.demand

    # en uzun rotayı bul
    def calculateMaxArc(self):
        max_arc_length = 0
        for route in self.routes:  # tüm routeları al
            for i in range(1, len(route.locations)):
                first_node_ID = route.locations[i - 1].nodeID
                second_node_ID = route.locations[i].nodeID
                arc_length = self.problem.distMatrix[first_node_ID][second_node_ID]
                if arc_length > max_arc_length:
                    max_arc_length = arc_length
        return max_arc_length

    def print(self):
        """
        Method that prints the solution
        """
        nRoutes = len(self.routes)
        nNotServed = len(self.notServed)
        nServed = len(self.served)
        print('total distance ' + str(self.distance) + " Solution with " + str(nRoutes) + " routes and " + str(
            nNotServed) + " unserved requests and " + str(nServed) + " served request: ")
        for route in self.routes:
            route.calculateServiceStartTime()
            route.print()

        print("\n\n")

    def printWithVehicle(self):
        """
        Method that prints the solution
        """

        resultArray = list()
        for vehicles in self.problem.vehicles:
            resultArray.append(vehicles.print())
        return resultArray

    def removeRequest(self, request):
        """
        Method that removes a request from the solution
        """
        # iterate over routes to find in which route the request is served
        for route in self.routes:
            if request in route.requests:
                # remove the request from the route and break from loop
                route.removeRequest(request)
                break
        self.served.remove(request)
        self.notServed.append(request)

    def manage_routes(self, req):
        if len(self.routes) < len(self.problem.vehicles):
            # Yeni rota oluşturabiliriz
            self.createNewRoute(req)
        else:
            # Mevcut rotalara ekleme yapmaya çalış
            return self.try_add_to_existing_routes(req)

    def try_add_to_existing_routes(self, req):
        feasibleInsert = None
        removedRoute = None
        minC = sys.maxsize
        minD = 0
        for route in self.routes:
            bestInsert, minCost, minDemand = route.greedyInsert(req, False)
            if bestInsert != None:
                if minCost < minC:
                    feasibleInsert = bestInsert
                    removedRoute = route
                    minC = minCost
                    minD = minDemand

        if feasibleInsert != None:
            self.updateRoute(removedRoute, feasibleInsert, minC, minD)
        else:
            self.removeRequest(req)

    def createNewRoute(self, req):
        locList = [self.problem.depot, req.pickUpLoc, req.deliveryLoc, self.problem.depot]
        newRoute = Route(locList, {req}, self.problem)
        self.routes.append(newRoute)
        self.distance += newRoute.distance
        self.demand += newRoute.demand

    def updateRoute(self, removedRoute, bestInsert, minCost, minDemand):
        self.routes.remove(removedRoute)
        self.routes.append(bestInsert)
        self.distance += minCost
        self.demand += minDemand

    def copy(self):
        """
        Method that creates a copy of the solution and returns it
        """
        # need a deep copy of routes because routes are modifiable
        routesCopy = list()
        for route in self.routes:
            routesCopy.append(route.copy())
        copy = Solution(self.problem, routesCopy, self.served.copy(), self.notServed.copy())
        copy.computeDistance()
        return copy

    def initialInsert(self, problem, solution, randomGen):
        repairSol = Repair(problem, solution)
        repairSol.executeRandomInsertion(randomGen)

    def setVehicle(self, vehicles):
        vehicleID = 0
        for route in self.routes:
            trolley_count_needed = route.calculateTrolley()

            # Aracın kapasitesi ve trolley sayısını kontrol et
            vehicles[vehicleID].routes.append({
                "trolleyCount": trolley_count_needed,
                "distance": route.distance,
                "demand": route.demand,
                "locations": route.locations,
                "route": route
            })

            # Araç toplamlarını güncelle
            vehicles[vehicleID].totalDistance += route.distance
            vehicles[vehicleID].totalDemand += route.demand

            vehicleID += 1
            if vehicleID >= len(vehicles):
                vehicleID = 0


class PDPTW:
    """
    Class that represents a pick-up and delivery problem with time windows
    Attributes
    ----------
    name : string
        name of the instance.
    requests : List of Requests
        The set containing all requests.
    depot : Location
        the depot where all vehicles must start and end.
    locations : Set of Locations
        The set containing all locations
     distMatrix : 2D array
         matrix with all distances between station
    capacity : int
        capacity of the vehicles

    """

    def __init__(self, name, requests, depot, vehicles, TIR, EarlinessPenalty, TardinessPenalty):
        self.name = name
        self.requests = requests
        self.depot = depot
        self.TIR = TIR
        self.TValue = sys.maxsize
        self.TValueMin = 0
        self.bestDistanceProblem = 1
        self.globalIteration = 1
        self.convProblem = 1
        self.vehicles = vehicles
        self.alpha = EarlinessPenalty
        self.beta = TardinessPenalty
        ##construct the set with all locations
        self.locations = set()
        self.locations.add(depot)
        for r in self.requests:
            self.locations.add(r.pickUpLoc)
            self.locations.add(r.deliveryLoc)

        # compute the distance matrix
        self.distMatrix = np.zeros((len(self.locations), len(self.locations)))  # init as nxn matrix
        for i in self.locations:
            for j in self.locations:
                distItoJ = Location.getDistance(i, j)
                self.distMatrix[i.nodeID, j.nodeID] = distItoJ

    def print(self):
        print(" MCVRPPDPTW problem " + self.name + " with " + str(
            len(self.requests)) + " requests and a vehicle capacity of " + str(self.capacity))
        for i in self.requests:
            print(i)


class ALNS:
    """
    Class that models the ALNS algorithm.

    Parameters
    ----------
    problem : PDPTW
        The problem instance that we want to solve.
    nDestroyOps : int
        number of destroy operators.
    nRepairOps : int
        number of repair operators.
    randomGen : Random
        random number generator
    currentSolution : Solution
        The current solution in the ALNS algorithm
    bestSolution : Solution
        The best solution currently found
    bestDistance : int
        Distance of the best solution
    """

    def __init__(self, problem, nDestroyOps, nRepairOps, nIterations, minSizeNBH, maxPercentageNHB, decayParameter,
                 noise, convinentInterval):
        self.bestDemand = -1
        self.bestDistance = -1
        self.bestSolution = None
        self.problem = problem
        self.nDestroyOps = nDestroyOps
        self.nRepairOps = nRepairOps
        self.randomGen = random.Random(Parameters.randomSeed)  # used for reproducibility

        self.wDestroy = [1 for i in range(nDestroyOps)]  # weights of the destroy operators
        self.wRepair = [1 for i in range(nRepairOps)]  # weights of the repair operators
        self.destroyUseTimes = [0 for i in range(nDestroyOps)]  # The number of times the destroy operator has been used
        self.repairUseTimes = [0 for i in range(nRepairOps)]  # The number of times the repair operator has been used
        self.destroyScore = [1 for i in range(nDestroyOps)]  # the score of destroy operators
        self.repairScore = [1 for i in range(nRepairOps)]  # the score of repair operators
        self.convinentStarter = convinentInterval

        # Parameters for tuning
        self.nIterations = nIterations
        self.minSizeNBH = minSizeNBH
        self.maxPercentageNHB = maxPercentageNHB
        self.decayParameter = decayParameter
        self.noise = noise

        self.time_best_objective_found = 0
        self.optimal_iteration_number = 0
        # Presenting results
        self.register_weights_over_time = False
        self.removal_weights_per_iteration = []
        self.insertion_weights_per_iteration = []
        self.insertion_weights_per_iteration = []

        self.register_objective_value_over_time = False
        self.list_objective_values = []
        self.list_objective_values_demand = []
        self.destroyList = {
            0: "Random request removal",
            1: "Worst-cost",
            2: "Worst-time",
            3: "Time based",
            4: "Demand Based",
            5: "Random Route removal",
            6: "Worst Neightbourhood removal",
        }
        self.repairList = {
            0: "Greedy Insert",
            1: "Random Insert"
        }

    def constructInitialSolution(self):
        """
        Method that constructs an initial solution using random insertion
        """
        self.currentSolution = Solution(self.problem, list(), list(), list(self.problem.requests.copy()))
        self.currentSolution.initialInsert(self.currentSolution.problem, self.currentSolution, self.randomGen)
        self.currentSolution.computeDistance()
        self.bestSolution = self.currentSolution.copy()
        self.bestDistance = self.currentSolution.distance
        self.bestDemand = self.currentSolution.demand
        self.problem.TValue = self.bestDistance + (self.bestDistance * 0.3)
        self.problem.TValueMin = (self.convinentStarter * .5) // len(self.problem.vehicles)
        self.problem.bestDistanceProblem = self.problem.TValueMin
        self.problem.convProblem = self.problem.TValueMin
        # print(f"convergent interval : {self.problem.TValueMin} - {self.problem.TValue}")
        # print(f"Initial Solution Value : {self.bestDistance}")

        # Print initial solution
        # self.maxSizeNBH = len(self.problem.requests)
        # number_of_request = len(self.problem.requests)
        self.maxSizeNBH = len(self.problem.requests)

    def execute(self):
        """
        Method that executes the ALNS
        """
        starttime = time.time()  # get the start time
        self.starttime_best_objective = time.time()
        self.real_dist = np.inf
        self.real_demand = np.inf

        self.constructInitialSolution()

        for i in range(self.nIterations):  # Iteration count
            self.max_arc_length = self.currentSolution.calculateMaxArc()
            # Simulated annealing
            self.iteration_number = i
            self.problem.globalIteration = i
            self.checkIfAcceptNewSol()
            # Print solution per iteration
            objective_value = self.tempSolution.distance
            objective_value_demand = self.tempSolution.demand

            # To plot weights of the operators over time
            if self.register_weights_over_time:
                self.removal_weights_per_iteration.append(self.wDestroyPlot)
                self.insertion_weights_per_iteration.append(self.wDestroyPlot)

            # To plot objective values over time
            if self.register_objective_value_over_time:
                self.list_objective_values.append(objective_value)
                self.list_objective_values_demand.append(objective_value_demand)

        # set vehicle in route
        self.bestSolution.setVehicle(self.problem.vehicles)
        endtime = time.time()  # get the end time
        cpuTime = round(endtime - starttime, 3)

        print("ALNS - Final cost: " + str(self.bestSolution.distance) + ", cpuTime: " + str(
            cpuTime) + " seconds")

        time_best_objective = self.time_best_objective_found - self.starttime_best_objective

        # print(
        #     f'Best objective value found after: {round(time_best_objective, 3) if self.optimal_iteration_number > 0 else 0} seconds and {self.optimal_iteration_number} iterations')

        solutionResult = self.bestSolution.printWithVehicle()

        return self.bestSolution.distance, cpuTime, solutionResult


    def checkIfAcceptNewSol(self):
        """
        Method that checks if we accept the newly found solution
        """
        # Copy the current solution
        self.tempSolution = self.currentSolution.copy()
        # decide on the size of the neighbourhood
        sizeNBH = self.randomGen.randint(self.minSizeNBH, self.maxSizeNBH)
        destroyOpNr = self.determineDestroyOpNr()  # çeşitlilik sağlanmak istenirse 9 a çıkar
        repairOpNr = self.determineRepairOpNr()  # çeşitlilik sağlanmak istenirse yorum satırından kaldır

        self.destroyAndRepair(destroyOpNr, repairOpNr, sizeNBH)

        self.tempSolution.computeDistanceWithNoise(self.max_arc_length, self.noise, self.randomGen)

        if self.tempSolution.distance < self.currentSolution.distance:
            if self.tempSolution.distanceType == 0:
                self.tempSolution.no_noise_succesful += 1
            elif self.tempSolution.distanceType == 1:
                self.tempSolution.noise_succesful += 1
            self.currentSolution = self.tempSolution.copy()
            # we found a global best solution
            if self.tempSolution.distance < self.bestDistance:  # update best solution
                self.bestDistance = self.tempSolution.distance
                self.bestDemand = self.tempSolution.demand
                self.bestSolution = self.tempSolution.copy()
                self.destroyScore[destroyOpNr] += Parameters.w1
                self.repairScore[repairOpNr] += Parameters.w1  # the new solution is a new global best, 1.5
                new_real_dist, new_real_demand = self.tempSolution.computeDistance()
                if new_real_dist < self.real_dist:
                    self.problem.bestDistanceProblem = new_real_dist
                    self.real_dist = new_real_dist
                    self.real_demand = new_real_demand
                    # print(
                    #     f'New best global solution found: distance :{self.real_dist}, demand : {self.real_demand}, iteration : {self.iteration_number}')
                    # print(
                    #     f'Destroy operator: {self.destroyList[destroyOpNr]}, Repair Operator : {self.repairList[repairOpNr]}')
                    self.time_best_objective_found = time.time()
                    self.optimal_iteration_number = self.iteration_number
            else:
                self.destroyScore[destroyOpNr] += Parameters.w2
                self.repairScore[repairOpNr] += Parameters.w2  # the new solution is better than the current one 1.2
        else:
            if self.randomGen.random() < np.exp(
                    - (self.tempSolution.distance - self.currentSolution.distance) / 1):
                self.currentSolution = self.tempSolution.copy()
                self.destroyScore[destroyOpNr] += Parameters.w3  # the new solution is accepted 0.8
                self.repairScore[repairOpNr] += Parameters.w3
            else:
                self.destroyScore[destroyOpNr] += Parameters.w4  # the new solution is rejected 0.6
                self.repairScore[repairOpNr] += Parameters.w4

        # Update the ALNS weights
        self.updateWeights(destroyOpNr, repairOpNr)

    def updateWeights(self, destroyOpNr, repairOpNr):
        """
        Method that updates the weights of the destroy and repair operators
        """
        self.destroyUseTimes[destroyOpNr] += 1
        self.repairUseTimes[repairOpNr] += 1

        self.wDestroy[destroyOpNr] = self.wDestroy[destroyOpNr] * (1 - self.decayParameter) + self.decayParameter * (
                self.destroyScore[destroyOpNr] / self.destroyUseTimes[destroyOpNr])

        self.wDestroyPlot = self.wDestroy.copy()

        self.wRepair[repairOpNr] = self.wRepair[repairOpNr] * (1 - self.decayParameter) + self.decayParameter * (
                self.repairScore[repairOpNr] / self.repairUseTimes[repairOpNr])
        self.wRepairPlot = self.wDestroy.copy()

    def determineDestroyOpNr(self):
        """
        Method that determines the destroy operator that will be applied.
        Currently we just pick a random one with equal probabilities.
        Could be extended with weights
        """
        destroyOperator = -1
        destroyRoulette = np.array(self.wDestroy).cumsum()
        r = self.randomGen.uniform(0, max(destroyRoulette))  # uniform distribution
        for i in range(len(self.wDestroy)):
            if destroyRoulette[i] >= r:
                destroyOperator = i
                break
        return destroyOperator

    def determineRepairOpNr(self):
        """
        Method that determines the repair operator that will be applied.
        Currently we just pick a random one with equal probabilities.
        Could be extended with weights
        """
        repairOperator = -1
        repairRoulette = np.array(self.wRepair).cumsum()
        r = self.randomGen.uniform(0, max(repairRoulette))
        for i in range(len(self.wRepair)):
            if repairRoulette[i] >= r:
                repairOperator = i
                break
        return repairOperator

    def destroyAndRepair(self, destroyHeuristicNr, repairHeuristicNr, sizeNBH):
        """
        Method that performs the destroy and repair. More destroy and/or
        repair methods can be added

        Parameters
        ----------
        destroyHeuristicNr : int
            number of the destroy operator.
        repairHeuristicNr : int
            number of the repair operator.
        sizeNBH : int
            size of the neighborhood.
        """
        destroySolution = Destroy(self.problem, self.tempSolution)
        if destroyHeuristicNr == 0:
            destroySolution.executeRandomRemoval(sizeNBH, self.randomGen)
        elif destroyHeuristicNr == 1:
            destroySolution.executeWorstCostRemoval(sizeNBH)
        elif destroyHeuristicNr == 2:
            destroySolution.executeWorstTimeRemoval(sizeNBH)
        elif destroyHeuristicNr == 3:
            destroySolution.executeTimeBasedRemoval(sizeNBH, self.randomGen)
        elif destroyHeuristicNr == 4:
            destroySolution.executeDemandBasedRemoval(sizeNBH, self.randomGen)
        elif destroyHeuristicNr == 5:
            destroySolution.executeRandomRouteRemoval(sizeNBH, self.randomGen)
        elif destroyHeuristicNr == 6:
            destroySolution.executeWorstNeighborhoodRemoval(sizeNBH)

        # Perform the repair
        repairSolution = Repair(self.problem, self.tempSolution)
        if repairHeuristicNr == 0:
            repairSolution.executeGreedyInsertion()
        elif repairHeuristicNr == 1:
            repairSolution.executeRandomInsertion(self.randomGen)


def read_instance(fileName, vehicleCount,
                  VehiclePerCapacity,
                  VehicleMaxTrolleyCount,
                  TrolleyImpactRate,
                  EarlinessPenalty,
                  TardinessPenalty) -> PDPTW:
    # global alpha
    # global beta
    """
    Method that reads an instance from a file and returns the instancesf
    """
    f = open(fileName)
    requests = list()
    unmatchedPickups = dict()
    unmatchedDeliveries = dict()
    nodeCount = 0
    requestCount = 0  # start with 1
    capacityOfTrolley = 50
    TIR = 1.2


    # for infoLine in f.readlines()[-6:]:
    #     if infoLine.startswith("VehicleCount"):
    #         vehicleCount = int(infoLine[-3:-1].strip())
    #     if infoLine.startswith("VehiclePerCapacity"):
    #         capacityOfTrolley = int(infoLine[-6:-1].strip())
    #     if infoLine.startswith("VehicleMaxTrolleyCount"):
    #         Mmax = int(infoLine[-2:-1].strip())
    #     if infoLine.startswith("TrolleyImpactRate"):
    #         TIR = float(infoLine[-4:-1].strip())
    #     if infoLine.startswith("EarlinessPenalty"):
    #         alpha = float(infoLine[-3:-1].strip())
    #     if infoLine.startswith("TardinessPenalty"):
    #         beta = float(infoLine[-3:-1].strip())

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
                    req = Request(pickup, deliv, deliv.requestID)
                    requests.append(req)
                else:
                    pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0, "pickup",
                                      nodeCount, lID)
                    nodeCount += 1
                    requestCount += 1
                    # lID -> partnerID
                    unmatchedPickups[lID] = pickup
            elif lType == "cd":  # cp ise pickup, #cd ise delivery point
                if partnerID in unmatchedPickups:
                    pickup = unmatchedPickups.pop(partnerID)
                    deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, 0,
                                     "delivery", nodeCount, lID)
                    nodeCount += 1
                    req = Request(pickup, deliv, pickup.requestID)
                    requests.append(req)
                else:
                    deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, 0,
                                     "delivery", nodeCount, lID)
                    nodeCount += 1
                    requestCount += 1
                    unmatchedDeliveries[lID] = deliv

    # Constraints 2
    if len(unmatchedDeliveries) + len(unmatchedPickups) > 0:
        raise Exception("Not all matched")

    vehicles = list()
    for i in range(vehicleCount):
        vehicles.append(Vehicles(i, VehicleMaxTrolleyCount, VehiclePerCapacity))

    return PDPTW(fileName, requests, depot, vehicles, TrolleyImpactRate, EarlinessPenalty, TardinessPenalty)


class Heuristic:
    def __init__(self, fileName, vehicleCount, VehiclePerCapacity, VehicleMaxTrolleyCount, TrolleyImpactRate, EarlinessPenalty, TardinessPenalty, iterationCount, convinentInterval):
        self.fileName = fileName
        self.nDestroyOps = 7
        self.nRepairOps = 2
        self.minSizeNBH = 1
        self.nIterations = iterationCount
        self.maxPercentageNHB = 5000
        self.decayParameter = 0.15
        self.noise = 0.015

        self.vehicleCount = vehicleCount
        self.VehiclePerCapacity = VehiclePerCapacity
        self.VehicleMaxTrolleyCount = VehicleMaxTrolleyCount
        self.TrolleyImpactRate = TrolleyImpactRate
        self.EarlinessPenalty = EarlinessPenalty
        self.TardinessPenalty = TardinessPenalty
        self.convinentInterval = convinentInterval

    def execute(self):
        global prevNode
        problem = read_instance(self.fileName,
                                self.vehicleCount,
                                self.VehiclePerCapacity,
                                self.VehicleMaxTrolleyCount,
                                self.TrolleyImpactRate,
                                self.EarlinessPenalty,
                                self.TardinessPenalty)

        alns = ALNS(problem, self.nDestroyOps, self.nRepairOps, self.nIterations, self.minSizeNBH,
                    self.maxPercentageNHB, self.decayParameter,
                    self.noise, self.convinentInterval)
        prevNode = 0
        bestDistance, cpuTime, solutionResult = alns.execute()
        return round(bestDistance), cpuTime, solutionResult
