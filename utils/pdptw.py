import numpy as np
from request import Request
from location import Location
from vehicle import Vehicle

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

    def __init__(self, name, requests, depot, vehicleCapacity, vehicleCount):
        self.name = name
        self.requests = requests
        self.depot = depot
        self.capacity = vehicleCapacity
        self.vehicles = []
        for i in range(vehicleCount):
            self.vehicles.append(Vehicle(i, 0, 3,  50))
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
        print(" PDPTW problem " + self.name + " with " + str(
            len(self.requests)) + " requests and a vehicle capacity of " + str(self.capacity))
        print(self.distMatrix)
        for i in self.requests:
            print(i)

    def readInstance(fileName):
        """
        Method that reads an instance from a file and returns the instancesf
        """
        servStartTime = 0 # serviceTime
        f = open(fileName)
        requests = list()
        # stations = list()
        unmatchedPickups = dict()
        unmatchedDeliveries = dict()
        nodeCount = 0
        requestCount = 1  # start with 1
        for line in f.readlines()[1:-6]:
            asList = []
            n = 13 # satırların sondan 13 karakteri booş o yüzden
            for index in range(0, len(line), n):
                asList.append(line[index: index + n].strip())

            lID = asList[0] # location tipi  D : depot, S: station, C : pickup / delivery point,
            x = int(asList[2][:-2])  # need to remove ".0" from the string
            y = int(asList[3][:-2])
            if lID.startswith("D"): # depot ise
                # it is the depot
                depot = Location(0, x, y, 0, 0, 0, 0, servStartTime, "depot", nodeCount, "D0")  # depot requestID=0
                nodeCount += 1

            elif lID.startswith("C"): # pickup/delivery point ise
                # it is a location
                lType = asList[1]
                demand = int(asList[4][:-2])
                startTW = int(asList[5][:-2])
                endTW = int(asList[6][:-2])
                servTime = int(asList[7][:-2])
                partnerID = asList[8]
                if lType == "cp": # cp ise pickup, #cd ise delivery point
                    if partnerID in unmatchedDeliveries:
                        deliv = unmatchedDeliveries.pop(partnerID) # pop listeden siler, sildiği değeri ise bir değişkene atar, burada deliv değişkenine atadı
                        pickup = Location(deliv.requestID, x, y, demand, startTW, endTW, servTime, servStartTime, "pickup", nodeCount, lID)
                        nodeCount += 1
                        req = Request(pickup, deliv, deliv.requestID)
                        requests.append(req)
                    else:
                        pickup = Location(requestCount, x, y, demand, startTW, endTW, servTime, servStartTime, "pickup", nodeCount, lID)
                        nodeCount += 1
                        requestCount += 1
                        # lID -> partnerID
                        unmatchedPickups[lID] = pickup
                elif lType == "cd": # cp ise pickup, #cd ise delivery point
                    if partnerID in unmatchedPickups:
                        pickup = unmatchedPickups.pop(partnerID)
                        deliv = Location(pickup.requestID, x, y, demand, startTW, endTW, servTime, servStartTime, "delivery", nodeCount, lID)
                        nodeCount += 1
                        req = Request(pickup, deliv, pickup.requestID)
                        requests.append(req)
                    else:
                        deliv = Location(requestCount, x, y, demand, startTW, endTW, servTime, servStartTime, "delivery", nodeCount, lID)
                        nodeCount += 1
                        requestCount += 1
                        unmatchedDeliveries[lID] = deliv

        # Constraints 2
        if len(unmatchedDeliveries) + len(unmatchedPickups) > 0:
            raise Exception("Not all matched")
        # f.close()

        # read the vehicle capacity 
        f = open(fileName)
        capLine = f.readlines()[-4]
        perTrolleyCapacity = int(capLine[-7:-3].strip())  # her trolleyin alabileceği max capacity

        # ftwo = open(fileName)
        # vehicleCountLine = ftwo.readlines()[-1]
        vehicleCount = 3

        totalCapacity = perTrolleyCapacity * vehicleCount # total capacity   vehicleCount * perTrolleyCapacity

        return PDPTW(fileName, requests, depot, totalCapacity, vehicleCount)
