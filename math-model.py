from ortools.linear_solver import pywraplp
import math

solver = pywraplp.Solver.CreateSolver('SCIP')

K = []  # vehicles
P_all = []  # total pickup nodes +
D_all = []  # total delivery nodes +
P = {}  # pickup nodes per vehicle +
D = {}  # delivery nodes +
N = {}  # service points for each vehicle +
l = {}  # order load + demand
o = {}  # depot node +
d = {}  # destination node +
A = {}

# velocity = 1 , dist = time
t = {}  # travel time + öklid

a = {}  # early start times
b = {}  # late start times
s = {}  # service time of nodes

class Location:
    def __init__(self, xLoc, yLoc, typeLoc, nodeID):
        self.xLoc = xLoc
        self.yLoc = yLoc
        self.typeLoc = typeLoc
        self.nodeID = nodeID
    def getDistance(l1, l2):
        """
        Method that computes the rounded euclidian distance between two locations
        """
        dx = l1.xLoc - l2.xLoc
        dy = l1.yLoc - l2.yLoc
        return round(math.sqrt(dx ** 2 + dy ** 2))


def read_data(fileName, vehicleCount):
    global K
    global A
    global N
    global P
    global D
    global d
    global o
    global P_all
    global D_all
    global t
    global l
    global a
    global b
    global s

    f = open(fileName)
    pickupNodes = 0
    deliveryNode = 0
    nodeCount = 0
    locations = []

    for v in range(vehicleCount):
        K.append(v + 1)

    for line in f.readlines()[1:-6]:
        asList = []
        n = 13  # satırların sondan 13 karakteri boş o yüzden
        for index in range(0, len(line), n):
            asList.append(line[index: index + n].strip())

        lID = asList[0]  # location tipi  D : depot, S: station, C : pickup / delivery point,
        x = int(asList[2][:-2])  # need to remove ".0" from the string
        y = int(asList[3][:-2])
        demand = int(asList[4][:-2])
        startTW = int(asList[5][:-2])
        endTW = int(asList[6][:-2])
        servTime = int(asList[7][:-2])

        l[nodeCount] = abs(demand)
        a[nodeCount] = startTW
        b[nodeCount] = endTW
        s[nodeCount] = servTime

        if lID.startswith("D"):
            locations.append(Location(x, y, "depot", 0))
            nodeCount += 1

        if lID.startswith("C"):  # pickup/delivery point ise
            # it is a location

            lType = asList[1]

            if lType == "cp":  # cp ise pickup, #cd ise delivery point
                locations.append(Location(x, y, "pickup", nodeCount))
                pickupNodes += 1
            elif lType == "cd":  # cp ise pickup, #cd ise delivery point
                locations.append(Location(x, y, "delivery", nodeCount))
                deliveryNode += 1
            nodeCount += 1



    for p in range(1, pickupNodes + 1):
        P_all.append(p)

    for dn in range(0, deliveryNode):
        pnNumber = pickupNodes + 1
        D_all.append(dn + pnNumber)

    locations.append(Location(locations[0].xLoc, locations[0].yLoc, "depot", len(P_all + D_all) + 1))

    for v in range(1, vehicleCount + 1):
        P[v] = P_all
        D[v] = D_all
        N[v] = P_all + D_all
        d[v] = len(N[v]) + 1
        o[v] = 0
        arcs = []
        for n1 in [0] + N[v] + [d[v]]:
            for n2 in [0] + N[v] + [d[v]]:
                if n1 != n2 and n1 != d[v]:
                    arcs.append((n1, n2))
        A[v] = arcs

    for v in range(1, vehicleCount + 1):
        for loc1 in locations:
            for loc2 in locations:
                if loc1.nodeID != loc2.nodeID:
                    calculateEuclid = Location.getDistance(loc1, loc2)
                    t[(loc1.nodeID, loc2.nodeID, v)] = calculateEuclid

    l[len(P_all + D_all) + 1] = l[0]
    a[len(P_all + D_all) + 1] = a[0]
    b[len(P_all + D_all) + 1] = b[0]
    s[len(P_all + D_all) + 1] = s[0]

#   "Instances/lrc5.txt"
#   "Instances/lrc5-demand-increase.txt"
#   "Instances/lrc5-location-increase.txt"
#   "Instances/lrc7.txt"
#   "Instances/lrc7-demand-increase.txt"
#   "Instances/lrc7-location-increase.txt"
#   "Instances/lrc9.txt"
#   "Instances/lrc9-demand-increase.txt"
#   "Instances/lrc9-location-increase.txt"
#   "Instances/lrc11.txt"
#   "Instances/lrc11-demand-increase.txt"

data = "Instances/lrc11.txt" # datayı yükle
read_data(data, 2)


alpha = 15  # Cost of penalty for early delivery
beta = 90  # Penalty for one unit of tardiness

C = 200  # Capacity of a trolley
Mmax = 3  # Maximum Trolley Addition Capacity
TIR = 0.2  # Trolley Impact Rate (TIR): The rate at which the addition of trolleys impacts operations
M = 10000  # Big M

x = {}  # xijk equal to 1 if arc (i, j) ∈ Ak is used by vehicle k, and 0 otherwise (binary flow variables)
T = {}  # Tik specifying when vehicle k starts the service at node i ∈  Vk
L = {}  # Lik giving the load of vehicle k after the service at node i ∈  Vk has been completed.
E = {}  # earliness of delivery arrived workstation i
TA = {}  # tardiness of delivery arrived workstation i
Y = {}  # number of trolley attached to AGV k
for k in K:
    # Y[k] = solver.IntVar(1, Mmax, f'Y[{k}]')  # constraint 19 is ensured.
    for i, j in A[k]:
        if i != j:
            x[(i, j, k)] = solver.BoolVar(f'x[{i},{j},{k}]')
    for i in range(max(max(A[k])) + 1):  # A listesindeki en büyük düğüm numarasına göre döngü
        T[(i, k)] = solver.NumVar(0, solver.infinity(), f'T[{i},{k}]')
        L[(i, k)] = solver.NumVar(0, solver.infinity(), f'L[{i},{k}]')
for i in D_all:
    E[i] = solver.NumVar(0, solver.infinity(), f'E[{i}]')
    TA[i] = solver.NumVar(0, solver.infinity(), f'TA[{i}]')


# Objective Function
objective = solver.Objective()
for k in K:
    for i, j in A[k]:
        objective.SetCoefficient(x[(i, j, k)], t.get((i, j, k), 0))
for i in D_all:
    objective.SetCoefficient(E[i], alpha)
    objective.SetCoefficient(TA[i], beta)
objective.SetMinimization()

# Constaint 22. The travel time for AGV k from point i to point j (t_ijk) increases by the Trolley Impact Rate (TIR) for each added trolley.
# This shows how additional trolleys affect the travel time of the AGV.
# for k in K:
#     for i, j in A[k]:
#         if (i,j,k) in t:
#             T[(i, k)]= T[(i, k)] + (Y[k]*TIR)

# Constraints 2, 3 impose that each request (i.e., the pickup and delivery nodes) is served exactly once and by the same vehicle
# 2
for i in P_all:
    solver.Add(solver.Sum(x[(i, j, k)] for k in K for j in N[k] + [d[k]] if (i, j, k) in x) == 1)

# 3
for k in K:
    for i in P[k]:
        solver.Add(
            solver.Sum(x[(i, j, k)] for j in N[k] if (i, j, k) in x) -
            solver.Sum(x[(j, len(P[k]) + i, k)] for j in N[k] if (j, len(P[k]) + i, k) in x) == 0
        )

# Constrainst 4 ensure that each vehicle k starts from its origin depot o(k)
for k in K:
    solver.Add(solver.Sum(x[(o[k], j, k)] for j in P[k] + [d[k]]) == 1)

# Constrainst 5 characterize a multicommodity flow structure
for k in K:
    for j in N[k]:
        solver.Add(
            solver.Sum(x[(i, j, k)] for i in N[k] + [o[k]] if i != j) -
            solver.Sum(x[(j, i, k)] for i in N[k] + [d[k]] if i != j) == 0
        )

# Constrainst 6 ensures that each vehicle k terminates its route at its destination depot d(k)
for k in K:
    solver.Add(solver.Sum(x[(i, d[k], k)] for i in D[k] + [o[k]]) == 1)

# constaint 7 & 9 earlines tardiness
# for k in K:
#     for i in D[k]:
#         solver.Add(E[i] >= a[i] - T[(i, k)])
#         solver.Add(TA[i] >= T[(i, k)] - b[i])

    # Constrainst 11 Time windows
for k in K:
    for i, j in A[k]:
        if (i, j, k) in x:
            solver.Add(
                T[(i, k)] + (s[i] + t[(i, j, k)]) * x[(i, j, k)]  <= T[(j, k)] + M * (1 - x[(i, j, k)])
            )

# constraints 12 force the vehicle to visit the pickup node before the delivery node
# j=i+len(P[k]) because of denoting the set of pickup nodes by P = {1,...,n} and the set of delivery nodes by D = {n +1,..., 2n}
for k in K:
    for i in P[k]:
        if (i, i + len(P[k]), k) in x:
            solver.Add(T[(i, k)] + t[(i, i + len(P[k]), k)]  <= T[(i + len(P[k]), k)])

# Constraints 13a and 13b compatibility requirements between routes and vehicle loads
# 10a
for k in K:
    for i, j in A[k]:
        if (i, j, k) in x:
            solver.Add(L[(i, k)] + l[j] - L[(j, k)] <= M * (1 - x[(i, j, k)]))

# Constraint 13b
for k in K:
    for i, j in A[k]:
        if (i, j, k) in x:
            solver.Add(-L[(i, k)] - l[j] + L[(j, k)] <= M * (1 - x[(i, j, k)]))

# Constaint 14 ensure vehicle dependent capacity restrictions at pick-up points
for k in K:
    for i in P[k]:
        solver.Add(l[i] <= L[(i, k)])
        solver.Add(L[(i, k)] <= C)

# Constaint 15 ensure vehicle dependent capacity restrictions at delivery points
# n_i TANIMLI DEĞİLDİ TANIMLADIK SONUÇ DEĞİŞTİ EN SON ELLE YAPTIKTAN SONRA KONTROL ET
for k in K:
    for i in D[k]:  # D[k], k aracının teslimat noktaları
        n_i = i + len(P[k])
        if n_i in L:
            solver.Add(0 <= L[(n_i, k)])
            solver.Add(L[(n_i, k)] <= C - l[i])

# Constraint 16 ensure initial vehicle load is zero
for k in K:
    solver.Add(L[(o[k], k)] == 0)

# Constaint 19 The equation stating that each AGV requires at least one trolley to be able to
# carry a load and that the number of trolleys attached to AGV k (Y_k) can be at most M_max:
# for k in K:
#     solver.Add(Mmax >= Y[k])
#     solver.Add(Y[k] >= 1)

# Çözümü hesapla ve sonuçları yazdır
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', objective.Value())
    for k in K:
        for i, j in A[k]:
            if x[(i, j, k)].solution_value() > 0:
                print(
                    f'Road {(i, j)} vehiche {k} used. And Cost : {t[(i, j, k)]}')
else:
    print('No optimal solution was found.')