import random
import numpy as np


def calculate_total_distance(routes, distance_matrix):
    total_distance = 0
    for route in routes:
        route_distance = 0
        for i in range(len(route) - 1):
            route_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += route_distance + distance_matrix[route[-1]][route[0]]  # Return to start
    return total_distance


def two_opt_swap(route):
    new_route = route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j] = reversed(new_route[i:j])
    return new_route


def vns_for_mtsp(distance_matrix, num_salesmen, max_iterations):
    # Initialize random routes
    cities = list(range(len(distance_matrix))) # 4
    random.shuffle(cities)
    routes = [cities[i::num_salesmen] for i in range(num_salesmen)]


    best_distance = calculate_total_distance(routes, distance_matrix)
    best_routes = routes

    for iteration in range(max_iterations):
        for salesman in range(num_salesmen):
            new_route = two_opt_swap(routes[salesman])
            new_routes = routes.copy()
            new_routes[salesman] = new_route
            new_distance = calculate_total_distance(new_routes, distance_matrix)

            if new_distance < best_distance:
                best_distance = new_distance
                best_routes = new_routes
                break  # Exit the inner loop and restart shaking

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Distance: {best_distance}")

    return best_routes, best_distance


# np.random.seed(42) # Sonuçların tutarlı olması için
# distance_matrix = np.random.randint(1, 100, size=(5, 5))
#
# # Kendi kendine olan mesafeleri 0 yapma
# np.fill_diagonal(distance_matrix, 0)
distance_matrix = np.array(

        [
            [0, 30, 15, 72, 61, 75],
            [20, 0, 15, 72, 61, 75],
            [21, 40, 0, 75, 75, 80],
            [88, 24, 50, 0, 53, 85],
            [2, 88, 30, 50, 0, 90],
            [64, 60, 21, 33, 90, 0]
    ]
)
# distance_matrix = np.array(
#     [
#         [ 0, 93, 15, 72, 61, 21, 83, 87, 75, 75, 88, 24,  3, 22, 53,  2, 88, 30, 38,  2],
#         [64,  0, 21, 33, 76, 58, 22, 89, 49, 91, 59, 42, 92, 60, 80, 15, 62, 62, 47, 62],
#         [51, 55,  0,  3, 51,  7, 21, 73, 39, 18,  4, 89, 60, 14,  9, 90, 53,  2, 84, 92],
#         [60, 71, 44,  0, 47, 35, 78, 81, 36, 50,  4,  2,  6, 54,  4, 54, 93, 63, 18, 90],
#         [44, 34, 74, 62,  0, 95, 48, 15, 72, 78, 87, 62, 40, 85, 80, 82, 53, 24, 26, 89],
#         [60, 41, 29, 15, 45,  0, 89, 71,  9, 88,  1,  8, 88, 63, 11, 81,  8, 35, 35, 33],
#         [ 5, 41, 28,  7, 73, 72,  0, 34, 33, 48, 23, 62, 88, 37, 99, 44, 86, 91, 35, 65],
#         [99, 47, 78,  3,  1,  5, 90,  0, 27,  9, 79, 15, 90, 42, 77, 51, 63, 96, 52, 96],
#         [ 4, 94, 23, 15, 43, 29, 36, 13,  0, 71, 59, 86, 28, 66, 42, 45, 62, 57,  6, 28],
#         [28, 44, 84, 30, 62, 75, 92, 89, 62,  0,  1, 27, 62, 77,  3, 70, 72, 27,  9, 62],
#         [37, 97, 51, 44, 24, 79, 59, 32, 96, 88,  0, 62, 58, 52, 12, 39,  2,  3, 56, 81],
#         [59,  2,  2, 92, 54, 87, 96, 97,  1, 19,  2,  0, 44, 90, 32, 70, 32, 68, 55, 75],
#         [56, 17, 38, 24, 69, 98, 70, 86, 11, 16, 97, 73,  0, 70, 80, 93,  3, 20, 59, 36],
#         [19, 90, 67, 19, 20, 96, 71, 52, 33, 40, 39, 82,  1,  0, 92, 57, 89, 50, 23, 31],
#         [94, 42, 99,  7, 16, 90, 60,  2,  1, 48, 12, 69, 37, 32,  0, 99, 19, 48, 80,  3],
#         [20, 24, 54, 33, 24, 75, 72, 36, 38, 84, 99, 89, 99, 25, 93,  0, 82, 66, 54, 35],
#         [80, 61, 41, 33, 68, 33, 14, 21, 48, 20,  8,  7, 67, 17, 33, 48,  0, 59, 86, 22],
#         [30, 38, 51, 54,  8, 27, 27, 98, 21, 30, 97, 28, 64, 97, 69, 61, 48,  0,  4, 35],
#         [64, 49, 17, 44, 92, 30, 93, 46,  6, 99, 37, 24, 93, 46, 53, 95, 99, 60,  0, 63],
#         [85, 32, 87, 33, 67, 18, 25, 95, 54, 58, 67, 46, 24, 32, 47, 86, 23, 66, 27,  0]
#     ]
# )


print(distance_matrix)

num_salesmen = 2
max_iterations = 100

best_routes, best_distance = vns_for_mtsp(distance_matrix, num_salesmen, max_iterations)
print("Best Routes:", best_routes)
print("Best Distance:", best_distance)
