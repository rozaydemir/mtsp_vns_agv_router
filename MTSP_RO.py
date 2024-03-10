def modified_nearest_neighbor(vehicles, pickups, deliveries, depot, time_windows):
    """
    Modified Nearest Neighbor Algorithm for multiple vehicles with specific pickup and delivery points.

    Parameters:
    - vehicles: Number of vehicles available.
    - pickups: List of tuples representing pickup points (x, y).
    - deliveries: List of tuples representing delivery points (x, y), corresponding to pickups.
    - depot: Tuple representing the depot location (x, y).
    - time_windows: Dictionary with keys as point index (pickup or delivery) and values as (earliest, latest) time windows.

    Returns:
    - Dictionary with vehicle routes and total distance for each vehicle.
    """
    import math

    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Initialize routes for each vehicle
    routes = {v: {'route': [depot], 'total_distance': 0, 'current_time': 0} for v in range(vehicles)}

    # Create a list of all points with time windows
    all_points = pickups + deliveries
    points_time_windows = [time_windows[i] for i in range(len(pickups))] + [time_windows[i + len(pickups)] for i in
                                                                            range(len(deliveries))]

    # Function to find the nearest valid point considering time windows
    def find_nearest_valid_point(current_point, points, visited, current_time):
        nearest_point = None
        nearest_distance = float('inf')
        for i, point in enumerate(points):
            if i not in visited:
                dist = distance(current_point, point)
                # Check if the point can be reached within its time window
                if current_time + dist <= points_time_windows[i][1]:
                    if dist < nearest_distance:
                        nearest_point = i
                        nearest_distance = dist


        return nearest_point, nearest_distance

    for vehicle in range(vehicles):
        current_time = 0
        visited = set()
        while len(visited) < len(all_points):
            current_point = routes[vehicle]['route'][-1]
            nearest_point, nearest_dist = find_nearest_valid_point(current_point, all_points, visited, current_time)
            if nearest_point is None:  # No valid point found within time windows
                break  # Consider vehicle's route complete
            visited.add(nearest_point)
            # Update route and total distance
            routes[vehicle]['route'].append(all_points[nearest_point])
            routes[vehicle]['total_distance'] += nearest_dist
            # Update current time
            current_time += nearest_dist
            # Ensure delivery after pickup
            if nearest_point < len(pickups):  # If pickup, add corresponding delivery to visited
                visited.add(nearest_point + len(pickups))
            else:  # If delivery, ensure corresponding pickup is visited
                visited.add(nearest_point - len(pickups))

            # Check if we need to return to the depot
            if len(visited) == len(all_points):  # All points visited
                dist_to_depot = distance(current_point, depot)
                routes[vehicle]['route'].append(depot)
                routes[vehicle]['total_distance'] += dist_to_depot

        routes[vehicle]['current_time'] += current_time

    return routes


# Example usage
vehicles = 2 # Number of vehicles
pickups = [(0, 1), (2, 3), (3, 4)]  # Pickup points
deliveries = [(3, 5), (1, 0), (5, 6)]  # Delivery points corresponding to pickups
depot = (0, 0)  # Depot location
time_windows = {0: (0, 10), 1: (0, 10), 2: (5, 15), 3: (5, 15), 4: (10, 15), 5: (10,20) }  # Time windows for each point

# Run the modified algorithm
modified_routes = modified_nearest_neighbor(vehicles, pickups, deliveries, depot, time_windows)

print(modified_routes)