def nearest_neighbor_heuristic(points, start=0):
    """
    Nearest Neighbor Heuristic for the Traveling Salesman Problem.

    Parameters:
    - points: A list of tuples, where each tuple represents the coordinates (x, y) of each point.
    - start: Index of the starting point in the list.

    Returns:
    - A tuple containing the total distance of the path and the order of visited points.
    """
    import math

    # Calculate the distance between two points
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    visited = [start]
    current = start
    total_distance = 0

    while len(visited) < len(points):
        nearest = None
        nearest_distance = float('inf')

        for i, point in enumerate(points):
            if i not in visited:
                dist = distance(points[current], point)
                if dist < nearest_distance:
                    nearest = i
                    nearest_distance = dist

        visited.append(nearest)
        total_distance += nearest_distance
        current = nearest

    # Add distance back to the start point for a round trip
    total_distance += distance(points[current], points[start])
    visited.append(start)

    return total_distance, visited


# Example points ccordinate
points = [(0, 0), (1, 2), (2, 2), (3, 1), (0, 1)]

# Run the heuristic
i,j = nearest_neighbor_heuristic(points)
print(i,j)