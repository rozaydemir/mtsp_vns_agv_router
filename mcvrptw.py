import random

K = [1,2,3]
P_all = [1,2,3]
D_all = [4,5,6]

def generate_random_solution():
    # Generate a random solution by randomly assigning requests to vehicles
    solution = {}
    for k in K:
        solution[k] = random.sample(P_all + D_all, len(P_all + D_all))
    return solution

def evaluate_solution(solution):
    # Reset the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Set the variables according to the solution
    for k in K:
        for i, j in A[k]:
            if i in solution[k] and solution[k].index(i) + 1 < len(solution[k]) and solution[k][solution[k].index(i) + 1] == j:
                x[(i, j, k)].SetBounds(1, 1)  # If (i, j) is in the solution for vehicle k, set x[(i, j, k)] to 1
            else:
                x[(i, j, k)].SetBounds(0, 0)  # Otherwise, set x[(i, j, k)] to 0

    # Solve the problem with the new variables
    status = solver.Solve()

    # Return the objective value if the problem is solved optimally
    if status == pywraplp.Solver.OPTIMAL:
        return objective.Value()
    else:
        return None

# Run the heuristic test
random_solution = generate_random_solution()
objective_value = evaluate_solution(random_solution)
if objective_value is not None:
    print(f'The objective value of the random solution is {objective_value}')
else:
    print('The random solution is not feasible')