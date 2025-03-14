import numpy as np
import random
import matplotlib.pyplot as plt

def read_qaplib_instance(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    n = int(lines[0])  
    distance_matrix = np.zeros((n, n), dtype=int)
    flow_matrix = np.zeros((n, n), dtype=int)
    
    current_line = 1
    for i in range(n):
        distance_matrix[i] = np.array(list(map(int, lines[current_line].split())))
        current_line += 1
    for i in range(n):
        flow_matrix[i] = np.array(list(map(int, lines[current_line].split())))
        current_line += 1
    
    return distance_matrix, flow_matrix, n

# Load problem instance
distance_matrix, flow_matrix, num_facilities = read_qaplib_instance('els19.txt')

# Parameters
num_ants = 50
alpha = 5  # Pheromone influence
beta = 2   # Heuristic desirability influence
gamma = 0.7  # Pheromone evaporation rate
elitism_rank = 10  # Top-k ranked solutions influence pheromone update
iterations = 4000
q0 = 0.24  # Exploration vs. exploitation balance

tau_init = 1.0 / (num_facilities ** 2)
tau = np.full((num_facilities, num_facilities), tau_init, dtype=float)

def compute_cost(assignment):
    return np.sum(flow_matrix * distance_matrix[np.ix_(assignment, assignment)])

def construct_solution():
    unassigned_facilities = list(range(num_facilities))
    assignment = np.zeros(num_facilities, dtype=int)
    
    for i in range(num_facilities):
        if random.random() < q0:
            selected = max(unassigned_facilities, key=lambda j: tau[i, j])
        else:
            probabilities = np.array([
                (tau[i, j] ** alpha) * ((1 / (np.sum(distance_matrix[j]) + 1e-6)) ** beta)
                for j in unassigned_facilities
            ])
            probabilities += 1e-10  # Avoid division by zero
            probabilities /= probabilities.sum()  
            selected = np.random.choice(unassigned_facilities, p=probabilities)
        
        assignment[i] = selected
        unassigned_facilities.remove(selected)
    
    return assignment

def aco_facility_layout():
    global tau
    best_cost = float('inf')
    best_assignment = None
    best_fitness_progress = []
    avg_fitness_progress = []

    for iteration in range(iterations):
        solutions, costs = [], []
        for _ in range(num_ants):
            assignment = construct_solution()
            cost = compute_cost(assignment)
            solutions.append(assignment)
            costs.append(cost)
            if cost < best_cost:
                best_cost, best_assignment = cost, assignment
        
        ranked_solutions = sorted(zip(solutions, costs), key=lambda x: x[1])
        tau *= (1 - gamma)  # Pheromone evaporation
        
        # Rank-based elitist pheromone update
        for rank, (assignment, cost) in enumerate(ranked_solutions[:elitism_rank]):
            weight = (elitism_rank - rank) / elitism_rank
            for j in range(num_facilities):
                tau[j, assignment[j]] += weight / (cost + 1e-6)
        
        # Global best deposits the most pheromone
        for j in range(num_facilities):
            tau[j, best_assignment[j]] += 2 / (best_cost + 1e-6)
        
        best_fitness_progress.append(best_cost)
        avg_fitness_progress.append(np.mean(costs))
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Best Cost = {best_cost}")
    
    return best_assignment, best_cost, best_fitness_progress, avg_fitness_progress

def save_best_solution(filename, best_assignment, best_cost):
    with open(filename, 'w') as f:
        f.write(f"{num_facilities}  {best_cost}\n")
        f.write(" ".join(map(str, best_assignment + 1)))

best_solution, best_cost, best_fitness, avg_fitness = aco_facility_layout()
save_best_solution("aco_solution_final_try_1.txt", best_solution, best_cost)

plt.figure(figsize=(10, 5))
plt.plot(best_fitness, label='Best Fitness So Far', color='blue')
plt.plot(avg_fitness, label='Average Fitness So Far', color='red', linestyle='dashed')
plt.xlabel('Iterations')
plt.ylabel('Fitness (Cost)')
plt.title('ACO Optimization Progress')
plt.legend()
plt.show()

print("Best Cost Achieved:", best_cost)
print("Best Facility Assignment:", best_solution)
print("✅ Solution saved in `aco_solution.txt`")
