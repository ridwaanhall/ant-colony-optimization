import numpy as np
from typing import List, Tuple

class AntColonyOptimization:
    def __init__(self, distance_matrix: np.ndarray, num_ants: int, num_iterations: int, alpha: float, beta: float, evaporation_rate: float, pheromone_constant: float, random_seed: int = None):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.num_cities = distance_matrix.shape[0]
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        self.heuristic_matrix = 1 / (distance_matrix + np.eye(self.num_cities))
        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self) -> Tuple[List[int], float]:
        best_route = None
        best_length = float('inf')

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}")
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)

            for route in all_routes:
                length = self.route_length(route)
                if length < best_length:
                    best_length = length
                    best_route = route

            print(f"Best Route: {best_route}, Length: {best_length}\n")

        return best_route, best_length

    def construct_solutions(self) -> List[List[int]]:
        all_routes = []
        start_cities = [0, 3, 6]  # Semut 1 dari A (0), Semut 2 dari D (3), Semut 3 dari G (6)

        for ant in range(self.num_ants):
            route = [start_cities[ant]]
            while len(route) < self.num_cities:
                current_city = route[-1]
                probabilities = self.calculate_probabilities(current_city, route)
                next_city = np.random.choice(range(self.num_cities), p=probabilities)
                route.append(next_city)

            all_routes.append(route)
            print(f"Ant {ant + 1} Route: {route}")

        return all_routes

    def calculate_probabilities(self, current_city: int, route: List[int]) -> np.ndarray:
        probabilities = []
        for city in range(self.num_cities):
            if city not in route:
                pheromone = self.pheromone_matrix[current_city, city] ** self.alpha
                heuristic = self.heuristic_matrix[current_city, city] ** self.beta
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return probabilities

    def update_pheromones(self, all_routes: List[List[int]]):
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        for route in all_routes:
            for i in range(self.num_cities - 1):
                self.pheromone_matrix[route[i], route[i + 1]] += self.pheromone_constant / self.route_length(route)
            self.pheromone_matrix[route[-1], route[0]] += self.pheromone_constant / self.route_length(route)

        print(f"Pheromone Matrix:\n{self.pheromone_matrix}\n")

    def route_length(self, route: List[int]) -> float:
        return sum(self.distance_matrix[route[i], route[i + 1]] for i in range(self.num_cities - 1)) + self.distance_matrix[route[-1], route[0]]

# Parameter
distance_matrix = np.array([
    [0, 4, 3, 7, 3, 6, 8],
    [4, 0, 6, 3, 4, 7, 5],
    [3, 6, 0, 5, 2, 3, 4],
    [7, 3, 5, 0, 6, 3, 4],
    [3, 4, 2, 6, 0, 5, 6],
    [6, 7, 3, 3, 5, 0, 2],
    [8, 5, 4, 4, 6, 2, 0]
])
num_ants = 3
num_iterations = 1
alpha = 1
beta = 2
evaporation_rate = 0.5
pheromone_constant = 100
random_seed = 42  # Set seed for reproducibility

aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant, random_seed)
best_route, best_length = aco.run()

print(f"Final Best Route: {best_route}, Length: {best_length}")