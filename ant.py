class Ant:
    def __init__(self, start_city, num_cities):
        self.start_city = start_city
        self.num_cities = num_cities
        self.route = [start_city]
        self.distance = 0

    def visit_city(self, city, distance):
        self.route.append(city)
        self.distance += distance

    def complete_route(self, distance_matrix):
        self.visit_city(self.start_city, distance_matrix[self.route[-1]][self.start_city])

class AntColonyOptimization:
    def __init__(self, num_cities, distance_matrix, num_ants, evaporation_rate, alpha=1, beta=2, Q=100):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.pheromone_matrix = [[1 for _ in range(num_cities)] for _ in range(num_cities)]
        self.ants = [Ant(start_city, num_cities) for start_city in [0, 3, 6]]

    def run(self, iterations):
        for _ in range(iterations):
            self.construct_solutions()
            self.update_pheromones()

    def construct_solutions(self):
        for ant in self.ants:
            visited = set(ant.route)
            while len(ant.route) < self.num_cities:
                current_city = ant.route[-1]
                next_city = self.select_next_city(ant, current_city, visited)
                ant.visit_city(next_city, self.distance_matrix[current_city][next_city])
                visited.add(next_city)
            ant.complete_route(self.distance_matrix)

    def select_next_city(self, ant, current_city, visited):
        probabilities = []
        for city in range(self.num_cities):
            if city not in visited:
                pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
                visibility = (1 / self.distance_matrix[current_city][city]) ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)
        next_city = probabilities.index(max(probabilities))
        return next_city

    def update_pheromones(self):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.pheromone_matrix[i][j] *= (1 - self.evaporation_rate)
        for ant in self.ants:
            for i in range(len(ant.route) - 1):
                self.pheromone_matrix[ant.route[i]][ant.route[i + 1]] += self.Q / ant.distance
                self.pheromone_matrix[ant.route[i + 1]][ant.route[i]] += self.Q / ant.distance

distance_matrix = [
    [0, 4, 3, 7, 3, 6, 8],
    [4, 0, 6, 3, 4, 7, 5],
    [3, 6, 0, 5, 2, 3, 4],
    [7, 3, 5, 0, 6, 3, 4],
    [3, 4, 2, 6, 0, 5, 6],
    [6, 7, 3, 3, 5, 0, 2],
    [8, 5, 4, 4, 6, 2, 0]
]

aco = AntColonyOptimization(num_cities=7, distance_matrix=distance_matrix, num_ants=3, evaporation_rate=0.5, alpha=1, beta=2, Q=100)
aco.run(iterations=1)

for row in aco.pheromone_matrix:
    print(row)
