import numpy as np

# Parameter Algoritma
num_ants = 2
num_iterations = 3
pheromone_evaporation_coefficient = 0.5
pheromone_constant = 1
visibility_constant = 2

# Jarak antar kota
distances = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Matriks feromon awal
pheromones = np.full(distances.shape, 0.1)

# Fungsi untuk menghitung visibilitas
def calculate_visibility(distances):
    visibility = np.zeros_like(distances, dtype=float)
    with np.errstate(divide='ignore'):
        visibility[distances > 0] = 1 / distances[distances > 0]
    return visibility

# Fungsi untuk menghitung probabilitas perpindahan
def calculate_transition_probabilities(pheromones, visibility, current_city, visited):
    probabilities = np.zeros_like(pheromones[current_city])
    for city in range(len(probabilities)):
        if city not in visited:
            probabilities[city] = (pheromones[current_city, city] ** pheromone_constant) * (visibility[current_city, city] ** visibility_constant)
    probabilities /= probabilities.sum()
    return probabilities

# Fungsi untuk memperbarui feromon
def update_pheromones(pheromones, all_routes, distances):
    pheromones *= (1 - pheromone_evaporation_coefficient)
    for route in all_routes:
        route_length = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
        for i in range(len(route) - 1):
            pheromones[route[i], route[i + 1]] += 1 / route_length
    return pheromones

# Algoritma ACO
visibility = calculate_visibility(distances)
for iteration in range(num_iterations):
    all_routes = []
    for ant in range(num_ants):
        route = [np.random.randint(0, len(distances))]
        while len(route) < len(distances):
            current_city = route[-1]
            probabilities = calculate_transition_probabilities(pheromones, visibility, current_city, route)
            next_city = np.random.choice(range(len(distances)), p=probabilities)
            route.append(next_city)
        route.append(route[0])  # kembali ke kota asal
        all_routes.append(route)
    pheromones = update_pheromones(pheromones, all_routes, distances)

# Menampilkan hasil
for i, route in enumerate(all_routes):
    print(f"Rute semut {i + 1}: {route} dengan panjang rute {sum(distances[route[j], route[j + 1]] for j in range(len(route) - 1))}")
