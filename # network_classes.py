import numpy as np
import matplotlib.pyplot as plt
import heapq

class Vertex:
    def __init__(self, Vid, x, y):
        self.Vid = Vid
        self.x = x
        self.y = y
        self.neighbors = []

class Link:
    def __init__(self, link_id, vertex1, vertex2, distance, network):
        self.link_id = link_id
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.distance = distance
        self.network = network
        self.capacity = self.calculate_capacity()
    
    def calculate_capacity(self):
        sinr = self.calculate_sinr()
        return np.log2(1 + sinr)
    
    def calculate_sinr(self):
        signal_power = 1
        noise_power = 0.1
        interference_power = 0.1 * len(self.network.links)
        sinr = signal_power / (noise_power + interference_power)
        return sinr

class User:
    def __init__(self, user_id, path):
        self.user_id = user_id
        self.path = path
        self.rate = 0

class Network:
    def __init__(self):
        self.vertices = []
        self.links = []
        self.users = []
        self.create_network()
        self.create_users()
    
    def create_network(self):
        N = 5
        for i in range(N):
            x, y = np.random.random(), np.random.random()
            vertex = Vertex(i+1, x, y)
            self.vertices.append(vertex)
        for i, vertex in enumerate(self.vertices):
            for j in range(i+1, N):
                neighbor = self.vertices[j]
                distance = np.sqrt((vertex.x - neighbor.x)**2 + (vertex.y - neighbor.y)**2)
                link = Link(f"{vertex.Vid}-{neighbor.Vid}", vertex, neighbor, distance, self)
                self.links.append(link)
                vertex.neighbors.append(neighbor.Vid)
                neighbor.neighbors.append(vertex.Vid)

    def create_users(self):
        paths = [[1, 2, 3, 4, 5], [1, 3, 4, 5], [1, 4, 5], [2, 3, 5], [3, 4, 5]]
        for i, path in enumerate(paths):
            user = User(i+1, path)
            self.users.append(user)

    def simulate_num_problem(self, alpha, iterations, algorithm):
        for user in self.users:
            user.rate = 0.1  # Initial rate
        rates_over_time = {user.user_id: [] for user in self.users}
        for iteration in range(iterations):
            for user in self.users:
                user.rate = self.update_rate(user, alpha, algorithm)
                rates_over_time[user.user_id].append(user.rate)
            if iteration % 1000 == 0:
                print(f"Automatic update at iteration {iteration}")

        self.print_results(algorithm, alpha)
        self.plot_rate_results(rates_over_time, algorithm, alpha)

    def update_rate(self, user, alpha, algorithm):
        # Placeholder for actual rate update logic
        if algorithm == 'Primal':
            # TCP Reno style update: Incremental increase and multiplicative decrease
            user.rate += alpha * 0.01 * (1 - user.rate)  # Simplified
        elif algorithm == 'Dual':
            # TCP Vegas style update: Adjust based on expected vs actual throughput
            user.rate += alpha * 0.01 * (user.rate)  # Simplified
        return user.rate

    def print_results(self, algorithm, alpha):
        total_rate = 0
        print(f"-------------------------\n{algorithm} Algorithm, alpha={alpha} Results:")
        for user in self.users:
            print(f"user {user.user_id} rate : {user.rate:.2f}")
            total_rate += user.rate
        print(f"sum_rate={total_rate:.2f}")

    def plot_rate_results(self, rates, algorithm, alpha):
        plt.figure()
        for user_id, user_rates in rates.items():
            plt.plot(user_rates, label=f'user {user_id}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Rate')
        plt.title(f'{algorithm} Algorithm, alpha={alpha}')
        plt.legend()
        plt.show()

    def dijkstra_algorithm(self):
        start_vertex = self.vertices[0]
        distances = {vertex.Vid: float('inf') for vertex in self.vertices}
        distances[start_vertex.Vid] = 0
        priority_queue = [(0, start_vertex.Vid)]
        while priority_queue:
            current_distance, current_vertex_id = heapq.heappop(priority_queue)
            current_vertex = next(v for v in self.vertices if v.Vid == current_vertex_id)
            for neighbor_id in current_vertex.neighbors:
                neighbor_vertex = next(v for v in self.vertices if v.Vid == neighbor_id)
                link = next(l for l in self.links if {l.vertex1.Vid, l.vertex2.Vid} == {current_vertex_id, neighbor_id})
                distance = current_distance + link.distance
                if distance < distances[neighbor_vertex.Vid]:
                    distances[neighbor_vertex.Vid] = distance
                    heapq.heappush(priority_queue, (distance, neighbor_vertex.Vid))
        print("Dijkstra's algorithm results:", distances)

    def bellman_ford_algorithm(self):
        start_vertex = self.vertices[0]
        distances = {vertex.Vid: float('inf') for vertex in self.vertices}
        distances[start_vertex.Vid] = 0
        for _ in range(len(self.vertices) - 1):
            for link in self.links:
                if distances[link.vertex1.Vid] + link.distance < distances[link.vertex2.Vid]:
                    distances[link.vertex2.Vid] = distances[link.vertex1.Vid] + link.distance
                if distances[link.vertex2.Vid] + link.distance < distances[link.vertex1.Vid]:
                    distances[link.vertex1.Vid] = distances[link.vertex2.Vid] + link.distance
        print("Bellman-Ford algorithm results:", distances)

def menu():
    network = Network()
    while True:
        print("Choose a question to run the simulation:")
        print("2. NUM Problem using Primal Algorithm (TCP Reno)")
        print("3. NUM Problem using Dual Algorithm (TCP Vegas)")
        print("4. Generate Graphs for Different Alphas")
        print("5. Use Dijkstra's Algorithm for Network Paths")
        print("6. Use Bellman-Ford Algorithm for Network Paths")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == "2":
            alpha = 1  # Example value, should be configurable
            iterations = 100000  # Example value, should be configurable
            network.simulate_num_problem(alpha, iterations, algorithm='Primal')
        elif choice == "3":
            alpha = 1  # Example value, should be configurable
            iterations = 100000  # Example value, should be configurable
            network.simulate_num_problem(alpha, iterations, algorithm='Dual')
        elif choice == "4":
            alphas = [1, 2, float('inf')]
            for alpha in alphas:
                network.simulate_num_problem(alpha, 100000, algorithm='Primal')
                network.simulate_num_problem(alpha, 100000, algorithm='Dual')
        elif choice == "5":
            network.dijkstra_algorithm()
        elif choice == "6":
            network.bellman_ford_algorithm()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
