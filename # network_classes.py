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
        self.LagrangianMultiplier = 0
    
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
    def __init__(self, user_id, path, data):
        self.user_id = user_id
        self.path = path
        self.data = data
        self.rate = 0

class Network:
    def __init__(self):
        self.vertices = []
        self.links = []
        self.users = []
        self.create_network()
        self.create_users()
    
    def create_network(self):
        N = 6
        R = 10  # Radius of the circle
        for i in range(N):
            angle = 2 * np.pi * i / N
            x, y = R * np.cos(angle), R * np.sin(angle)
            vertex = Vertex(i+1, x, y)
            self.vertices.append(vertex)

        for i, vertex in enumerate(self.vertices):
            for j in range(i + 1, N):
                neighbor = self.vertices[j]
                distance = np.sqrt((vertex.x - neighbor.x)**2 + (vertex.y - neighbor.y)**2)
                if distance <= 10:  # Ensure nodes are within the connection radius
                    link = Link(f"{vertex.Vid}-{neighbor.Vid}", vertex, neighbor, distance, self)
                    self.links.append(link)
                    vertex.neighbors.append(neighbor.Vid)
                    neighbor.neighbors.append(vertex.Vid)

    def ensure_connectivity(self):
        # Check if all vertices are connected
        visited = set()
        self.dfs(self.vertices[0], visited)
        if len(visited) != len(self.vertices):
            raise Exception("The network is not fully connected. Adjust the connection radius.")

    def dfs(self, vertex, visited):
        visited.add(vertex)
        for neighbor_id in vertex.neighbors:
            neighbor = next(v for v in self.vertices if v.Vid == neighbor_id)
            if neighbor not in visited:
                self.dfs(neighbor, visited)

    def create_users(self):
        user1_path = [i.Vid for i in self.vertices]  # User 1 uses all links
        self.users.append(User(1, user1_path, 1000))  # Assign large data demand for user 1

        for i, vertex in enumerate(self.vertices):
            if i == 0:  # Skip user 1 as it's already created
                continue
            next_vertex = self.vertices[(i + 1) % len(self.vertices)]
            user_path = [vertex.Vid, next_vertex.Vid]
            self.users.append(User(i + 1, user_path, np.random.randint(50, 100)))  # Assign random data demand

    def simulate_num_problem(self, alpha, iterations, algorithm):
        for user in self.users:
            user.rate = 0.1  # Initial rate
        rates_over_time = {user.user_id: [] for user in self.users}
        for iteration in range(iterations):
            for user in self.users:
                if algorithm == 'Primal':
                    user.rate = self.calculate_primal_rate(user, alpha)
                elif algorithm == 'Dual':
                    user.rate = self.calculate_dual_rate(user, alpha)
                rates_over_time[user.user_id].append(user.rate)
            if iteration % 10000 == 0:
                print(f"Automatic update at iteration {iteration}")

        self.print_results(algorithm, alpha)
        self.plot_rate_results(rates_over_time, algorithm, alpha)

    def calculate_primal_rate(self, user, alpha, step_size=0.0001):
        if alpha == float('inf'):
            avg_rate = sum(u.rate for u in self.users) / len(self.users)
            return max(0, min(1, avg_rate + step_size)) if user.rate < avg_rate else max(0, user.rate - step_size)

        payment = 0
        for link in self.links:
            if link.vertex1.Vid in user.path and link.vertex2.Vid in user.path:
                rate_sum = sum(u.rate for u in self.users if link.vertex1.Vid in u.path and link.vertex2.Vid in u.path)
                payment += self.penalty_function(rate_sum, link.capacity)
        return step_size * (pow(user.rate, -alpha) - payment) + user.rate

    def penalty_function(self, rate_sum, capacity):
        if rate_sum < capacity:
            return rate_sum * capacity
        else:
            return pow(rate_sum, 3) * 2

    def calculate_dual_rate(self, user, alpha, step_size=0.0001):
        if alpha == float('inf'):
            max_excess = max((sum(u.rate for u in self.users if link.vertex1.Vid in u.path and link.vertex2.Vid in u.path) - link.capacity) for link in self.links if link.vertex1.Vid in user.path and link.vertex2.Vid in user.path)
            return max(0, min(1, user.rate - step_size * max_excess))

        Q_l = 0
        for link in self.links:
            if link.vertex1.Vid in user.path and link.vertex2.Vid in user.path:
                rate_sum = sum(u.rate for u in self.users if link.vertex1.Vid in u.path and link.vertex2.Vid in u.path)
                L_delta = (rate_sum - link.capacity) * step_size
                link.LagrangianMultiplier = max(0, link.LagrangianMultiplier + L_delta)
                Q_l += link.LagrangianMultiplier
        return pow(Q_l, -1 / alpha) if Q_l != 0 else 0

    def print_results(self, algorithm, alpha):
        total_rate = 0
        print(f"-------------------------\n{algorithm} Algorithm, alpha={alpha} Results:")
        for user in self.users:
            print(f"User {user.user_id}:\n  Path: {' -> '.join(map(str, user.path))}\n  Data: {user.data}\n  Rate: {user.rate:.2f}")
            total_rate += user.rate
        print(f"Sum Rate={total_rate:.2f}")

    def plot_rate_results(self, rates, algorithm, alpha):
        plt.figure()
        for user_id, user_rates in rates.items():
            plt.plot(user_rates, label=f'user {user_id}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Rate')
        plt.title(f'{algorithm} Algorithm, alpha={alpha}')
        plt.legend()
        plt.show()

    def plot_network(self):
        plt.figure()
        for vertex in self.vertices:
            plt.scatter(vertex.x, vertex.y, c='blue')
            plt.text(vertex.x, vertex.y, str(vertex.Vid), fontsize=12, ha='right')
        for link in self.links:
            plt.plot([link.vertex1.x, link.vertex2.x], [link.vertex1.y, link.vertex2.y], 'b-')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Network Topology')
        plt.show()

def menu():
    network = Network()
    network.plot_network()  # Plot the network topology
    while True:
        print("Choose a question to run the simulation:")
        print("2. NUM Problem using Primal Algorithm (TCP Reno)")
        print("3. NUM Problem using Dual Algorithm (TCP Vegas)")
        print("4. Generate Graphs for Different Alphas")
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
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
