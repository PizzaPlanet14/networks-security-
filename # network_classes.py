import numpy as np
import matplotlib.pyplot as plt
import random

class Vertex:
    def __init__(self, Vid, x, y):
        self.Vid = Vid
        self.x = x
        self.y = y
        self.neighbors = []

class Link:
    def __init__(self, link_id, vertex1, vertex2, distance):
        self.link_id = link_id
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.distance = distance
        self.capacity = 1.0
        self.LagrangianMultiplier = 0

class User:
    def __init__(self, user_id, path, data):
        self.user_id = user_id
        self.path = path
        self.rate = 0.0001  # Initialize with a small positive value
        self.data = data

class Network:
    def __init__(self, R=10, r=3):
        self.vertices = []
        self.links = []
        self.users = []
        self.create_network(R, r)
        self.create_users()

    def create_network(self, R, r):
        N = 6
        for i in range(N):
            angle = 2 * np.pi * i / N
            x, y = R * np.cos(angle), R * np.sin(angle)
            vertex = Vertex(i + 1, x, y)
            self.vertices.append(vertex)

        while True:
            self.links = []
            for i, vertex in enumerate(self.vertices):
                for j in range(i + 1, N):
                    neighbor = self.vertices[j]
                    distance = np.sqrt((vertex.x - neighbor.x)**2 + (vertex.y - neighbor.y)**2)
                    if distance <= r:
                        link = Link(f"{vertex.Vid}-{neighbor.Vid}", vertex, neighbor, distance)
                        self.links.append(link)
                        vertex.neighbors.append(neighbor.Vid)
                        neighbor.neighbors.append(vertex.Vid)

            if self.ensure_connectivity():
                break
            else:
                r += 1  # Increment the radius until the network is connected

    def ensure_connectivity(self):
        visited = set()
        to_visit = [self.vertices[0]]

        while to_visit:
            current = to_visit.pop()
            if current.Vid not in visited:
                visited.add(current.Vid)
                to_visit.extend(v for v in self.vertices if v.Vid in current.neighbors and v.Vid not in visited)

        return len(visited) == len(self.vertices)

    def create_users(self):
        self.users = [
            User(1, [5, 4, 3, 2, 1, 6], 743),
            User(2, [4, 5], 761),
            User(3, [3, 4], 276),
            User(4, [2, 3], 334),
            User(5, [1, 2], 669),
            User(6, [6, 1], 89)
        ]

    def simulate_num_problem(self, alpha, iterations, algorithm, initial_step_size=0.0001):
        self.initialize_lagrangian_multipliers()
        rates_over_time = {user.user_id: [] for user in self.users}

        step_size = initial_step_size

        for iteration in range(iterations):
            for user in self.users:
                if algorithm == 'Primal':
                    user.rate = self.calculate_primal_rate(user, alpha, step_size)
                elif algorithm == 'Dual':
                    user.rate = self.calculate_dual_rate(user, alpha, step_size)
                rates_over_time[user.user_id].append(user.rate)

            # Adjust step size dynamically if needed
            if iteration % 10000 == 0 and iteration > 0:
                step_size *= 0.9  # Reduce step size to refine convergence

        self.print_results(algorithm, alpha)
        self.plot_rate_results(rates_over_time, algorithm, alpha)


    def initialize_lagrangian_multipliers(self):
        for link in self.links:
            link.LagrangianMultiplier = 0.1   # Initialize with a small positive value to avoid zero initialization issues

    def calculate_primal_rate(self, user, alpha, step_size=0.00001):
        if alpha == float('inf'):
            avg_rate = sum(u.rate for u in self.users) / len(self.users)
            if user.rate < avg_rate:
                return min(1, user.rate + step_size)
            else:
                return max(0, user.rate - step_size)
        
        if user.rate == 0:
            user.rate = 1e-6  # Small positive value to avoid zero division error

        payment = 0
        for link in self.get_user_links(user):  # calculate the payment of the user
            rate_sum = sum(u.rate for u in self.users if link in self.get_user_links(u))
            payment += self.penalty_function(rate_sum, link.capacity)
        return max(0, min(1, step_size * (pow(user.rate, -alpha) - payment) + user.rate))



    def calculate_dual_rate(self, user, alpha, step_size=0.0001):
        Q_l = 0
        for link in self.get_user_links(user):
            rate_sum = sum(u.rate for u in self.users if link in self.get_user_links(u))
            link.LagrangianMultiplier = max(0, link.LagrangianMultiplier + (rate_sum - link.capacity) * step_size)
            Q_l += link.LagrangianMultiplier

        # Ensure that Q_l is positive to avoid division by zero
        if Q_l == 0:
            return 0.0001  # A small positive rate to avoid zero rate issue

        return pow(Q_l, -1 / alpha)





    def penalty_function(self, rate, capacity):
        return rate * capacity if rate < capacity else pow(rate, 3) * 2

    def get_link(self, v1, v2):
        for link in self.links:
            if {link.vertex1.Vid, link.vertex2.Vid} == {v1, v2}:
                return link
        raise ValueError(f"No link between {v1} and {v2}")

    def get_user_links(self, user):
        return [self.get_link(user.path[i], user.path[i + 1]) for i in range(len(user.path) - 1)]

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
        for link in self.links:
            x_values = [link.vertex1.x, link.vertex2.x]
            y_values = [link.vertex1.y, link.vertex2.y]
            plt.plot(x_values, y_values, 'b-')
        for vertex in self.vertices:
            plt.plot(vertex.x, vertex.y, 'bo')
            plt.text(vertex.x, vertex.y, f'{vertex.Vid}', fontsize=12, ha='right')
        plt.title('Network Topology')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

def menu():
    while True:
        print("Choose a question to run the simulation:")
        print("2. NUM Problem using Primal Algorithm (TCP Reno)")
        print("3. NUM Problem using Dual Algorithm (TCP Vegas)")
        print("4. Generate Graphs for Different Alphas")
        print("0. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "2":
            alpha = float('inf')
            iterations = 100000
            network = Network()
            network.plot_network()
            network.simulate_num_problem(alpha, iterations, algorithm='Primal')
        elif choice == "3":
            alpha = float('1')
            iterations = 100000
            network = Network()
            network.plot_network()
            network.simulate_num_problem(alpha, iterations, algorithm='Dual')
        elif choice == "4":
            alphas = [1, 2, float('inf')]
            network = Network()
            network.plot_network()
            for alpha in alphas:
                network.simulate_num_problem(alpha, 100000, algorithm='Primal')
                network.simulate_num_problem(alpha, 100000, algorithm='Dual')
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
