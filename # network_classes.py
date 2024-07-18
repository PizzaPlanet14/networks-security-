import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

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
        self.rate = 0
        self.links = []
        self.data = data

class Network:
    def __init__(self):
        self.vertices = []
        self.links = []
        self.users = []
        self.create_network()
        self.create_users()

    def create_network(self):
        N = 6
        np.random.seed(42)  # For consistent graph generation
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
        paths = [[1, 2, 3, 4, 5, 6], [1, 3, 4, 5, 6], [2, 3, 5, 6], [1, 2, 5, 6], [3, 4, 5, 6], [1, 4, 5, 6]]
        data = [743, 761, 276, 334, 669, 89]
        for i, (path, data) in enumerate(zip(paths, data)):
            user = User(i+1, path, data)
            user.links = [link for link in self.links if link.vertex1.Vid in path and link.vertex2.Vid in path]
            self.users.append(user)

    def simulate_num_problem(self, alpha, iterations, algorithm):
        CalcNetworkRate(self, alpha, algorithm, iterations)

    def initial_users_rates(self):
        for user in self.users:
            user.rate = 0.1  # Initial rate

    def print_user_paths(self):
        for user in self.users:
            print(f"User {user.user_id}:")
            print(f"  Path: {' -> '.join(map(str, user.path))}")
            print(f"  Data: {user.data}")
            print(f"  Rate: {user.rate:.2f}")

    def plot_network(self):
        plt.figure()
        for link in self.links:
            x = [link.vertex1.x, link.vertex2.x]
            y = [link.vertex1.y, link.vertex2.y]
            plt.plot(x, y, 'bo-')
            plt.text((link.vertex1.x + link.vertex2.x) / 2, (link.vertex1.y + link.vertex2.y) / 2, f"{link.link_id}")
        for vertex in self.vertices:
            plt.text(vertex.x, vertex.y, f"{vertex.Vid}", fontsize=12, ha='right')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Network Graph')
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

def CalcNetworkRate(network, alpha, Algorithm, N=1e5):
    network.initial_users_rates()
    algorithm_functions = {"Primal": CalcPrimalRate, "Dual": CalcDualRate}
    CalcRate = algorithm_functions.get(Algorithm)
    users = network.users

    xAxis = []
    yAxis = []
    for _ in users:  # initialize the graph
        xAxis.append([])
        yAxis.append([])

    for i in range(int(N)):
        curUser = random.choice(users)
        id = curUser.user_id - 1
        x_r = curUser.rate
        curUser.rate = CalcRate(curUser, users, alpha, x_r)
        xAxis[id].append(i)
        yAxis[id].append(curUser.rate)

    PrintRateResults(xAxis, yAxis, users, alpha, Algorithm)

def CalcPrimalRate(user, users, alpha, x_r, stepSize=0.0001):
    if alpha == float("inf"):
        avg_rate = sum(u.rate for u in users) / len(users)
        return max(0, min(1, avg_rate + stepSize)) if user.rate < avg_rate else max(0, user.rate - stepSize)

    payment = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = 0
        for u in users:  # calculate the sum of the rates of all the users on the link
            if link in u.links:
                rateSum += u.rate
        payment += penaltyFunction(rateSum, link.capacity)
    return stepSize * (pow(user.rate, -1 * alpha) - payment) + x_r  # calculate the next rate of the user

def penaltyFunction(rate, capacity):
    if rate < capacity:
        return rate * capacity
    else:
        try:
            return pow(rate, 3) * 2
        except OverflowError:  # TODO: check why it is overflow error
            return 0

def CalcDualRate(user, users, alpha, x_r, stepSize=0.0001):
    if alpha == float("inf"):
        max_excess = max((sum(u.rate for u in users if link in u.links) - link.capacity) for link in user.links)
        return max(0, min(1, x_r - stepSize * max_excess))

    Q_l = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = sum(u.rate for u in users if link in u.links)  # Y_l
        L_delta = (rateSum - link.capacity) * stepSize
        link.LagrangianMultiplier = max(0, link.LagrangianMultiplier + L_delta)
        Q_l += link.LagrangianMultiplier
    return pow(Q_l, -1/alpha) if Q_l != 0 else 0  # the inverse function of the utilization function

def PrintRateResults(xAxis, yAxis, users, alpha, Algorithm):
    plt.figure()
    for user_id, (x, y) in enumerate(zip(xAxis, yAxis), start=1):
        plt.plot(x, y, label=f'user {user_id}')
    plt.xlabel('Iteration Number')
    plt.ylabel('Rate')
    plt.title(f'{Algorithm} Algorithm, alpha={alpha}')
    plt.legend()
    plt.show()

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
            network.print_user_paths()
        elif choice == "3":
            alpha = 1  # Example value, should be configurable
            iterations = 100000  # Example value, should be configurable
            network.simulate_num_problem(alpha, iterations, algorithm='Dual')
            network.print_user_paths()
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
