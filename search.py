import sys
import time
import tracemalloc
from collections import deque, defaultdict
import heapq
from queue import PriorityQueue


class Node:
        def __init__(self, x, cost):
            self.x = x
            self.cost = cost

        def __lt__(self, other):
            return self.cost < other.cost


class Graph:
    def __init__(self):
        self.graph = defaultdict(set)
        self.n = 0
        self.start = 0
        self.end = 0
        self.cost = defaultdict(int)
        self.heuristics = defaultdict(int)
    

    def add_edge(self, u, v, cost):
        self.graph[u].add(v)
        self.cost[(u, v)] = cost


    def read_graph(self, filename):
        with open(filename, 'r') as f:
            self.n = int(f.readline())
            self.start, self.end = map(int, f.readline().split())
            for i in range(self.n):
                row = list(map(int, f.readline().split()))
                for j in range(self.n):
                    if row[j] != 0:
                        self.add_edge(i, j, row[j])

            row = list(map(int, f.readline().split()))
            for i in range(self.n):
                self.heuristics[i] = row[i]

            return self.n, self.start, self.end, self.graph, self.cost, self.heuristics
        

    def print_path(self, path):
        print(' -> '.join(map(str, path)))


    def save_result(self, algorithm, path, time_taken, memory_used):
        filename = algorithm.lower().replace(' ', '_') + '.txt'
        with open(filename, 'w') as f:
            f.write(algorithm)
            f.write('\n')
            f.write('Path: ')
            f.write(' -> '.join(map(str, path)))
            f.write('\n')
            f.write(f'Time: {time_taken:.10f}\n')
            f.write(f'Memory: {memory_used / 1024:.10f} KB\n')
    

    def BFS(self, start, end):
        if start == end:
            return [start]
        
        visited = [False] * self.n
        visited[start] = True
        queue = [(start, [start])]

        while queue:
            node, path = queue.pop(0)

            for i in sorted(self.graph[node]):
                if not visited[i]:
                    visited[i] = True
                    if i == end:
                        return path + [i]
                    queue.append((i, path + [i]))
            
        return -1
    

    def DFS(self, start, end):
        if start == end:
            return [start]

        stack = [(start, [start])]

        while stack:
            (vertex, path) = stack.pop()

            # neighbors_list = self.graph[vertex]
            # sorted_neighbors = sorted(neighbors_list, reverse=True)
            for neighbor in sorted(self.graph[vertex], reverse=True):
                if neighbor not in path:
                    if neighbor == end:
                        return path + [neighbor]
                    stack.append((neighbor, path + [neighbor]))

        return -1
    

    def UCS(self, start, end):
        if start == end:
            return [start], 0

        visited = defaultdict(bool)
        queue = PriorityQueue()
        queue.put((0, [start]))
        while queue:
            cost, path = queue.get()
            node = path[-1]
            if node == end:
                return path, cost
            if not visited[node]:
                visited[node] = True
                for neighbour in self.graph[node]:
                    if neighbour not in path:
                        new_path = list(path)
                        new_path.append(neighbour)
                        queue.put((cost + self.cost[(node, neighbour)], new_path))

        return -1, -1
    

    def DLS(self, start, end, limit, path=[]):
        if start == end:
            return True
        
        if limit <= 0:
            return False
        
        for neighbour in self.graph[start]:
            if neighbour not in path:
                path.append(neighbour)
                if self.DLS(neighbour, end, limit - 1, path):
                    return path
                path.pop() 
            

    def IDS(self, start, end):
        if start == end:
            return [start]

        for depth in range(self.n):
            path = self.DLS(start, end, depth)
            if path:
                return [start] + path
        
    
    def GBFS(self, start, end):
        if start == end:
            return [start]

        visited = set()
        visited.add(start)
        pq = []
        heapq.heappush(pq, Node(start, self.heuristics[start]))
        trace = {}

        while pq:
            current = heapq.heappop(pq)
            currentNode = current.x

            if currentNode == end:
                path = []
                while currentNode != start:
                    path.append(currentNode)
                    currentNode = trace[currentNode]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in self.graph[currentNode]:
                if neighbor not in visited:
                    heapq.heappush(pq, Node(neighbor, self.heuristics[neighbor]))
                    visited.add(neighbor)
                    trace[neighbor] = currentNode

        return -1  

        
    def A_STAR(self, start, end):
        if start == end:
            return [start]

        open_list = set([start])
        closed_list = set()

        g = {start: 0}
        parents = {start: start}

        while open_list:
            n = None

            for v in open_list:
                if n is None or g[v] + self.heuristics[v] < g[n] + self.heuristics[n]:
                    n = v

            if n is None:
                return -1

            if n == end:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start)
                reconst_path.reverse()

                return reconst_path

            for m in self.graph[n]:
                weight = self.cost[(n, m)]
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        return -1
    

    def HC(self, start, end):
        if start == end:
            return [start]

        current_node = (start, self.heuristics[start])
        path = [current_node[0]]

        while True:
            next_node = False

            for neighbour in self.graph[current_node[0]]:
                if neighbour not in path:
                    if self.heuristics[neighbour] < current_node[1]:
                        next_node = True
                        current_node = (neighbour, self.heuristics[neighbour])

            if not next_node:
                return -1
            
            if current_node[0] == end:
                path.append(current_node[0])
                return path
            
            path.append(current_node[0])
    

def main(*args):
    g = Graph()
    n, start, end, graph, cost, heuristics = g.read_graph(args[0])

    result = {}
    
    start_time = time.time()
    tracemalloc.start()
    path = g.BFS(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['BFS'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path = g.DFS(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['DFS'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path, _ = g.UCS(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['UCS'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path = g.IDS(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['IDS'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path = g.GBFS(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['GBFS'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path = g.A_STAR(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['A_STAR'] = (path, time_taken, peak)

    start_time = time.time()
    tracemalloc.start()
    path = g.HC(start, end)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = time.time() - start_time
    result['Hill Climbing'] = (path, time_taken, peak)

    print(result)

    output_file = args[0].split('.')[0] + '.out'
    with open(output_file, 'w') as f:
        for algo, (path, time_taken, peak) in result.items():
            f.write(f'{algo}:\n')
            if path != -1:
                f.write(f'Path: {" -> ".join(map(str, path))}\n')
            else:
                f.write('Path: -1\n')
            f.write(f'Time: {time_taken:.10f} seconds\n')
            f.write(f'Memory: {peak / 1024:.10f} KB\n\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python search.py <input_file>')
        sys.exit(1)
    main(*sys.argv[1:])