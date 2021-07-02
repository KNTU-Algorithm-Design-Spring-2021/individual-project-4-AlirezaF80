from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    def BFS(self, s, t, parent):
        visited = []
        for i in range(self.ROW):
            visited.insert(i, False)

        queue = []

        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)

            for idx, val in enumerate(self.graph[u]):
                if not visited[idx] and val > 0:
                    queue.append(idx)
                    visited[idx] = True
                    parent[idx] = parent[idx] + [u]

        return True if visited[t] else False

    def findDisjointPaths(self, source, target):
        parent = [[-1]] * self.ROW

        max_flow = 0

        while self.BFS(source, target, parent):
            path_flow = float("Inf")
            s = target
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s][-1]][s])
                s = parent[s][-1]

            max_flow += path_flow

            v = target
            while v != source:
                u = parent[v][-1]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v][-1]

        parent = [list(set(i)) for i in parent]

        found_paths = []
        self.find_paths(target, source, parent, [target], found_paths)

        return_paths = []
        for i, cur_p in enumerate(found_paths):
            cln_paths = self.clean_possible_paths(cur_p, found_paths[i + 1:])
            if len(cln_paths) > 1:
                for p in cln_paths:
                    # print(found_paths[i][::-1], p[::-1])
                    return_paths.append([found_paths[i][::-1], p[::-1]])

        print(f'max flow going from {source} to {target} is {max_flow}')

        return return_paths

    @staticmethod
    def clean_possible_paths(cur_path, other_paths):
        disjoint_paths = []
        for other_p in other_paths:
            has_similar_edge = False
            for i in range(len(cur_path) - 1):
                for j in range(len(other_p) - 1):
                    if cur_path[i] == other_p[j] and cur_path[i + 1] == other_p[j + 1]:
                        has_similar_edge = True
                        break
            if not has_similar_edge:
                disjoint_paths.append(other_p)
        return disjoint_paths

    def find_paths(self, current, target: int, parents: List[List[int]], path: list, paths: List[list]):
        if current == target:
            paths.append(path)
            return True
        for p_c in parents[current]:
            if p_c != -1:
                self.find_paths(p_c, target, parents, path + [p_c], paths)
        return False


if __name__ == '__main__':
    graph = [[0, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0]]
    A = np.array(graph)

    g = Graph(graph)

    source = 0
    target = 7
    paths = g.findDisjointPaths(source, target)

    print('possible paths:')
    for i, j in paths:
        print(i, j)

    if paths:
        G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())

        first_paths = paths[0]

        colors = {}
        for e in G.edges:
            colors[e] = 'black'

        for i in range(len(first_paths[0]) - 1):
            colors[(first_paths[0][i], first_paths[0][i + 1], 0)] = 'red'

        for i in range(len(first_paths[1]) - 1):
            colors[(first_paths[1][i], first_paths[1][i + 1], 0)] = 'blue'

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, edge_color=colors.values())

        plt.gcf().canvas.manager.set_window_title('Wolf and Sheep')
        plt.show()
