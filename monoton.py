import json
import networkx as nx
from MaximiN import visualize_graph  # Используем функцию визуализации из проекта


def get_subtree_size(G, u, p):
    """Рекурсивно вычисляет количество вершин в поддереве узла u."""
    size = 1
    for v in G.neighbors(u):
        if v != p:
            size += get_subtree_size(G, v, u)
    return size


def find_centroid(G):
    """Находит центроид дерева для выбора сбалансированного корня."""
    n = G.number_of_nodes()
    centroids = []
    min_max_subtree = float('inf')

    for v in G.nodes():
        max_subtree = 0
        for neighbor in G.neighbors(v):
            branch = nx.node_connected_component(G.subgraph(set(G.nodes()) - {v}), neighbor)
            max_subtree = max(max_subtree, len(branch))

        if max_subtree < min_max_subtree:
            min_max_subtree = max_subtree
            centroids = [v]
        elif max_subtree == min_max_subtree:
            centroids.append(v)
    return centroids[0]


def get_optimal_root(G, centroid):
    """Ищет самый удаленный лист в самой тяжелой ветке от центроида."""
    max_weight = -1
    best_neighbor = None

    for neighbor in G.neighbors(centroid):
        branch = nx.node_connected_component(G.subgraph(set(G.nodes()) - {centroid}), neighbor)
        if len(branch) > max_weight:
            max_weight = len(branch)
            best_neighbor = neighbor

    if not best_neighbor:
        return centroid

    branch_nodes = nx.node_connected_component(G.subgraph(set(G.nodes()) - {centroid}), best_neighbor)
    branch_sub = G.subgraph(branch_nodes)
    dist = nx.single_source_shortest_path_length(branch_sub, best_neighbor)
    return max(dist, key=dist.get)


def build_min_monotone_order(G, root):
    """
    Строит нумерацию согласно алгоритму из учебника:
    DFS обход, приоритет — ветвь с МЕНЬШИМ числом вершин.
    """
    order = []

    def dfs(u, p):
        # 1. Присвоить текущий номер вершине (добавляем в список порядка)
        order.append(u)

        # 2. Найти всех детей и размеры их поддеревьев
        children = []
        for v in G.neighbors(u):
            if v != p:
                size = get_subtree_size(G, v, u)
                children.append((v, size))

        # 3. Сортировка по ВОЗРАСТАНИЮ размера поддерева (меньшая ветка первой)
        # Это соответствует правилу из image_f5d838.png
        children.sort(key=lambda x: x[1])

        # 4. Рекурсивный обход
        for child_v, _ in children:
            dfs(child_v, u)

    dfs(root, None)
    return order


def main(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vertices = data["vertices"]
    edges_raw = data["edges"]
    G = nx.Graph()
    for e in edges_raw:
        u, v = vertices[e["vertex1"]]["name"], vertices[e["vertex2"]]["name"]
        G.add_edge(u, v)

    if G.number_of_nodes() == 0:
        print("Граф пуст.")
        return

    # Находим корень и строим порядок по новому алгоритму
    centroid = find_centroid(G)
    optimal_root = get_optimal_root(G, centroid)
    final_order = build_min_monotone_order(G, optimal_root)

    print("=== Минимальная монотонная нумерация (Smallest Branch First DFS) ===")
    print(f"Выбран корень: {optimal_root}")

    pos = {name: i + 1 for i, name in enumerate(final_order)}
    total_len = 0
    for u, v in G.edges():
        d = abs(pos[u] - pos[v])
        total_len += d
        print(f"{pos[u]} - {pos[v]} = {d}")

    print(f"Итоговая сумма длин (Δ): {total_len}")

    # Визуализация с сохранением структуры координат
    visualize_graph(vertices, edges_raw, final_order, "Min Monotone (Smallest Branch First)",
                    "graph_monotone.png", relayout="preserve")
    print("✅ Визуализация сохранена в graph_monotone.png")