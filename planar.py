import json
import sys
import networkx as nx
from MaximiN import visualize_graph  # Используем вашу визуализацию

# Увеличиваем лимит рекурсии для глубоких деревьев
sys.setrecursionlimit(5000)


def get_subtree_sizes_for_root(G, root):
    """Итеративное вычисление размеров поддерева для каждого узла относительно корня."""
    subtree_sizes = {}
    # BFS для получения порядка и родителей
    nodes_ordered = []
    queue = [root]
    visited = {root}
    parent = {root: None}
    while queue:
        u = queue.pop(0)
        nodes_ordered.append(u)
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                queue.append(v)

    # Считаем размеры снизу вверх
    for u in reversed(nodes_ordered):
        size = 1
        for v in G.neighbors(u):
            if v != parent[u]:
                size += subtree_sizes[v]
        subtree_sizes[u] = size
    return subtree_sizes, parent


def get_heaviest_path_iterative(G, start_node):
    """Находит цепь максимального веса (Теорема 10, стр. 81)."""
    sizes, parent = get_subtree_sizes_for_root(G, start_node)
    path = [start_node]
    curr = start_node
    while True:
        next_node = None
        max_s = -1
        for v in G.neighbors(curr):
            if v != (parent[curr] if curr in parent else None) and v not in path:
                if sizes.get(v, 0) > max_s:
                    max_s = sizes[v]
                    next_node = v
        if next_node is None: break
        path.append(next_node)
        curr = next_node
    return path


def build_planar_recursive(G, nodes_subset, start_val, end_val):
    """Рекурсивное разложение дерева на цепи (алгоритм со стр. 81)."""
    if not nodes_subset: return {}
    subG = G.subgraph(nodes_subset)
    if subG.number_of_nodes() == 1:
        return {list(subG.nodes())[0]: start_val}

    # Поиск основной цепи sigma_1
    any_v = list(subG.nodes())[0]
    v1 = get_heaviest_path_iterative(subG, any_v)[-1]
    chain = get_heaviest_path_iterative(subG, v1)
    v2 = chain[-1]

    res = {v1: start_val, v2: end_val}

    # Распределение номеров и рекурсия для ветвей (стр. 82)
    ptr = start_val + 1
    for i in range(1, len(chain) - 1):
        u = chain[i]
        res[u] = ptr
        ptr += 1
        for v in subG.neighbors(u):
            if v not in chain:
                branch = nx.node_connected_component(subG.subgraph(set(subG.nodes()) - {u}), v)
                size = len(branch)
                res.update(build_planar_recursive(G, branch, ptr, ptr + size - 1))
                ptr += size
    return res


def main(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()
    vertices = data["vertices"]
    for e in data["edges"]:
        u, v = vertices[e["vertex1"]]["name"], vertices[e["vertex2"]]["name"]
        G.add_edge(u, v)

    n = G.number_of_nodes()
    # Запуск алгоритма минимальной плоской нумерации
    res_map = build_planar_recursive(G, set(G.nodes()), 1, n)

    # Формируем итоговый порядок для визуализации
    final_order = [None] * n
    for name, val in res_map.items():
        final_order[val - 1] = name

    # РАСЧЕТ ДЛИНЫ (добавлено)
    print("=== Минимальная плоская нумерация ===")
    total_len = 0
    # Проходим по всем ребрам исходного графа
    for u_name, v_name in G.edges():
        num_u = res_map[u_name]
        num_v = res_map[v_name]
        dist = abs(num_u - num_v)
        total_len += dist
        print(f"{num_u} - {num_v} = {dist}")

    print(f"Итоговая сумма длин (Δ): {total_len}")

    # Визуализация с использованием существующих координат
    visualize_graph(vertices, data["edges"], final_order, "Min Planar",
                    "graph_planar.png", relayout="preserve")
    print("✅ Визуализация сохранена в graph_planar.png")