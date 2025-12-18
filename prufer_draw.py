import json
import networkx as nx
import math


def prufer_to_graph_file(prufer, filename):
    """
    Генерирует граф из кода Прюфера с радиальной укладкой (Spider Look).
    """
    n = len(prufer) + 2
    # NetworkX создает узлы от 0 до n-1
    G = nx.from_prufer_sequence([p - 1 for p in prufer])

    # 1. Находим корень для радиальной укладки (узел с наибольшим числом связей)
    root = max(G.nodes(), key=G.degree)
    pos = {}

    # 2. Рекурсивный расчет координат (Радиальный обход)
    def layout_radial(node, parent=None, angle_min=0, angle_max=2 * math.pi, depth=0):
        # Вычисляем угол посередине сектора
        angle = (angle_min + angle_max) / 2
        # Расстояние от центра увеличивается с глубиной (200 - шаг между уровнями)
        pos[node] = (
            depth * math.cos(angle) * 200,
            depth * math.sin(angle) * 200
        )

        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            return

        # Делим текущий сектор между детьми
        angle_step = (angle_max - angle_min) / len(children)
        for i, child in enumerate(children):
            layout_radial(
                child,
                node,
                angle_min + i * angle_step,
                angle_min + (i + 1) * angle_step,
                depth + 1
            )

    # Запускаем расчет от корня
    layout_radial(root)

    # 3. Формируем список вершин (сдвигаем к центру холста 600, 600)
    vertices = []
    # Сортируем узлы 0..n-1, чтобы их индекс в списке совпадал с номером
    for node in sorted(G.nodes()):
        vertices.append({
            "x": int(pos[node][0] + 600),
            "y": int(pos[node][1] + 600),
            "name": str(node + 1),  # Имя вершины будет 1..n
            "radius": 20,
            "background": "#ffffff",
            "fontSize": 18,
            "color": "#000000",
            "border": "#000000"
        })

    # 4. Формируем список ребер
    edges_json = []
    for u, v in G.edges():
        edges_json.append({
            "vertex1": u,  # Индекс в списке vertices (совпадает с node)
            "vertex2": v,
            "isDirected": False,
            "lineWidth": 2,
            "color": "#000000"
        })

    # 5. Сохраняем итоговый JSON один раз
    graph_data = {
        "vertices": vertices,
        "edges": edges_json
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Граф с радиальной укладкой сохранен в {filename}")
    return filename