import json
import random
import math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# === 1. Подсчёт стоимости (С ВОЗВРАЩЕННЫМ ВЫВОДОМ) ===
def numbering_length(order, edges, verbose=True):
    name_to_number = {name: i + 1 for i, name in enumerate(order)}
    total = 0

    # Фильтруем ребра
    valid_edges = []
    for u, v in edges:
        if u in name_to_number and v in name_to_number:
            valid_edges.append((u, v))

    max_edge = (None, 0)
    min_edge = (None, float("inf"))

    for v1, v2 in valid_edges:
        dist = abs(name_to_number[v1] - name_to_number[v2])
        total += dist

        if dist > max_edge[1]:
            max_edge = ((v1, v2), dist)
        if dist < min_edge[1]:
            min_edge = ((v1, v2), dist)

        # --- ВЕРНУЛ ЭТОТ БЛОК ---
        if verbose:
            # Выводим: Номер1 - Номер2 = Длина
            print(f"{name_to_number[v1]} - {name_to_number[v2]} = {dist}")

    if verbose:
        print(f"Итоговая длина нумерации: {total}")
        if max_edge[0]:
            print(f"Минимальное ребро: {min_edge[0]} = {min_edge[1]}")
            print(f"Максимальное ребро: {max_edge[0]} = {max_edge[1]}")

    return total, name_to_number


# === 2. Мутации (Три вида движений для макс. качества) ===
def apply_random_move(order):
    new_order = order[:]
    n = len(order)
    if n < 2: return new_order

    move_type = random.random()

    if move_type < 0.33:
        # Swap
        i, j = random.sample(range(n), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    elif move_type < 0.66:
        # Insert
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            val = new_order.pop(i)
            new_order.insert(j, val)
    else:
        # Reverse (2-opt)
        i, j = random.sample(range(n), 2)
        if i > j: i, j = j, i
        new_order[i:j + 1] = reversed(new_order[i:j + 1])

    return new_order


# === 3. Тяжелая Имитация Отжига ===
def heavy_annealing(order, edges, maximize=False, steps=100000):
    current_order = order[:]
    best_order = order[:]

    # verbose=False здесь важно, чтобы не спамить во время расчетов
    current_len, _ = numbering_length(current_order, edges, verbose=False)
    best_len = current_len

    T_start = 50.0
    T_end = 0.001

    for step in range(steps):
        decay = math.log(T_end / T_start) / steps
        T = T_start * math.exp(decay * step)

        new_order = apply_random_move(current_order)
        new_len, _ = numbering_length(new_order, edges, verbose=False)

        delta = new_len - current_len

        accept = False
        if maximize:
            if delta > 0:
                accept = True
            elif random.random() < math.exp(delta / T):
                accept = True
        else:
            if delta < 0:
                accept = True
            elif random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_order = new_order
            current_len = new_len

            if maximize:
                if current_len > best_len:
                    best_len = current_len
                    best_order = current_order[:]
            else:
                if current_len < best_len:
                    best_len = current_len
                    best_order = current_order[:]

    return best_len, best_order


# === 4. Solver с перезапусками ===
def solve_heavy(vertex_names, edges_named, maximize=False, restarts=20):
    best_global_len = -1 if maximize else float('inf')
    best_global_order = None

    for r in range(restarts):
        start_order = random.sample(vertex_names, len(vertex_names))

        # 150 000 итераций на прогон
        score, order = heavy_annealing(start_order, edges_named, maximize=maximize, steps=150000)

        improved = False
        if maximize:
            if score > best_global_len:
                best_global_len = score
                best_global_order = order[:]
                improved = True
        else:
            if score < best_global_len:
                best_global_len = score
                best_global_order = order[:]
                improved = True

        mark = "🌟 НОВЫЙ РЕКОРД" if improved else ""

    return best_global_len, best_global_order


# === 5. Визуализация ===
def visualize_graph(vertices, edges, name_to_number, length, filename, maximize=False):
    color = "#ffa3a3" if maximize else "#a3d5ff"
    title_text = "Max" if maximize else "Min"
    positions = {name_to_number[v["name"]]: (v["x"], -v["y"]) for v in vertices}

    fig, ax = plt.subplots(figsize=(12, 8))

    for e in edges:
        try:
            v1 = vertices[e["vertex1"]]["name"]
            v2 = vertices[e["vertex2"]]["name"]
            if v1 not in name_to_number or v2 not in name_to_number: continue

            u, v = name_to_number[v1], name_to_number[v2]
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            color_edge = e.get("color", "#000000")
            width = e.get("lineWidth", 2)
            control = e.get("controlStep", 0)

            if control == 0:
                ax.plot([x1, x2], [y1, y2], color=color_edge, linewidth=width, zorder=1)
            else:
                rad = control / 300.0
                patch = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                                        arrowstyle="-", color=color_edge, linewidth=width, zorder=2)
                ax.add_patch(patch)
        except:
            continue

    for num, (x, y) in positions.items():
        ax.scatter(x, y, s=800, color=color, edgecolors="black", zorder=3)
        ax.text(x, y, str(num), fontsize=10, weight="bold", ha="center", va="center", zorder=4)

    plt.title(f"{title_text} \nДлина = {length}", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    xs = [v[0] for v in positions.values()]
    ys = [v[1] for v in positions.values()]
    margin = 50
    if xs:
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Граф сохранён в {filename}")


# === Main ===
def main(file_path, choice="2"):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка открытия файла: {e}")
        return

    vertices = data["vertices"]
    edges_raw = data["edges"]

    vertex_names = [v["name"] for v in vertices]
    edges_named = []

    for e in edges_raw:
        try:
            v1_name = vertices[e["vertex1"]]["name"]
            v2_name = vertices[e["vertex2"]]["name"]
            edges_named.append((v1_name, v2_name))
        except IndexError:
            continue

    if choice == "1":
        print("\n=== Режим 1: Проверка исходной нумерации ===")
        try:
            try:
                order_from_file = sorted(vertex_names, key=lambda x: int(x))
            except ValueError:
                order_from_file = sorted(vertex_names)
            # Здесь verbose=True выведет все ребра
            length_file, map_file = numbering_length(order_from_file, edges_named, verbose=True)
            visualize_graph(vertices, edges_raw, map_file, length_file, "graph_from_file.png")
        except Exception as e:
            print(f"Ошибка: {e}")

    else:
        print("\n=== Режим 2: Поиск оптимальной ===")

        # --- MINIMIZATION ---
        min_len, min_order = solve_heavy(vertex_names, edges_named, maximize=False, restarts=30)

        print(f"\n🏆 ФИНАЛЬНЫЙ MIN: {min_len}")
        print("--- Детализация ребер для MIN ---")
        # Здесь verbose=True покажет ребра лучшего решения
        _, map_min = numbering_length(min_order, edges_named, verbose=True)
        visualize_graph(vertices, edges_raw, map_min, min_len, "graph_min.png", maximize=False)

        # --- MAXIMIZATION ---
        max_len, max_order = solve_heavy(vertex_names, edges_named, maximize=True, restarts=20)

        print(f"\n🏆 ФИНАЛЬНЫЙ MAX: {max_len}")
        print("--- Детализация ребер для MAX ---")
        # Здесь verbose=True покажет ребра лучшего решения
        _, map_max = numbering_length(max_order, edges_named, verbose=True)
        visualize_graph(vertices, edges_raw, map_max, max_len, "graph_max.png", maximize=True)


if __name__ == "__main__":
    main("graph7.graph", choice="2")