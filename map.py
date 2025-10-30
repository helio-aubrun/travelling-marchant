import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx
from christofides import *

# -------------------------------
# Préparation des graphes
# -------------------------------
G = build_complete_graph(villes)
mst_G = compute_mst(G)
odd_nodes = find_odd_nodes(mst_G)
matching_edges = compute_minimum_matching(G, odd_nodes)
eulerian_G = build_eulerian_graph(mst_G, matching_edges, G)
tsp_path = compute_tsp_path(eulerian_G)

# -------------------------------
# Visualisation interactive
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
m = Basemap(projection='merc',
            llcrnrlat=41.0, urcrnrlat=51.5,
            llcrnrlon=-5.5, urcrnrlon=9.5,
            resolution='i', ax=ax)

pos = {v: m(lon, lat) for v, (lat, lon) in villes.items()}

step = 0
steps = 6  # 0→5 : les 6 étapes

def draw_step():
    ax.clear()
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='beige', lake_color='lightblue')

    title = ""
    odd_to_draw, edges_to_draw, matching_to_draw, eulerian_to_draw, tsp_edges = [], [], [], [], []

    if step == 0:
        edges_to_draw = G.edges()
        title = "Graphe complet"
    elif step == 1:
        edges_to_draw = mst_G.edges()
        title = "MST"
    elif step == 2:
        edges_to_draw = mst_G.edges()
        odd_to_draw = odd_nodes
        title = "MST + sommets impairs"
    elif step == 3:
        edges_to_draw = mst_G.edges()
        matching_to_draw = matching_edges
        odd_to_draw = odd_nodes
        title = "MST + sommets impairs + matching"
    elif step == 4:
        eulerian_to_draw = list(eulerian_G.edges())
        title = "Graphe eulérien (MST + matching)"
    elif step == 5:
        tsp_edges = [(tsp_path[i], tsp_path[i+1]) for i in range(len(tsp_path)-1)]
        title = "Circuit final approximatif (Christofides)"

    # Tracer les arêtes
    for u, v in edges_to_draw:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.7, alpha=0.7)
    for u, v in matching_to_draw:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=1.5, alpha=0.8)
    for u, v in eulerian_to_draw:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='purple', linewidth=1.2, alpha=0.8)
    for u, v in tsp_edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='green', linewidth=2, alpha=0.8)

    # Tracer les villes
    for ville, (x, y) in pos.items():
        size = 10 if ville in odd_to_draw else 5
        ax.plot(x, y, 'ro', markersize=size)
        ax.text(x + 10000, y + 10000, ville, fontsize=8)

    ax.set_title(f"{title} (← / → pour naviguer)", fontsize=12)

def on_key(event):
    global step
    if event.key == 'right':
        step = (step + 1) % steps
        draw_step()
        plt.draw()
    elif event.key == 'left':
        step = (step - 1) % steps
        draw_step()
        plt.draw()

draw_step()
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
