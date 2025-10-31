import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx
from christofides import *
from time import time
import psutil
import os

# -------------------------------
# Préparation des graphes avec monitoring
# -------------------------------
process = psutil.Process(os.getpid())
memory_start = process.memory_info().rss / 1024 / 1024  # MB

time_general_start = time()
time_G_start = time()
G = build_complete_graph(villes)
time_G_end = time()
time_G = time_G_end - time_G_start
print(f"Temps de construction du graphe complet : {time_G:.6f} secondes")

time_mst_start = time()
mst_G = compute_mst(G)
time_mst_end = time()
time_mst = time_mst_end - time_mst_start
print(f"Temps de calcul du MST : {time_mst:.6f} secondes")

time_odd_start = time()
odd_nodes = find_odd_nodes(mst_G)
time_odd_end = time()
time_odd = time_odd_end - time_odd_start
print(f"Temps de recherche des sommets impairs : {time_odd:.6f} secondes")

time_matching_start = time()
matching_edges = compute_minimum_matching(G, odd_nodes)
time_matching_end = time()
time_matching = time_matching_end - time_matching_start
print(f"Temps de calcul du matching minimum : {time_matching:.6f} secondes")

time_eulerian_start = time()
eulerian_G = build_eulerian_graph(mst_G, matching_edges, G)
time_eulerian_end = time()
time_eulerian = time_eulerian_end - time_eulerian_start
print(f"Temps de construction du graphe eulérien : {time_eulerian:.6f} secondes")

time_tsp_start = time()
tsp_path = compute_tsp_path(eulerian_G)
time_tsp_end = time()
time_tsp = time_tsp_end - time_tsp_start
print(f"Temps de calcul du chemin TSP approximatif : {time_tsp:.6f} secondes")

time_general_end = time()
time_total = time_general_end - time_general_start
print(f"Temps total de l'algorithme de Christofides : {time_total:.6f} secondes")

# Ressources
memory_end = process.memory_info().rss / 1024 / 1024  # MB
memory_used = memory_end - memory_start
cpu_percent = process.cpu_percent(interval=0.1)

print(f"\n--- Ressources ---")
print(f"Mémoire utilisée : {memory_used:.2f} MB")
print(f"Mémoire totale : {memory_end:.2f} MB")
print(f"CPU : {cpu_percent:.1f}%")

# Stocker les temps pour l'affichage
timing_info = {
    'Graphe complet': time_G,
    'MST': time_mst,
    'Sommets impairs': time_odd,
    'Matching minimum': time_matching,
    'Graphe eulérien': time_eulerian,
    'Chemin TSP': time_tsp,
    'TOTAL': time_total
}

resource_info = {
    'Mémoire utilisée': f"{memory_used:.2f} MB",
    'Mémoire totale': f"{memory_end:.2f} MB",
    'CPU': f"{cpu_percent:.1f}%"
}

# -------------------------------
# Visualisation interactive
# -------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
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

    ax.set_title(f"{title} (← / → pour naviguer)", fontsize=12, pad=20)
    
    # Ajouter les informations de temps et ressources
    timing_text = "=== Temps d'exécution ===\n"
    for key, value in timing_info.items():
        if key == 'TOTAL':
            timing_text += f"\n{key}: {value:.4f}s"
        else:
            timing_text += f"{key}: {value:.4f}s\n"
    
    resource_text = "\n\n=== Ressources ===\n"
    for key, value in resource_info.items():
        resource_text += f"{key}: {value}\n"
    
    info_text = timing_text + resource_text
    
    # Positionner le texte dans le coin supérieur gauche
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

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