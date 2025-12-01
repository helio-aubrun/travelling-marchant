# filepath: map.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.basemap import Basemap
import networkx as nx
from christofides import *
from genetique import *
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
steps = 6

# -------------------------------
# Algorithme génétique : configuration et exécution
# -------------------------------
ga_cfg = GAConfig(population_size=320, generations=600, mutation_rate=0.18,
                  mutation_op="inversion", seed=7, patience=140)
ga_path, ga_best_km, ga_history = run_ga_tsp(villes, ga_cfg)

def _dist(path, cities):
    return sum(haversine(cities[path[i]], cities[path[i+1]]) for i in range(len(path)-1))

christofides_km = _dist(tsp_path, villes)

# -------------------------------
# Algorithme génétique : état UI et helpers
# -------------------------------
gen_mode, ga_step, GA_STEPS = False, 0, 3
edges = lambda p: [(p[i], p[i+1]) for i in range(len(p)-1)]
eset  = lambda p: {frozenset((p[i], p[i+1])) for i in range(len(p)-1)}

def _cities():
    for city, (x, y) in pos.items():
        ax.plot(x, y, 'ro', markersize=5)
        ax.text(x+10000, y+10000, city, fontsize=8)

def _text(x, y, s, right=False):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9,
            va='top', ha='right' if right else 'left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1),
            family='monospace')

def _ga_params_text():
    return (f"=== Paramètres GA ===\nPop: {ga_cfg.population_size}\nGen: {ga_cfg.generations}\n"
            f"Tournoi k: {ga_cfg.tournament_k}\nOX: {ga_cfg.crossover_rate}\n"
            f"Mutation: {ga_cfg.mutation_rate} ({ga_cfg.mutation_op})\n"
            f"Élitisme: {ga_cfg.elitism}\nPatience: {ga_cfg.patience}\nSeed: {ga_cfg.seed}")

def _ga_prog_text():
    if not ga_history: return ""
    a, z = ga_history[0], ga_history[-1]
    mid = ga_history[len(ga_history)//2]
    gain = (a - z) / a * 100 if a > 0 else 0
    return (f"=== Progression GA ===\nDépart: {a:.1f} km\nMi: {mid:.1f} km\n"
            f"Final: {z:.1f} km\nGain: {gain:.1f}%")

def _ga_vs_ch_text():
    ch, ga = eset(tsp_path), eset(ga_path)
    common = ch & ga
    ch_only = ch - ga
    ga_only = ga - ch
    diff_km = ga_best_km - christofides_km
    rel = (diff_km / christofides_km * 100) if christofides_km > 0 else 0.0
    sign = "+" if diff_km >= 0 else ""
    return (
        "=== Comparaison GA vs Christofides ===\n"
        f"Christofides: {christofides_km:.1f} km\n"
        f"GA: {ga_best_km:.1f} km\n"
        f"Δ distance: {sign}{diff_km:.1f} km ({sign}{rel:.1f}%)\n"
        f"Arêtes communes: {len(common)}\n"
        f"Uniq. Christofides: {len(ch_only)}\n"
        f"Uniq. GA: {len(ga_only)}"
    )

def _draw_ga(mode: int):
    _cities()
    if mode == 0:
        _text(0.02, 0.98, _ga_params_text())
        ax.set_title("Algorithme Génétique — paramètres (← / →)", fontsize=12, pad=20); return
    if mode == 1:
        for u, v in edges(ga_path):
            x1, y1 = pos[u]; x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color='orange', lw=2.5, alpha=0.95, ls='--')
        _text(0.02, 0.98, _ga_params_text())
        _text(0.98, 0.98, _ga_prog_text(), right=True)
        ax.set_title(f"Algorithme Génétique — meilleur circuit ({ga_best_km:.1f} km)", fontsize=12, pad=20); return
    ch, ga = eset(tsp_path), eset(ga_path)
    common, ch_only, ga_only = ch & ga, ch - ga, ga - ch
    for e in common:
        u, v = tuple(e); x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='gray', lw=0.8, alpha=0.6)
    for u, v in (tuple(e) for e in ch_only):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='green', lw=2.5, alpha=0.95)
    for u, v in (tuple(e) for e in ga_only):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='orange', lw=2.5, alpha=0.95, ls='--')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color='green',lw=3,label='Christofides'),
               Line2D([0],[0],color='orange',lw=3,ls='--',label='GA'),
               Line2D([0],[0],color='gray',lw=1,label='Commun')],
              ['Uniquement Christofides','Uniquement GA','Commun'], loc='lower left')
    _text(0.02, 0.98, _ga_vs_ch_text())
    ax.set_title(f"Comparaison GA vs Christofides — GA: {ga_best_km:.1f} km | Christofides: {christofides_km:.1f} km",
                 fontsize=12, pad=20)

def draw_step():
    ax.clear()
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='beige', lake_color='lightblue')

    title = ""
    odd_to_draw, edges_to_draw, matching_to_draw, eulerian_to_draw, tsp_edges = [], [], [], [], []

    if gen_mode:
        _draw_ga(ga_step); return

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
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1),
    
            family='monospace')

def on_key(event):
    global step, ga_step
    if gen_mode:
        if event.key == 'right':
            ga_step = (ga_step + 1) % GA_STEPS
            draw_step()
            plt.draw()
        elif event.key == 'left':
            ga_step = (ga_step - 1) % GA_STEPS
            draw_step()
            plt.draw()
    else:
        if event.key == 'right':
            step = (step + 1) % steps
            draw_step()
            plt.draw()
        elif event.key == 'left':
            step = (step - 1) % steps
            draw_step()
            plt.draw()

def on_toggle_gen(_):
    gen_mode_list = globals()
    gen_mode_list['gen_mode'], gen_mode_list['ga_step'] = (not gen_mode_list['gen_mode']), 0
    btn.label.set_text('Retour Christofides' if gen_mode_list['gen_mode'] else 'Génétique')
    draw_step(); plt.draw()

btn_ax = plt.axes([0.72, 0.02, 0.26, 0.06])
btn = Button(btn_ax, 'Génétique')
btn.on_clicked(on_toggle_gen)

draw_step()
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
