import math
import networkx as nx

# -------------------------------
# Données : villes françaises
# -------------------------------
villes = {
    "Paris": (48.8566, 2.3522),
    "Marseille": (43.2965, 5.3698),
    "Lyon": (45.764, 4.8357),
    "Toulouse": (43.6047, 1.4442),
    "Nice": (43.7102, 7.262),
    "Nantes": (47.2184, -1.5536),
    "Strasbourg": (48.5734, 7.7521),
    "Montpellier": (43.6119, 3.8772),
    "Bordeaux": (44.8378, -0.5792),
    "Lille": (50.6292, 3.0573),
    "Rennes": (48.1173, -1.6778),
    "Reims": (49.2583, 4.0317),
    "Le Havre": (49.4944, 0.1079),
    "Saint-Étienne": (45.4397, 4.3872),
    "Toulon": (43.1242, 5.928),
    "Grenoble": (45.1885, 5.7245),
    "Dijon": (47.322, 5.0415),
    "Angers": (47.4784, -0.5632),
    "Nîmes": (43.8367, 4.3601),
    "Clermont-Ferrand": (45.7772, 3.087),
}

# -------------------------------
# Fonctions de l’algorithme
# -------------------------------

def haversine(coord1, coord2):
    """Distance Haversine entre deux coordonnées (lat, lon)."""
    R = 6371
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_complete_graph(villes):
    """Construit le graphe complet avec distances Haversine."""
    G = nx.Graph()
    for ville, (lat, lon) in villes.items():
        G.add_node(ville, pos=(lon, lat))
    for v1, (lat1, lon1) in villes.items():
        for v2, (lat2, lon2) in villes.items():
            if v1 != v2:
                dist = haversine((lat1, lon1), (lat2, lon2))
                G.add_edge(v1, v2, weight=dist)
    return G

def compute_mst(G):
    """Calcule l’arbre couvrant minimal."""
    return nx.minimum_spanning_tree(G)

def find_odd_nodes(G):
    """Renvoie la liste des sommets de degré impair."""
    return [n for n, d in G.degree() if d % 2 == 1]

def compute_minimum_matching(G, odd_nodes):
    """Apparie les sommets impairs par un matching de poids minimal."""
    odd_subgraph = G.subgraph(odd_nodes)
    matching = nx.algorithms.matching.min_weight_matching(odd_subgraph, weight='weight')
    return list(matching)

def build_eulerian_graph(mst_G, matching_edges, G):
    """Fusionne MST + matching pour créer un graphe eulérien."""
    eulerian_G = nx.MultiGraph()
    eulerian_G.add_nodes_from(mst_G.nodes())
    eulerian_G.add_edges_from(mst_G.edges())
    for u, v in matching_edges:
        eulerian_G.add_edge(u, v, weight=G[u][v]['weight'])
    return eulerian_G

def compute_tsp_path(eulerian_G):
    """Construit le circuit approximatif TSP (Christofides)."""
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_G))
    visited = set()
    tsp_path = []
    for u, v in eulerian_circuit:
        if u not in visited:
            tsp_path.append(u)
            visited.add(u)
        if v not in visited:
            tsp_path.append(v)
            visited.add(v)
    tsp_path.append(tsp_path[0])  # retour au départ
    return tsp_path
