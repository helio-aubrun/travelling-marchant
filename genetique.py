from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from christofides import villes, haversine  

@dataclass
class GAConfig:
    population_size: int = 400
    generations: int = 800
    tournament_k: int = 5
    crossover_rate: float = 0.95
    mutation_rate: float = 0.15
    mutation_op: str = "inversion"  
    elitism: int = 4
    patience: int = 120
    seed: Optional[int] = 42

def _build_matrix(cities: Dict[str, Tuple[float, float]]) -> tuple[List[str], List[List[float]]]:
    names = list(cities.keys()); n = len(names)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(cities[names[i]], cities[names[j]])
            mat[i][j] = mat[j][i] = d
    return names, mat

def _tour_len(t: List[int], M: List[List[float]]) -> float:
    s = sum(M[t[i]][t[i+1]] for i in range(len(t)-1))
    return s + M[t[-1]][t[0]]

def _random_tour(n: int, rng: random.Random) -> List[int]:
    t = list(range(n)); rng.shuffle(t); return t

def _tournament(pop: List[List[int]], fit: List[float], k: int, rng: random.Random) -> List[int]:
    best_idx = min((rng.randrange(len(pop)) for _ in range(k)), key=lambda i: fit[i])
    return pop[best_idx][:]

def _ox(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    n = len(p1); a, b = sorted((rng.randrange(n), rng.randrange(n)))
    child = [-1]*n; child[a:b+1] = p1[a:b+1]; used = set(child[a:b+1]); pos = (b+1)%n
    for x in p2:
        if x in used: continue
        while child[pos] != -1: pos = (pos+1)%n
        child[pos] = x
    return child

def _mut_swap(t: List[int], rng: random.Random) -> None:
    i, j = rng.randrange(len(t)), rng.randrange(len(t))
    if i != j: t[i], t[j] = t[j], t[i]

def _mut_inv(t: List[int], rng: random.Random) -> None:
    i, j = sorted((rng.randrange(len(t)), rng.randrange(len(t))))
    if i != j: t[i:j+1] = reversed(t[i:j+1])

def run_ga_tsp(cities: Dict[str, Tuple[float, float]], config: GAConfig = GAConfig()) -> tuple[List[str], float, List[float]]:
    names, M = _build_matrix(cities); n = len(names)
    rng = random.Random(config.seed); mutate = _mut_inv if config.mutation_op == "inversion" else _mut_swap

    pop = [_random_tour(n, rng) for _ in range(config.population_size)]
    fit = [_tour_len(ind, M) for ind in pop]
    best_idx = min(range(len(pop)), key=lambda i: fit[i]); best, best_val = pop[best_idx][:], fit[best_idx]
    history, stall = [best_val], 0

    for _ in range(config.generations):
        elites_idx = sorted(range(len(pop)), key=lambda i: fit[i])[:config.elitism]
        new_pop = [pop[i][:] for i in elites_idx]

        while len(new_pop) < config.population_size:
            p1 = _tournament(pop, fit, config.tournament_k, rng)
            p2 = _tournament(pop, fit, config.tournament_k, rng)
            child = _ox(p1, p2, rng) if rng.random() < config.crossover_rate else p1[:]
            if rng.random() < config.mutation_rate: mutate(child, rng)
            new_pop.append(child)

        pop = new_pop; fit = [_tour_len(ind, M) for ind in pop]
        i = min(range(len(pop)), key=lambda k: fit[k])
        if fit[i] + 1e-9 < best_val: best, best_val, stall = pop[i][:], fit[i], 0
        else: stall += 1
        history.append(best_val)
        if stall >= config.patience: break

    best_names = [names[i] for i in best] + [names[best[0]]]
    return best_names, best_val, history
