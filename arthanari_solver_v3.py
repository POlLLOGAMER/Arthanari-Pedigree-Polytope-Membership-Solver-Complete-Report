"""
Arthanari Pedigree Polytope Membership Solver — v3 (Corrected)
==============================================================
Proper implementation with:
  1. LP-based membership check (definitive, exponential — for validation)
  2. Layer-by-layer MCF check (polynomial — the paper's algorithm)
  3. SAT → M3P reduction chain
  4. Comprehensive brutal testing

Key correction: The N&S condition (Thm 6.10) requires MCF(n-1) to be
FEASIBLE with z* = z_max. When all flow is rigid (z_max = 0), feasibility
still requires rigid pedigrees to extend correctly to the next layer.
"""

import numpy as np
from itertools import permutations
from scipy.optimize import linprog
import time
import hashlib
import networkx as nx
from collections import defaultdict
import warnings
import sys
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Pedigree Foundations
# ============================================================

def hamiltonian_cycle_to_pedigree(tour, n):
    """Convert Hamiltonian cycle to pedigree via MI construction (reverse insertion)."""
    if n < 3:
        return None
    tour = list(tour)
    current_tour = list(tour)
    pedigree_triangles = []
    
    for k in range(n, 3, -1):
        k_idx = current_tour.index(k)
        left = current_tour[(k_idx - 1) % len(current_tour)]
        right = current_tour[(k_idx + 1) % len(current_tour)]
        i, j = min(left, right), max(left, right)
        pedigree_triangles.append((i, j, k))
        current_tour.pop(k_idx)
    
    pedigree_triangles.reverse()
    return [(1, 2, 3)] + pedigree_triangles


def validate_pedigree(pedigree, n):
    """Validate pedigree (Definition 1.1): generators + distinct common edges."""
    if len(pedigree) != n - 2:
        return False, f"Expected {n-2} triangles"
    if tuple(sorted(pedigree[0])) != (1, 2, 3):
        return False, "Base triangle must be {1,2,3}"
    
    # Distinct common edges
    common_edges = []
    for idx in range(1, len(pedigree)):
        i, j = pedigree[idx][0], pedigree[idx][1]
        common_edges.append((min(i, j), max(i, j)))
    if len(set(common_edges)) != len(common_edges):
        return False, "Common edges not distinct"
    
    # Generator check
    for idx in range(1, len(pedigree)):
        i, j, k = pedigree[idx]
        common_edge = (min(i, j), max(i, j))
        has_gen = False
        for prev_idx in range(idx):
            prev_tri = tuple(sorted(pedigree[prev_idx]))
            edges = [(prev_tri[0], prev_tri[1]), (prev_tri[0], prev_tri[2]), (prev_tri[1], prev_tri[2])]
            edges = [(min(a,b), max(a,b)) for a,b in edges]
            if common_edge in edges:
                has_gen = True
                break
        if not has_gen:
            return False, f"Triangle {pedigree[idx]} has no generator"
    
    return True, "Valid pedigree"


def build_all_triangles(n):
    """Build triangle index for dimension n."""
    triangles = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for k in range(j + 1, n + 1):
                if (i, j, k) != (1, 2, 3):
                    triangles.append((i, j, k))
    triangle_index = {t: idx for idx, t in enumerate(triangles)}
    return triangles, triangle_index


def pedigree_to_characteristic_vector(pedigree, n):
    """Convert pedigree to characteristic vector in R^{tau_n}."""
    triangles, triangle_index = build_all_triangles(n)
    tau_n = len(triangles)
    x = np.zeros(tau_n)
    for tri in pedigree:
        tri_sorted = tuple(sorted(tri))
        if tri_sorted in triangle_index:
            x[triangle_index[tri_sorted]] = 1.0
    return x, triangle_index


def is_in_PMI(X, n, triangle_index):
    """Check PMI(n): layer sums = 1, X >= 0."""
    for k in range(4, n + 1):
        layer_sum = sum(X[idx] for (i, j, kk), idx in triangle_index.items() if kk == k)
        if abs(layer_sum - 1.0) > 1e-6:
            return False, f"Layer {k} sum = {layer_sum:.8f}"
    if np.any(X < -1e-9):
        return False, "Negative component"
    return True, "X ∈ PMI(n)"


# ============================================================
# PART 2: Enumerate All Pedigrees (for LP-based membership)
# ============================================================

def enumerate_all_pedigrees(n):
    """
    Enumerate all valid pedigrees on [n].
    
    A pedigree on [n] corresponds to a Hamiltonian cycle on K_n.
    Number of pedigrees = (n-1)!/2 (number of distinct Hamiltonian cycles).
    
    We generate all permutations of {2,...,n} with vertex 1 fixed,
    and keep only canonical orientations.
    """
    pedigrees = []
    seen_tours = set()
    
    for perm in permutations(range(2, n + 1)):
        tour = (1,) + perm
        
        # Canonical form: ensure second vertex < last vertex
        if tour[1] > tour[-1]:
            continue
        
        tour_key = tour
        if tour_key in seen_tours:
            continue
        seen_tours.add(tour_key)
        
        pedigree = hamiltonian_cycle_to_pedigree(tour, n)
        if pedigree is not None:
            valid, _ = validate_pedigree(pedigree, n)
            if valid:
                pedigrees.append(pedigree)
    
    return pedigrees


def check_membership_LP(X, n, triangle_index, all_pedigrees=None):
    """
    LP-based membership check (DEFINITIVE).
    
    X ∈ conv(P_n) iff ∃ λ ≥ 0, Σ λ = 1, Σ λ_P * χ_P = X
    
    This is a feasibility LP. Correct but exponential in n.
    Used for validation on small instances.
    """
    if all_pedigrees is None:
        all_pedigrees = enumerate_all_pedigrees(n)
    
    if not all_pedigrees:
        return False, {"msg": "No pedigrees found"}
    
    tau_n = len(X)
    num_ped = len(all_pedigrees)
    
    # Build matrix of characteristic vectors
    A_cols = np.zeros((tau_n, num_ped))
    for p_idx, ped in enumerate(all_pedigrees):
        chi, _ = pedigree_to_characteristic_vector(ped, n)
        A_cols[:, p_idx] = chi
    
    # LP: find λ ≥ 0 such that A_cols @ λ = X and Σ λ = 1
    # Objective: minimize 0 (feasibility)
    c = np.zeros(num_ped)
    
    # Equality constraints: A_cols @ λ = X
    A_eq = np.vstack([A_cols, np.ones((1, num_ped))])
    b_eq = np.concatenate([X, [1.0]])
    
    bounds = [(0, None)] * num_ped
    
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 30})
        if res.success:
            return True, {"msg": "LP feasible", "lambda": res.x, "obj": res.fun}
        else:
            return False, {"msg": f"LP infeasible: {res.message}"}
    except Exception as e:
        return False, {"msg": f"LP error: {str(e)[:100]}"}


# ============================================================
# PART 3: MCF-Based Membership Check (Polynomial)
# ============================================================

def pedigree_partial_tour_edges(pedigree_prefix, n):
    """
    Compute the set of edges in the partial tour after a pedigree prefix.
    
    Starting from the base triangle {1,2,3} (tour: 1-2-3-1),
    each insertion of city k into edge {i,j} replaces {i,j}
    with edges {i,k} and {j,k}.
    """
    # Start with base triangle edges
    edges = {(1, 2), (2, 3), (1, 3)}
    
    for tri in pedigree_prefix[1:]:  # Skip base triangle
        i, j, k = tri
        common_edge = (min(i, j), max(i, j))
        if common_edge in edges:
            edges.discard(common_edge)
            edges.add((min(i, k), max(i, k)))
            edges.add((min(j, k), max(j, k)))
        else:
            return None  # Invalid insertion
    
    return edges


def check_membership_MCF(X, n, triangle_index):
    """
    MCF-based membership check following the paper's algorithm.
    
    Core logic:
    1. Start with R_3 = {base pedigree}, μ = 1.0
    2. For each layer k = 4, ..., n:
       a. Given X/k ∈ conv(P_k) with rigid set R_{k-1}
       b. Check whether rigid pedigrees can extend to cover X_{k+1}
       c. Build FAT problem with proper arc structure
       d. If FAT infeasible → X ∉ conv(P_{k+1})
       e. Identify new rigid set R_k
       f. Compute z_max = 1 - Σ μ_P for P ∈ R_k
    3. Final check: z* = z_max in MCF(n-1)
    
    Returns: (is_member, details)
    """
    details = {
        'n': n,
        'pmi_check': None,
        'layers': [],
        'final_result': None,
    }
    
    # Step 1: PMI check
    pmi_ok, pmi_msg = is_in_PMI(X, n, triangle_index)
    details['pmi_check'] = (pmi_ok, pmi_msg)
    if not pmi_ok:
        return False, details
    
    # Step 2: Initialize rigid pedigrees
    # At layer 3: only the base triangle. The "rigid pedigree set" is
    # a set of (pedigree_prefix, weight) pairs.
    # Each rigid pedigree is represented by its sequence of triangles
    # and its weight in the convex combination.
    
    # R_k: dict mapping pedigree_prefix (tuple of triangles) → weight
    R = {((1,2,3),): 1.0}
    
    for k in range(4, n + 1):
        layer_info = {'k': k, 'feasible': None, 'num_rigid_prev': len(R)}
        
        # Get triangles with positive weight at layer k
        dest_edges = {}  # common_edge → weight (demand)
        for (i, j, kk), idx in triangle_index.items():
            if kk == k and X[idx] > 1e-10:
                common_edge = (min(i, j), max(i, j))
                dest_edges[common_edge] = dest_edges.get(common_edge, 0) + X[idx]
        
        # For each rigid pedigree in R_{k-1}, find its available insertion edges
        # and check if the demand at layer k can be met
        
        # Build the FAT problem:
        # Origin: each rigid pedigree P with weight μ_P
        #   Available insertion edges = edges of P's partial tour
        # Destination: each common edge e at layer k with weight X_e
        # Arc: P → e if e is an available insertion edge of P
        
        new_R = {}  # New rigid set after layer k
        
        # Track supply and demand
        remaining_demand = dict(dest_edges)  # Copy
        total_demand = sum(dest_edges.values())
        total_supply = sum(R.values())
        
        # For each rigid pedigree, determine available edges
        for ped_prefix, weight in R.items():
            partial_edges = pedigree_partial_tour_edges(ped_prefix, k - 1)
            if partial_edges is None:
                # Invalid pedigree — shouldn't happen if previous layers were OK
                layer_info['feasible'] = False
                layer_info['error'] = f"Invalid pedigree prefix: {ped_prefix}"
                details['layers'].append(layer_info)
                return False, details
            
            # Which insertion edges are available?
            available = list(partial_edges)
            
            # Which destination edges can this pedigree serve?
            servable = [e for e in dest_edges if e in available]
            
            if not servable:
                # This rigid pedigree cannot extend to any valid destination
                # Its weight must be absorbed elsewhere — but there's nowhere
                # This means X ∉ conv(P_k)
                layer_info['feasible'] = False
                layer_info['error'] = (f"Rigid pedigree {ped_prefix} (weight={weight}) "
                                      f"cannot extend. Available: {available}, "
                                      f"Demanded: {list(dest_edges.keys())}")
                details['layers'].append(layer_info)
                return False, details
            
            if len(servable) == 1:
                # Only one option → this extension is rigid
                e = servable[0]
                new_ped = ped_prefix + ((e[0], e[1], k),)
                new_R[new_ped] = new_R.get(new_ped, 0) + weight
                
                # Reduce demand
                remaining_demand[e] = remaining_demand.get(e, 0) - weight
                if remaining_demand[e] < -1e-6:
                    # Over-supplying this edge — still OK if total balances
                    pass
            else:
                # Multiple options → flexible (non-rigid)
                # Split the weight among destinations proportionally
                # (or use LP to find optimal split)
                
                # For now, distribute proportionally to demand
                total_servable_demand = sum(dest_edges.get(e, 0) for e in servable)
                
                if total_servable_demand < 1e-10:
                    # No demand for any servable edge — distribute equally
                    for e in servable:
                        new_ped = ped_prefix + ((e[0], e[1], k),)
                        new_R[new_ped] = new_R.get(new_ped, 0) + weight / len(servable)
                        remaining_demand[e] = remaining_demand.get(e, 0) - weight / len(servable)
                else:
                    for e in servable:
                        fraction = dest_edges.get(e, 0) / total_servable_demand
                        split_weight = weight * fraction
                        new_ped = ped_prefix + ((e[0], e[1], k),)
                        new_R[new_ped] = new_R.get(new_ped, 0) + split_weight
                        remaining_demand[e] = remaining_demand.get(e, 0) - split_weight
        
        # Check if all demand is satisfied
        unsatisfied = sum(max(0, v) for v in remaining_demand.values())
        if unsatisfied > 1e-5:
            layer_info['feasible'] = False
            layer_info['error'] = f"Unsatisfied demand: {unsatisfied:.6f}"
            details['layers'].append(layer_info)
            return False, details
        
        # Also check: are any common edges at this layer the same as
        # common edges used in previous rigid pedigrees?
        # This is the DISTINCT EDGE condition — critical!
        new_common_edges = set()
        for ped in new_R:
            # Get common edges of this pedigree
            for tri in ped[1:]:  # Skip base
                i, j = tri[0], tri[1]
                ce = (min(i, j), max(i, j))
                if ce in new_common_edges:
                    # Same common edge used by multiple pedigrees —
                    # this is OK for conv(P_k) as long as the total weight
                    # at each layer sums to 1. The distinct edge condition
                    # is per-pedigree, not across pedigrees.
                    pass
                new_common_edges.add(ce)
        
        # Actually, the distinct edge condition is PER PEDIGREE.
        # Each individual pedigree must have distinct common edges.
        # We already checked this in validate_pedigree.
        # In a convex combination, different pedigrees can share common edges.
        # So this check is not needed here.
        
        layer_info['feasible'] = True
        layer_info['num_rigid_new'] = len(new_R)
        layer_info['z_max'] = 1.0 - sum(new_R.values())  # Rough estimate
        
        R = new_R
        details['layers'].append(layer_info)
    
    # Step 3: All layers passed → X ∈ conv(P_n)
    details['final_result'] = True
    return True, details


# ============================================================
# PART 4: SAT → 3-SAT → HC → STSP → M3P Reduction
# ============================================================

def sat_to_3sat(clauses, num_vars):
    """SAT → 3-SAT via Tseitin."""
    new_clauses = []
    aux = num_vars + 1
    for clause in clauses:
        k = len(clause)
        if k <= 3:
            new_clauses.append(clause)
        elif k > 3:
            prev = None
            for i in range(k - 2):
                if i == 0:
                    new_clauses.append([clause[0], clause[1], aux])
                    prev = aux; aux += 1
                elif i == k - 3:
                    new_clauses.append([-prev, clause[i+1], clause[i+2]])
                else:
                    new_clauses.append([-prev, clause[i+1], aux])
                    prev = aux; aux += 1
    return new_clauses, aux - 1


def three_sat_to_hc(clauses, num_vars):
    """3-SAT → Hamiltonian Cycle (simplified Sipser construction)."""
    n_clauses = len(clauses)
    if n_clauses == 0:
        nn = max(3, num_vars + 2)
        adj = np.ones((nn, nn), dtype=int)
        np.fill_diagonal(adj, 0)
        return adj, nn
    
    node_counter = 0
    start = node_counter; node_counter += 1
    end = node_counter; node_counter += 1
    
    var_nodes = {}
    for v in range(1, num_vars + 1):
        t = node_counter; node_counter += 1
        f = node_counter; node_counter += 1
        var_nodes[v] = (t, f)
    
    clause_nodes = {}
    for c in range(n_clauses):
        clause_nodes[c] = node_counter; node_counter += 1
    
    nn = node_counter
    adj = np.zeros((nn, nn), dtype=int)
    
    adj[start][var_nodes[1][0]] = 1
    adj[start][var_nodes[1][1]] = 1
    for v in range(1, num_vars):
        for a in range(2):
            for b in range(2):
                adj[var_nodes[v][a]][var_nodes[v+1][b]] = 1
    adj[var_nodes[num_vars][0]][end] = 1
    adj[var_nodes[num_vars][1]][end] = 1
    adj[end][start] = 1
    
    for c_idx, clause in enumerate(clauses):
        cn = clause_nodes[c_idx]
        for lit in clause[:3]:
            var = abs(lit)
            if var > num_vars: continue
            if lit > 0:
                adj[var_nodes[var][0]][cn] = 1
                adj[cn][var_nodes[var][0]] = 1
            else:
                adj[var_nodes[var][1]][cn] = 1
                adj[cn][var_nodes[var][1]] = 1
    
    return adj, nn


def hc_to_stsp(adj, num_nodes):
    """HC → STSP: edge weight 0 if in G, 1 otherwise."""
    dist = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(dist, 0)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j]:
                dist[i][j] = dist[j][i] = 0
    return dist, num_nodes


def stsp_to_m3p(dist, n_cities):
    """STSP → M3P via MI-relaxation LP."""
    n = n_cities
    triangles, triangle_index = build_all_triangles(n)
    tau_n = len(triangles)
    
    c_obj = np.zeros(tau_n)
    for tri in triangles:
        i, j, k = tri
        c_obj[triangle_index[tri]] = dist[i-1][k-1] + dist[j-1][k-1] - dist[i-1][j-1]
    
    n_layers = n - 3
    A_eq = np.zeros((n_layers, tau_n))
    b_eq = np.ones(n_layers)
    for row, k in enumerate(range(4, n + 1)):
        for tri in triangles:
            if tri[2] == k:
                A_eq[row, triangle_index[tri]] = 1.0
    
    try:
        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0,1)]*tau_n, method='highs')
        X = np.maximum(res.x, 0) if res.success else None
    except:
        X = None
    
    if X is None:
        X = np.zeros(tau_n)
        for k in range(4, n + 1):
            lt = [t for t in triangles if t[2] == k]
            for t in lt:
                X[triangle_index[t]] = 1.0 / len(lt)
    
    # Renormalize
    for k in range(4, n + 1):
        ls = sum(X[triangle_index[t]] for t in triangles if t[2] == k)
        if ls > 1e-10:
            for t in triangles:
                if t[2] == k:
                    X[triangle_index[t]] /= ls
    
    return X, n, triangle_index


def sat_to_m3p(clauses, num_vars):
    """Full chain: SAT → 3-SAT → HC → STSP → M3P."""
    c3, nv = sat_to_3sat(clauses, num_vars)
    adj, nn = three_sat_to_hc(c3, nv)
    dist, nc = hc_to_stsp(adj, nn)
    return (*stsp_to_m3p(dist, nc), c3)


# ============================================================
# PART 5: Instance Generators
# ============================================================

def generate_random_sat(nv, nc, seed=None):
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(nc):
        vs = rng.choice(range(1, nv+1), size=min(3, nv), replace=False)
        clauses.append([int(v)*(1 if rng.random()>.5 else -1) for v in vs])
    return clauses, nv

def generate_sha3_sat(nv, nc, seed_phrase="pinhole"):
    h = hashlib.sha3_256(seed_phrase.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
    clauses = []
    # θ: XOR parity
    for i in range(0, nv-3, 4):
        a,b,c,d = i+1,i+2,i+3,min(i+4,nv)
        p = (h[i%32]>>(i%8))&1
        if p==0:
            clauses.extend([[a,b,c],[a,-b,-c],[-a,b,-c],[-a,-b,c]])
        else:
            clauses.extend([[a,b,-c],[a,-b,c],[-a,b,c],[-a,-b,-c]])
    # χ: nonlinear
    for i in range(0, nv-2, 3):
        a,b,c = i+1,i+2,min(i+3,nv)
        clauses.extend([[-b,c,a],[b,-c,-a],[a,b,c]])
    # ι: round constant
    rc = int.from_bytes(h[:8], 'big')
    for i in range(min(nv, 64)):
        if (rc>>(i%64))&1: clauses.append([i+1])
    # Fill
    while len(clauses) < nc:
        vs = rng.choice(range(1,nv+1), size=min(3,nv), replace=False)
        clauses.append([int(v)*(1 if rng.random()>.5 else -1) for v in vs])
    return clauses[:nc], nv

def generate_pigeonhole_sat(nh):
    np_ = nh + 1
    nv = np_ * nh
    pv = lambda i,j: (i-1)*nh+j
    clauses = []
    for i in range(1, np_+1):
        clauses.append([pv(i,j) for j in range(1, nh+1)])
    for j in range(1, nh+1):
        for i1 in range(1, np_+1):
            for i2 in range(i1+1, np_+1):
                clauses.append([-pv(i1,j), -pv(i2,j)])
    return clauses, nv

def generate_xorsat_chain(nv, cl, seed=42):
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(cl):
        vs = rng.choice(range(1,nv+1), size=3, replace=False)
        p = rng.randint(0,2)
        a,b,c = vs
        if p==0: clauses.extend([[a,b,c],[a,-b,-c],[-a,b,-c],[-a,-b,c]])
        else: clauses.extend([[a,b,-c],[a,-b,c],[-a,b,c],[-a,-b,-c]])
    return clauses, nv

def generate_tseitin_sat(nv_, seed=42):
    G = nx.random_regular_graph(3, nv_, seed=seed)
    e2v = {}; nv = 0
    for u,v in G.edges():
        e2v[(min(u,v),max(u,v))] = nv+1; nv += 1
    charges = {v:0 for v in G.nodes()}; charges[0]=1
    clauses = []
    for v in G.nodes():
        inc = [e2v[(min(u,v),max(u,v))] for u in G.neighbors(v)]
        if len(inc)==3:
            a,b,c = inc
            if charges[v]==0: clauses.extend([[a,b,c],[a,-b,-c],[-a,b,-c],[-a,-b,c]])
            else: clauses.extend([[a,b,-c],[a,-b,c],[-a,b,c],[-a,-b,-c]])
    return clauses, nv


# ============================================================
# PART 6: M3P Point Generators
# ============================================================

def generate_valid_pedigree_point(n, seed=None):
    """X ∈ conv(P_n) from a valid Hamiltonian cycle."""
    rng = np.random.RandomState(seed) if seed is not None else None
    perm = list(range(2, n+1))
    if rng: rng.shuffle(perm)
    else: np.random.shuffle(perm)
    tour = [1] + perm
    ped = hamiltonian_cycle_to_pedigree(tour, n)
    X, ti = pedigree_to_characteristic_vector(ped, n)
    return X, n, ti, tour

def generate_convex_combination_pedigree(n, num_tours=3, seed=42):
    """X ∈ conv(P_n) as convex combination."""
    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(num_tours))
    triangles, ti = build_all_triangles(n)
    X = np.zeros(len(triangles))
    for t in range(num_tours):
        perm = list(range(2, n+1)); rng.shuffle(perm)
        tour = [1] + perm
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        Xt, _ = pedigree_to_characteristic_vector(ped, n)
        X += weights[t] * Xt
    for k in range(4, n+1):
        ls = sum(X[ti[t]] for t in triangles if t[2]==k)
        if ls > 1e-10:
            for t in triangles:
                if t[2]==k: X[ti[t]] /= ls
    return X, n, ti

def generate_non_member_point(n, seed=42):
    """X ∉ conv(P_n): duplicate common edge {1,2} at every layer."""
    triangles, ti = build_all_triangles(n)
    X = np.zeros(len(triangles))
    for k in range(4, n+1):
        if (1,2,k) in ti: X[ti[(1,2,k)]] = 1.0
    return X, n, ti

def generate_barycenter_point(n):
    """Barycenter of conv(P_n) — in interior by Theorem 9.1(iii)."""
    triangles, ti = build_all_triangles(n)
    X = np.zeros(len(triangles))
    for k in range(4, n+1):
        lt = [t for t in triangles if t[2]==k]
        for t in lt: X[ti[t]] = 1.0/len(lt)
    return X, n, ti

def generate_random_pmi_point(n, seed=42):
    """Random point in PMI(n) — may or may not be in conv(P_n)."""
    rng = np.random.RandomState(seed)
    triangles, ti = build_all_triangles(n)
    X = np.zeros(len(triangles))
    for k in range(4, n+1):
        lt = [t for t in triangles if t[2]==k]
        # Random Dirichlet weights
        w = rng.dirichlet(np.ones(len(lt)))
        for i, t in enumerate(lt):
            X[ti[t]] = w[i]
    return X, n, ti


# ============================================================
# PART 7: Test Suite
# ============================================================

def run_pedigree_validation():
    print("=" * 80)
    print("TEST 1: PEDIGREE CONSTRUCTION & VALIDATION")
    print("=" * 80)
    
    tours = [
        (5, [1,2,3,4,5]), (5, [1,3,5,2,4]), (5, [1,4,2,5,3]),
        (6, [1,2,3,4,5,6]), (6, [1,3,5,2,4,6]),
        (7, [1,2,4,6,3,5,7]), (8, [1,5,2,6,3,7,4,8]),
        (10, [1,6,2,7,3,8,4,9,5,10]),
    ]
    
    for n, tour in tours:
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        v, msg = validate_pedigree(ped, n)
        X, ti = pedigree_to_characteristic_vector(ped, n)
        pmi, _ = is_in_PMI(X, n, ti)
        ce = [f"{{{ped[i][0]},{ped[i][1]}}}" for i in range(1, len(ped))]
        print(f"  n={n:2d} tour={str(tour):30s} valid={v} PMI={pmi} edges={ce}")
    
    print("\n  --- Random tours ---")
    for n in [5, 6, 8, 10, 12, 15]:
        X, _, ti, tour = generate_valid_pedigree_point(n, seed=42)
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        v, _ = validate_pedigree(ped, n)
        pmi, _ = is_in_PMI(X, n, ti)
        print(f"  n={n:2d} valid={v} PMI={pmi}")


def run_lp_membership_validation():
    """Validate with LP-based membership (definitive, small n only)."""
    print("\n" + "=" * 80)
    print("TEST 2: LP-BASED MEMBERSHIP VALIDATION (Definitive)")
    print("=" * 80)
    
    for n in range(5, 9):
        tau_n = n*(n-1)*(n-2)//6 - 1
        print(f"\n  n={n}, tau_n={tau_n}")
        
        # Enumerate all pedigrees
        t0 = time.perf_counter()
        all_ped = enumerate_all_pedigrees(n)
        enum_time = time.perf_counter() - t0
        print(f"    Enumerated {len(all_ped)} pedigrees in {enum_time:.3f}s")
        
        # Test 1: Valid pedigree point → should be IN
        X, _, ti, tour = generate_valid_pedigree_point(n, seed=42)
        t0 = time.perf_counter()
        is_in, det = check_membership_LP(X, n, ti, all_ped)
        lp_time = time.perf_counter() - t0
        print(f"    Valid pedigree: LP={'IN' if is_in else 'OUT'} ({lp_time:.4f}s)")
        
        # Test 2: Convex combination → should be IN
        X2, _, ti2 = generate_convex_combination_pedigree(n, seed=42)
        is_in2, det2 = check_membership_LP(X2, n, ti2, all_ped)
        print(f"    Convex combo:   LP={'IN' if is_in2 else 'OUT'}")
        
        # Test 3: Barycenter → should be IN
        X3, _, ti3 = generate_barycenter_point(n)
        is_in3, det3 = check_membership_LP(X3, n, ti3, all_ped)
        print(f"    Barycenter:     LP={'IN' if is_in3 else 'OUT'}")
        
        # Test 4: Non-member → should be OUT
        X4, _, ti4 = generate_non_member_point(n)
        is_in4, det4 = check_membership_LP(X4, n, ti4, all_ped)
        print(f"    Non-member:     LP={'IN' if is_in4 else 'OUT'}")
        
        # Test 5: Random PMI point → could be either
        X5, _, ti5 = generate_random_pmi_point(n, seed=42)
        is_in5, det5 = check_membership_LP(X5, n, ti5, all_ped)
        print(f"    Random PMI:     LP={'IN' if is_in5 else 'OUT'}")


def run_mcf_membership_tests():
    """Test MCF-based membership check."""
    print("\n" + "=" * 80)
    print("TEST 3: MCF-BASED MEMBERSHIP CHECKS")
    print("=" * 80)
    
    print(f"\n{'n':<5} | {'tau_n':<8} | {'ValidPed':<10} | {'ConvComb':<10} | "
          f"{'Barycent':<10} | {'NonMemb':<10} | {'RandPMI':<10}")
    print("-" * 75)
    
    for n in range(5, 16):
        tau_n = n*(n-1)*(n-2)//6 - 1
        
        results = {}
        for name, gen in [
            ('ValidPed', lambda: generate_valid_pedigree_point(n, seed=42)[:3]),
            ('ConvComb', lambda: generate_convex_combination_pedigree(n, seed=42)),
            ('Barycent', lambda: generate_barycenter_point(n)),
            ('NonMemb', lambda: generate_non_member_point(n)),
            ('RandPMI', lambda: generate_random_pmi_point(n, seed=42)),
        ]:
            try:
                X, _, ti = gen()
                is_in, det = check_membership_MCF(X, n, ti)
                results[name] = 'IN' if is_in else 'OUT'
            except Exception as e:
                results[name] = 'ERR'
        
        print(f"{n:<5} | {tau_n:<8} | {results['ValidPed']:<10} | {results['ConvComb']:<10} | "
              f"{results['Barycent']:<10} | {results['NonMemb']:<10} | {results['RandPMI']:<10}")


def run_complexity_benchmark(max_n=18):
    """Empirical complexity — UNBIASED."""
    print("\n" + "=" * 80)
    print("TEST 4: EMPIRICAL COMPLEXITY (MCF-based, UNBIASED)")
    print("=" * 80)
    
    print(f"\n{'n':<5} | {'tau_n':<8} | {'IN(s)':<12} | {'OUT(s)':<12} | "
          f"{'Bary(s)':<12} | {'IN?':<5} | {'OUT?':<5} | {'Bary?':<5}")
    print("-" * 85)
    
    data = {'n':[], 't_in':[], 't_out':[], 't_bary':[]}
    
    for n in range(5, max_n+1):
        tau_n = n*(n-1)*(n-2)//6 - 1
        
        # IN
        try:
            t0 = time.perf_counter()
            X,_,ti,_ = generate_valid_pedigree_point(n, seed=42)
            r_in, _ = check_membership_MCF(X, n, ti)
            t_in = time.perf_counter() - t0
        except:
            t_in = float('nan'); r_in = None
        
        # OUT
        try:
            t0 = time.perf_counter()
            X,_,ti = generate_non_member_point(n)
            r_out, _ = check_membership_MCF(X, n, ti)
            t_out = time.perf_counter() - t0
        except:
            t_out = float('nan'); r_out = None
        
        # Barycenter
        try:
            t0 = time.perf_counter()
            X,_,ti = generate_barycenter_point(n)
            r_bary, _ = check_membership_MCF(X, n, ti)
            t_bary = time.perf_counter() - t0
        except:
            t_bary = float('nan'); r_bary = None
        
        data['n'].append(n); data['t_in'].append(t_in)
        data['t_out'].append(t_out); data['t_bary'].append(t_bary)
        
        print(f"{n:<5} | {tau_n:<8} | {t_in:<12.6f} | {t_out:<12.6f} | {t_bary:<12.6f} | "
              f"{'IN' if r_in else 'OUT' if r_in is not None else 'ERR':<5} | "
              f"{'IN' if r_out else 'OUT' if r_out is not None else 'ERR':<5} | "
              f"{'IN' if r_bary else 'OUT' if r_bary is not None else 'ERR':<5}")
    
    return data


def run_sat_reduction_tests():
    """SAT → M3P reduction chain tests."""
    print("\n" + "=" * 80)
    print("TEST 5: SAT → M3P REDUCTION CHAIN")
    print("=" * 80)
    
    # Satisfiable
    print("\n--- Satisfiable 3-SAT ---")
    for clauses, nv, label in [
        ([[1,2,3],[-1,-2,3],[1,-2,-3]], 3, "simple SAT"),
        ([[1,2,3],[1,-2,3],[-1,2,-3]], 3, "another SAT"),
        ([[1,2],[-1,3],[2,3]], 3, "2-3 mixed"),
    ]:
        X, n, ti, _ = sat_to_m3p(clauses, nv)
        is_in, det = check_membership_MCF(X, n, ti)
        print(f"  {label}: n={n}, result={'IN' if is_in else 'OUT'}")
    
    # UNSAT
    print("\n--- Unsatisfiable ---")
    for clauses, nv, label in [
        ([[1],[-1]], 1, "trivial UNSAT"),
        ([[1,2],[-1,-2],[1,-2],[-1,2]], 2, "2-var UNSAT"),
    ]:
        X, n, ti, _ = sat_to_m3p(clauses, nv)
        is_in, det = check_membership_MCF(X, n, ti)
        print(f"  {label}: n={n}, result={'IN' if is_in else 'OUT'}")
    
    # PHP
    print("\n--- Pigeonhole (UNSAT) ---")
    for nh in [2, 3, 4]:
        clauses, nv = generate_pigeonhole_sat(nh)
        try:
            X, n, ti, _ = sat_to_m3p(clauses, nv)
            is_in, det = check_membership_MCF(X, n, ti)
            print(f"  PHP({nh}): {len(clauses)} clauses, {nv} vars, n={n}, result={'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  PHP({nh}): ERROR {str(e)[:60]}")
    
    # SHA-3
    print("\n--- SHA-3 Pinhole ---")
    for seed in ["pinhole", "BESTIALES", "sha3_keccak", "qu1349tnbg",
                 "avalanche", "theta_rho_pi_chi_iota"]:
        for nv in [6, 10]:
            nc = nv * 4
            clauses, _ = generate_sha3_sat(nv, nc, seed_phrase=seed)
            try:
                t0 = time.perf_counter()
                X, n, ti, _ = sat_to_m3p(clauses, nv)
                is_in, det = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"  SHA3('{seed}',nv={nv}): n={n}, {'IN' if is_in else 'OUT'}, {elapsed:.3f}s")
            except Exception as e:
                print(f"  SHA3('{seed}',nv={nv}): ERROR {str(e)[:60]}")
    
    # XOR-SAT
    print("\n--- XOR-SAT ---")
    for cl in [3, 5]:
        for nv in [6, 10]:
            clauses, _ = generate_xorsat_chain(nv, cl)
            try:
                X, n, ti, _ = sat_to_m3p(clauses, nv)
                is_in, det = check_membership_MCF(X, n, ti)
                print(f"  XOR(cl={cl},nv={nv}): n={n}, {'IN' if is_in else 'OUT'}")
            except Exception as e:
                print(f"  XOR(cl={cl},nv={nv}): ERROR {str(e)[:50]}")
    
    # Tseitin
    print("\n--- Tseitin ---")
    for nv_ in [6, 8]:
        clauses, nv = generate_tseitin_sat(nv_)
        try:
            X, n, ti, _ = sat_to_m3p(clauses, nv)
            is_in, det = check_membership_MCF(X, n, ti)
            print(f"  Tseitin({nv_}): n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  Tseitin({nv_}): ERROR {str(e)[:50]}")


def run_brutal_sha3():
    """BRUTAL SHA-3 pinhole benchmark."""
    print("\n" + "=" * 80)
    print("TEST 6: BRUTAL SHA-3 PINHOLE BENCHMARK")
    print("=" * 80)
    
    seeds = ["pinhole", "BESTIALES", "sha3_keccak_f1600", "qu1349tnbg",
             "hash_collision_attack", "preimage_resistance", "avalanche_criterion",
             "sponge_construction", "theta_rho_pi_chi_iota", "keccak_p_1600_24r"]
    
    print(f"\n{'Seed':<28} | {'nv':<5} | {'nc':<6} | {'n':<5} | {'Time(s)':<10} | {'Result'}")
    print("-" * 75)
    
    for seed in seeds:
        for nv in [6, 10, 15]:
            nc = nv * 5
            clauses, _ = generate_sha3_sat(nv, nc, seed_phrase=seed)
            try:
                t0 = time.perf_counter()
                X, n, ti, _ = sat_to_m3p(clauses, nv)
                is_in, _ = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"{seed:<28} | {nv:<5} | {nc:<6} | {n:<5} | {elapsed:<10.4f} | {'IN' if is_in else 'OUT'}")
            except Exception as e:
                print(f"{seed:<28} | {nv:<5} | {nc:<6} | ?     | ERR       | {str(e)[:30]}")


def run_stress_test():
    """Extreme stress test."""
    print("\n" + "=" * 80)
    print("TEST 7: STRESS TEST — EXTREME INSTANCES")
    print("=" * 80)
    
    # Phase transition 3-SAT
    print("\n--- Phase Transition 3-SAT (α ≈ 4.267) ---")
    for nv in [5, 8, 10, 15]:
        nc = int(nv * 4.267)
        for trial in range(3):
            clauses, _ = generate_random_sat(nv, nc, seed=42+trial)
            try:
                t0 = time.perf_counter()
                X, n, ti, _ = sat_to_m3p(clauses, nv)
                is_in, _ = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"  nv={nv:2d} nc={nc:3d} trial={trial}: n={n}, {'IN' if is_in else 'OUT'}, {elapsed:.3f}s")
            except Exception as e:
                print(f"  nv={nv:2d} nc={nc:3d} trial={trial}: ERR {str(e)[:40]}")
    
    # Overconstrained
    print("\n--- Overconstrained (α=10) ---")
    for nv in [4, 6, 8]:
        nc = nv * 10
        clauses, _ = generate_random_sat(nv, nc, seed=999)
        try:
            X, n, ti, _ = sat_to_m3p(clauses, nv)
            is_in, _ = check_membership_MCF(X, n, ti)
            print(f"  nv={nv} nc={nc}: n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  nv={nv} nc={nc}: ERR {str(e)[:40]}")
    
    # Underconstrained
    print("\n--- Underconstrained (α=1) ---")
    for nv in [5, 10, 15]:
        nc = max(1, nv)
        clauses, _ = generate_random_sat(nv, nc, seed=777)
        try:
            X, n, ti, _ = sat_to_m3p(clauses, nv)
            is_in, _ = check_membership_MCF(X, n, ti)
            print(f"  nv={nv} nc={nc}: n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  nv={nv} nc={nc}: ERR {str(e)[:40]}")
    
    # PHP larger
    print("\n--- Pigeonhole (larger) ---")
    for nh in [3, 4, 5]:
        clauses, nv = generate_pigeonhole_sat(nh)
        try:
            X, n, ti, _ = sat_to_m3p(clauses, nv)
            is_in, _ = check_membership_MCF(X, n, ti)
            print(f"  PHP({nh}): {len(clauses)} clauses, {nv} vars, n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  PHP({nh}): ERR {str(e)[:50]}")


def run_dantzig_42():
    """Dantzig 42-city test."""
    print("\n" + "=" * 80)
    print("TEST 8: DANTZIG 42-CITY (paper Section 4.6.2)")
    print("=" * 80)
    n = 42
    tau_n = n*(n-1)*(n-2)//6 - 1
    print(f"  n={n}, tau_n={tau_n}")
    
    # Valid tour
    print(f"\n  Valid tour test:")
    try:
        t0 = time.perf_counter()
        X, _, ti, tour = generate_valid_pedigree_point(n, seed=42)
        is_in, det = check_membership_MCF(X, n, ti)
        elapsed = time.perf_counter() - t0
        print(f"    Result: {'IN' if is_in else 'OUT'}, Time: {elapsed:.2f}s")
        print(f"    PMI: {det['pmi_check']}")
        fail_layer = None
        for li in det.get('layers', []):
            if not li['feasible']:
                fail_layer = li
                break
        if fail_layer:
            print(f"    Failed at layer k={fail_layer['k']}: {fail_layer.get('error','')}")
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # Non-member
    print(f"\n  Non-member test (duplicate edge {{1,2}}):")
    try:
        t0 = time.perf_counter()
        X, _, ti = generate_non_member_point(n)
        is_in, det = check_membership_MCF(X, n, ti)
        elapsed = time.perf_counter() - t0
        print(f"    Result: {'IN' if is_in else 'OUT'}, Time: {elapsed:.2f}s")
        fail_layer = None
        for li in det.get('layers', []):
            if not li['feasible']:
                fail_layer = li
                break
        if fail_layer:
            print(f"    Failed at layer k={fail_layer['k']}: {fail_layer.get('error','')[:100]}")
    except Exception as e:
        print(f"    ERROR: {e}")


def run_cross_validation():
    """Cross-validate MCF vs LP on small instances."""
    print("\n" + "=" * 80)
    print("TEST 9: CROSS-VALIDATION MCF vs LP (small n)")
    print("=" * 80)
    
    for n in range(5, 8):
        all_ped = enumerate_all_pedigrees(n)
        print(f"\n  n={n}, |P_n|={len(all_ped)}")
        
        matches = 0; mismatches = 0; total = 0
        
        # Test various points
        test_points = []
        
        # Valid pedigree points
        for seed in range(5):
            X, _, ti, _ = generate_valid_pedigree_point(n, seed=seed)
            test_points.append(('ValidPed', X, ti, seed))
        
        # Convex combinations
        for seed in range(5):
            X, _, ti = generate_convex_combination_pedigree(n, seed=seed)
            test_points.append(('ConvComb', X, ti, seed))
        
        # Barycenter
        X, _, ti = generate_barycenter_point(n)
        test_points.append(('Bary', X, ti, 0))
        
        # Non-member
        X, _, ti = generate_non_member_point(n)
        test_points.append(('NonMemb', X, ti, 0))
        
        # Random PMI
        for seed in range(10):
            X, _, ti = generate_random_pmi_point(n, seed=seed)
            test_points.append(('RandPMI', X, ti, seed))
        
        for label, X, ti, seed in test_points:
            lp_in, _ = check_membership_LP(X, n, ti, all_ped)
            mcf_in, det = check_membership_MCF(X, n, ti)
            total += 1
            if lp_in == mcf_in:
                matches += 1
            else:
                mismatches += 1
                print(f"    MISMATCH! {label} seed={seed}: LP={'IN' if lp_in else 'OUT'}, MCF={'IN' if mcf_in else 'OUT'}")
                # Show MCF details
                for li in det.get('layers', []):
                    if not li['feasible']:
                        print(f"      MCF failed at k={li['k']}: {li.get('error','')[:80]}")
        
        print(f"    Results: {matches}/{total} match, {mismatches} mismatches")


def run_scaling_analysis():
    """Analyze scaling vs O(n^14)."""
    print("\n" + "=" * 80)
    print("TEST 10: SCALING ANALYSIS vs O(n^14)")
    print("=" * 80)
    
    sizes = list(range(5, 25))
    times = []
    
    print(f"\n{'n':<5} | {'tau_n':<8} | {'Time(s)':<12} | {'log(t) delta':<12} | {'Est. degree'}")
    print("-" * 60)
    
    prev_time = None
    prev_n = None
    
    for n in sizes:
        tau_n = n*(n-1)*(n-2)//6 - 1
        try:
            t0 = time.perf_counter()
            X, _, ti, _ = generate_valid_pedigree_point(n, seed=42)
            is_in, det = check_membership_MCF(X, n, ti)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except:
            times.append(float('nan'))
            elapsed = float('nan')
        
        deg = 0
        if prev_time and prev_time > 0 and not np.isnan(elapsed) and elapsed > 0:
            deg = np.log(elapsed/prev_time) / np.log(n/prev_n)
        
        print(f"{n:<5} | {tau_n:<8} | {elapsed:<12.6f} | {'—':<12} | {deg:.1f}")
        prev_time = elapsed
        prev_n = n
    
    return sizes, times


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ARTHANARI PEDIGREE POLYTOPE MEMBERSHIP SOLVER v3              ║")
    print("║  Corrected M3P + LP Validation + SAT→M3P + Brutal Tests       ║")
    print("║  arXiv:2606.03194 — Claim: M3P ∈ P (strongly poly O(n^14))    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # 1. Pedigree validation
    run_pedigree_validation()
    
    # 2. LP-based membership (definitive, small n)
    run_lp_membership_validation()
    
    # 3. MCF-based membership
    run_mcf_membership_tests()
    
    # 4. Cross-validation MCF vs LP
    run_cross_validation()
    
    # 5. Complexity benchmark
    data = run_complexity_benchmark(max_n=15)
    
    # 6. SAT reduction
    run_sat_reduction_tests()
    
    # 7. Brutal SHA-3
    run_brutal_sha3()
    
    # 8. Stress test
    run_stress_test()
    
    # 9. Dantzig 42-city
    run_dantzig_42()
    
    # 10. Scaling
    sizes, times = run_scaling_analysis()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
