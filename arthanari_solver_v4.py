"""
Arthanari Pedigree Polytope Membership Solver — v4 (Corrected FAT-based MCF)
=============================================================================
Key fix: At each layer, solve the FAT (transportation) problem as an LP
instead of doing a greedy proportional split. This properly determines:
1. Whether the flow is feasible (all demand met)
2. Which arcs are rigid (frozen flow)
3. The correct flow distribution for the next layer

Algorithm per the paper:
  - R_k = set of rigid pedigrees at layer k (flow is frozen)
  - FAT_k = transportation problem from origins (rigid pedigrees) to destinations
  - MCF(k) checks the N&S condition: z* = z_max
"""

import numpy as np
from itertools import permutations
from scipy.optimize import linprog
import time
import hashlib
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Pedigree Foundations (same as v3)
# ============================================================

def hamiltonian_cycle_to_pedigree(tour, n):
    if n < 3: return None
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
    if len(pedigree) != n - 2: return False, f"Expected {n-2} triangles"
    if tuple(sorted(pedigree[0])) != (1, 2, 3): return False, "Base must be {1,2,3}"
    common_edges = []
    for idx in range(1, len(pedigree)):
        i, j = pedigree[idx][0], pedigree[idx][1]
        common_edges.append((min(i, j), max(i, j)))
    if len(set(common_edges)) != len(common_edges): return False, "Common edges not distinct"
    for idx in range(1, len(pedigree)):
        i, j, k = pedigree[idx]
        ce = (min(i, j), max(i, j))
        has_gen = False
        for prev_idx in range(idx):
            pt = tuple(sorted(pedigree[prev_idx]))
            edges = [(min(a,b), max(a,b)) for a,b in [(pt[0],pt[1]),(pt[0],pt[2]),(pt[1],pt[2])]]
            if ce in edges: has_gen = True; break
        if not has_gen: return False, f"Triangle {pedigree[idx]} has no generator"
    return True, "Valid pedigree"


def build_all_triangles(n):
    triangles = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            for k in range(j+1, n+1):
                if (i,j,k) != (1,2,3): triangles.append((i,j,k))
    ti = {t: idx for idx, t in enumerate(triangles)}
    return triangles, ti


def pedigree_to_characteristic_vector(pedigree, n):
    triangles, ti = build_all_triangles(n)
    x = np.zeros(len(triangles))
    for tri in pedigree:
        ts = tuple(sorted(tri))
        if ts in ti: x[ti[ts]] = 1.0
    return x, ti


def is_in_PMI(X, n, ti):
    for k in range(4, n+1):
        ls = sum(X[idx] for (i,j,kk), idx in ti.items() if kk == k)
        if abs(ls - 1.0) > 1e-6: return False, f"Layer {k} sum = {ls:.8f}"
    if np.any(X < -1e-9): return False, "Negative component"
    return True, "X ∈ PMI(n)"


def pedigree_partial_tour_edges(pedigree_prefix):
    """Compute edges of partial tour after pedigree prefix."""
    edges = {(1,2), (2,3), (1,3)}
    for tri in pedigree_prefix[1:]:
        i, j, k = tri
        ce = (min(i,j), max(i,j))
        if ce in edges:
            edges.discard(ce)
            edges.add((min(i,k), max(i,k)))
            edges.add((min(j,k), max(j,k)))
        else:
            return None
    return edges


# ============================================================
# PART 2: LP-based Membership (definitive, for validation)
# ============================================================

def enumerate_all_pedigrees(n):
    pedigrees = []
    seen = set()
    for perm in permutations(range(2, n+1)):
        tour = (1,) + perm
        if tour[1] > tour[-1]: continue
        if tour in seen: continue
        seen.add(tour)
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        if ped is not None:
            v, _ = validate_pedigree(ped, n)
            if v: pedigrees.append(ped)
    return pedigrees


def check_membership_LP(X, n, ti, all_ped=None):
    if all_ped is None: all_ped = enumerate_all_pedigrees(n)
    if not all_ped: return False, {"msg": "No pedigrees"}
    tau_n = len(X); num_ped = len(all_ped)
    A_cols = np.zeros((tau_n, num_ped))
    for p_idx, ped in enumerate(all_ped):
        chi, _ = pedigree_to_characteristic_vector(ped, n)
        A_cols[:, p_idx] = chi
    c = np.zeros(num_ped)
    A_eq = np.vstack([A_cols, np.ones((1, num_ped))])
    b_eq = np.concatenate([X, [1.0]])
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0,None)]*num_ped,
                     method='highs', options={'presolve': True, 'time_limit': 30})
        if res.success: return True, {"msg": "LP feasible", "lambda": res.x}
        else: return False, {"msg": f"LP infeasible: {res.message}"}
    except Exception as e:
        return False, {"msg": f"LP error: {str(e)[:100]}"}


# ============================================================
# PART 3: Corrected MCF-Based Membership Check
# ============================================================

def solve_FAT_LP(rigid_pedigrees, dest_edges, k):
    """
    Solve the FAT (Flow Across Triangles) problem at layer k as a transportation LP.
    
    Origins: rigid pedigrees from layer k-1, each with weight (supply)
    Destinations: common edges at layer k, each with weight (demand)
    Arcs: origin → destination if destination edge is in origin's partial tour
    
    Returns: (feasible, flow_dict, rigid_arcs)
    """
    # Build origin-destination-arc structure
    origins = list(rigid_pedigrees.keys())
    origin_weights = [rigid_pedigrees[o] for o in origins]
    
    destinations = list(dest_edges.keys())
    dest_weights = [dest_edges[d] for d in destinations]
    
    n_orig = len(origins)
    n_dest = len(destinations)
    
    if n_orig == 0 and n_dest > 0:
        return False, {}, {}
    if n_dest == 0:
        return True, {}, {}
    
    # Compute available edges for each origin
    origin_available = []
    for ped_prefix in origins:
        edges = pedigree_partial_tour_edges(ped_prefix)
        if edges is None:
            origin_available.append(set())
        else:
            origin_available.append(edges)
    
    # Build arc set: (orig_idx, dest_idx) pairs where destination edge is available
    arcs = []
    for oi in range(n_orig):
        for di in range(n_dest):
            if destinations[di] in origin_available[oi]:
                arcs.append((oi, di))
    
    if not arcs:
        return False, {}, {}
    
    # LP: variables f_{oi,di} for each arc
    # minimize 0 (feasibility)
    # subject to:
    #   sum_di f_{oi,di} = origin_weights[oi]  for each oi (supply)
    #   sum_oi f_{oi,di} = dest_weights[di]     for each di (demand)
    #   f_{oi,di} >= 0
    
    num_vars = len(arcs)
    arc_idx = {(oi, di): idx for idx, (oi, di) in enumerate(arcs)}
    
    c = np.zeros(num_vars)
    
    # Equality constraints
    # Supply constraints: sum_di f_{oi,di} = origin_weights[oi]
    A_eq_rows = []
    b_eq_rows = []
    
    for oi in range(n_orig):
        row = np.zeros(num_vars)
        for di in range(n_dest):
            if (oi, di) in arc_idx:
                row[arc_idx[(oi, di)]] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(origin_weights[oi])
    
    # Demand constraints: sum_oi f_{oi,di} = dest_weights[di]
    for di in range(n_dest):
        row = np.zeros(num_vars)
        for oi in range(n_orig):
            if (oi, di) in arc_idx:
                row[arc_idx[(oi, di)]] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(dest_weights[di])
    
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_rows)
    
    bounds = [(0, None)] * num_vars
    
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 10})
        
        if not res.success:
            return False, {}, {}
        
        # Extract flow
        flow = {}
        for (oi, di), idx in arc_idx.items():
            f_val = res.x[idx]
            if f_val > 1e-10:
                flow[(origins[oi], destinations[di])] = f_val
        
        # Identify rigid arcs: arcs where flow is the ONLY option
        # An arc is rigid if:
        #   1. The origin has only one outgoing arc with positive flow, OR
        #   2. The destination has only one incoming arc with positive flow
        
        rigid_arcs = set()
        
        # Check origins with single outgoing flow
        for oi in range(n_orig):
            outgoing = [(di, res.x[arc_idx[(oi, di)]]) for di in range(n_dest) if (oi, di) in arc_idx and res.x[arc_idx[(oi, di)]] > 1e-10]
            if len(outgoing) == 1:
                rigid_arcs.add((oi, outgoing[0][0]))
        
        # Check destinations with single incoming flow
        for di in range(n_dest):
            incoming = [(oi, res.x[arc_idx[(oi, di)]]) for oi in range(n_orig) if (oi, di) in arc_idx and res.x[arc_idx[(oi, di)]] > 1e-10]
            if len(incoming) == 1:
                rigid_arcs.add((incoming[0][0], di))
        
        return True, flow, rigid_arcs
        
    except Exception as e:
        return False, {}, {}


def check_membership_MCF(X, n, ti):
    """
    Corrected MCF-based membership check.
    
    At each layer k:
    1. Build FAT problem: rigid pedigrees (origins) → destination edges
    2. Solve FAT as transportation LP
    3. If infeasible → X ∉ conv(P_n)
    4. Update rigid pedigree set for next layer
    5. Continue until all layers checked
    
    Returns: (is_member, details)
    """
    details = {'n': n, 'pmi_check': None, 'layers': [], 'final_result': None}
    
    # PMI check
    pmi_ok, pmi_msg = is_in_PMI(X, n, ti)
    details['pmi_check'] = (pmi_ok, pmi_msg)
    if not pmi_ok:
        return False, details
    
    # Initialize: at layer 3, only the base pedigree
    # Rigid pedigrees: dict mapping pedigree_prefix (tuple) → weight
    R = {((1,2,3),): 1.0}
    
    for k in range(4, n + 1):
        layer_info = {'k': k, 'feasible': None, 'num_rigid_prev': len(R),
                      'num_rigid_new': 0, 'fat_arcs': 0}
        
        # Destination edges at layer k: common edges with positive X weight
        dest_edges = {}
        for (i, j, kk), idx in ti.items():
            if kk == k and X[idx] > 1e-10:
                ce = (min(i, j), max(i, j))
                dest_edges[ce] = dest_edges.get(ce, 0) + X[idx]
        
        # Solve FAT
        feasible, flow, rigid_arcs = solve_FAT_LP(R, dest_edges, k)
        
        layer_info['fat_feasible'] = feasible
        layer_info['fat_arcs'] = len(flow)
        layer_info['num_dest'] = len(dest_edges)
        
        if not feasible:
            layer_info['feasible'] = False
            # Provide diagnostic info
            layer_info['origins'] = len(R)
            layer_info['destinations'] = dest_edges
            # Check which origins can't serve any destination
            for ped_prefix, weight in R.items():
                avail = pedigree_partial_tour_edges(ped_prefix)
                if avail is None:
                    layer_info['error'] = f"Invalid pedigree: {ped_prefix}"
                    break
                servable = [e for e in dest_edges if e in avail]
                if not servable:
                    layer_info['error'] = (f"Pedigree {ped_prefix} (w={weight:.4f}) "
                                          f"cannot extend. Avail: {avail}, Need: {list(dest_edges.keys())}")
                    break
            else:
                layer_info['error'] = "FAT LP infeasible (transportation problem has no solution)"
            details['layers'].append(layer_info)
            return False, details
        
        # Build new rigid pedigree set
        new_R = {}
        
        for (ped_prefix, dest_edge), f_val in flow.items():
            if f_val > 1e-10:
                # Create new pedigree prefix by extending with this destination
                new_ped = ped_prefix + ((dest_edge[0], dest_edge[1], k),)
                new_R[new_ped] = new_R.get(new_ped, 0) + f_val
        
        # Check: total weight should be preserved
        total_new = sum(new_R.values())
        total_old = sum(R.values())
        if abs(total_new - total_old) > 1e-5:
            layer_info['feasible'] = False
            layer_info['error'] = f"Weight not preserved: {total_old:.6f} → {total_new:.6f}"
            details['layers'].append(layer_info)
            return False, details
        
        # Check: each new pedigree must be valid (distinct common edges)
        for new_ped, weight in new_R.items():
            common_edges = []
            for tri in new_ped[1:]:
                i, j = tri[0], tri[1]
                common_edges.append((min(i, j), max(i, j)))
            if len(set(common_edges)) != len(common_edges):
                # This pedigree has duplicate common edges — invalid!
                layer_info['feasible'] = False
                dup = [e for e in common_edges if common_edges.count(e) > 1]
                layer_info['error'] = (f"Invalid pedigree {new_ped}: "
                                      f"duplicate common edges {set(dup)}")
                details['layers'].append(layer_info)
                return False, details
        
        layer_info['feasible'] = True
        layer_info['num_rigid_new'] = len(new_R)
        
        R = new_R
        details['layers'].append(layer_info)
    
    # All layers passed
    details['final_result'] = True
    return True, details


# ============================================================
# PART 4: SAT → M3P Reduction (same as v3)
# ============================================================

def sat_to_3sat(clauses, num_vars):
    new_clauses = []; aux = num_vars + 1
    for clause in clauses:
        k = len(clause)
        if k <= 3: new_clauses.append(clause)
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
    n_clauses = len(clauses)
    if n_clauses == 0:
        nn = max(3, num_vars + 2)
        adj = np.ones((nn, nn), dtype=int); np.fill_diagonal(adj, 0)
        return adj, nn
    nc = 0; start = nc; nc += 1; end = nc; nc += 1
    var_nodes = {}
    for v in range(1, num_vars + 1):
        t = nc; nc += 1; f = nc; nc += 1
        var_nodes[v] = (t, f)
    clause_nodes = {}
    for c in range(n_clauses):
        clause_nodes[c] = nc; nc += 1
    nn = nc; adj = np.zeros((nn, nn), dtype=int)
    adj[start][var_nodes[1][0]] = 1; adj[start][var_nodes[1][1]] = 1
    for v in range(1, num_vars):
        for a in range(2):
            for b in range(2):
                adj[var_nodes[v][a]][var_nodes[v+1][b]] = 1
    adj[var_nodes[num_vars][0]][end] = 1; adj[var_nodes[num_vars][1]][end] = 1
    adj[end][start] = 1
    for c_idx, clause in enumerate(clauses):
        cn = clause_nodes[c_idx]
        for lit in clause[:3]:
            var = abs(lit)
            if var > num_vars: continue
            if lit > 0: adj[var_nodes[var][0]][cn] = 1; adj[cn][var_nodes[var][0]] = 1
            else: adj[var_nodes[var][1]][cn] = 1; adj[cn][var_nodes[var][1]] = 1
    return adj, nn

def hc_to_stsp(adj, nn):
    dist = np.ones((nn, nn)); np.fill_diagonal(dist, 0)
    for i in range(nn):
        for j in range(nn):
            if adj[i][j]: dist[i][j] = dist[j][i] = 0
    return dist, nn

def stsp_to_m3p(dist, n_cities):
    n = n_cities; triangles, ti = build_all_triangles(n); tau_n = len(triangles)
    c_obj = np.zeros(tau_n)
    for tri in triangles:
        i,j,k = tri; c_obj[ti[tri]] = dist[i-1][k-1] + dist[j-1][k-1] - dist[i-1][j-1]
    nl = n - 3; A_eq = np.zeros((nl, tau_n)); b_eq = np.ones(nl)
    for row, k in enumerate(range(4, n+1)):
        for tri in triangles:
            if tri[2] == k: A_eq[row, ti[tri]] = 1.0
    try:
        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0,1)]*tau_n, method='highs')
        X = np.maximum(res.x, 0) if res.success else None
    except: X = None
    if X is None:
        X = np.zeros(tau_n)
        for k in range(4, n+1):
            lt = [t for t in triangles if t[2]==k]
            for t in lt: X[ti[t]] = 1.0/len(lt)
    for k in range(4, n+1):
        ls = sum(X[ti[t]] for t in triangles if t[2]==k)
        if ls > 1e-10:
            for t in triangles:
                if t[2]==k: X[ti[t]] /= ls
    return X, n, ti

def sat_to_m3p(clauses, num_vars):
    c3, nv = sat_to_3sat(clauses, num_vars)
    adj, nn = three_sat_to_hc(c3, nv)
    dist, nc = hc_to_stsp(adj, nn)
    X, n, ti = stsp_to_m3p(dist, nc)
    return X, n, ti, c3


# ============================================================
# PART 5: Instance Generators (same)
# ============================================================

def generate_random_sat(nv, nc, seed=None):
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(nc):
        vs = rng.choice(range(1,nv+1), size=min(3,nv), replace=False)
        clauses.append([int(v)*(1 if rng.random()>.5 else -1) for v in vs])
    return clauses, nv

def generate_sha3_sat(nv, nc, seed_phrase="pinhole"):
    h = hashlib.sha3_256(seed_phrase.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
    clauses = []
    for i in range(0, nv-3, 4):
        a,b,c,d = i+1,i+2,i+3,min(i+4,nv)
        p = (h[i%32]>>(i%8))&1
        if p==0: clauses.extend([[a,b,c],[a,-b,-c],[-a,b,-c],[-a,-b,c]])
        else: clauses.extend([[a,b,-c],[a,-b,c],[-a,b,c],[-a,-b,-c]])
    for i in range(0, nv-2, 3):
        a,b,c = i+1,i+2,min(i+3,nv)
        clauses.extend([[-b,c,a],[b,-c,-a],[a,b,c]])
    rc = int.from_bytes(h[:8], 'big')
    for i in range(min(nv, 64)):
        if (rc>>(i%64))&1: clauses.append([i+1])
    while len(clauses) < nc:
        vs = rng.choice(range(1,nv+1), size=min(3,nv), replace=False)
        clauses.append([int(v)*(1 if rng.random()>.5 else -1) for v in vs])
    return clauses[:nc], nv

def generate_pigeonhole_sat(nh):
    np_ = nh+1; nv = np_*nh; pv = lambda i,j: (i-1)*nh+j
    clauses = []
    for i in range(1, np_+1): clauses.append([pv(i,j) for j in range(1, nh+1)])
    for j in range(1, nh+1):
        for i1 in range(1, np_+1):
            for i2 in range(i1+1, np_+1):
                clauses.append([-pv(i1,j), -pv(i2,j)])
    return clauses, nv

def generate_xorsat_chain(nv, cl, seed=42):
    rng = np.random.RandomState(seed); clauses = []
    for _ in range(cl):
        vs = rng.choice(range(1,nv+1), size=3, replace=False)
        p = rng.randint(0,2); a,b,c = vs
        if p==0: clauses.extend([[a,b,c],[a,-b,-c],[-a,b,-c],[-a,-b,c]])
        else: clauses.extend([[a,b,-c],[a,-b,c],[-a,b,c],[-a,-b,-c]])
    return clauses, nv

def generate_tseitin_sat(nv_, seed=42):
    G = nx.random_regular_graph(3, nv_, seed=seed)
    e2v = {}; nv = 0
    for u,v in G.edges(): e2v[(min(u,v),max(u,v))] = nv+1; nv += 1
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
    rng = np.random.RandomState(seed) if seed is not None else None
    perm = list(range(2, n+1))
    if rng: rng.shuffle(perm)
    else: np.random.shuffle(perm)
    tour = [1] + perm
    ped = hamiltonian_cycle_to_pedigree(tour, n)
    X, ti = pedigree_to_characteristic_vector(ped, n)
    return X, n, ti, tour

def generate_convex_combination_pedigree(n, num_tours=3, seed=42):
    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(num_tours))
    triangles, ti = build_all_triangles(n); X = np.zeros(len(triangles))
    for t in range(num_tours):
        perm = list(range(2, n+1)); rng.shuffle(perm)
        tour = [1] + perm; ped = hamiltonian_cycle_to_pedigree(tour, n)
        Xt, _ = pedigree_to_characteristic_vector(ped, n); X += weights[t] * Xt
    for k in range(4, n+1):
        ls = sum(X[ti[t]] for t in triangles if t[2]==k)
        if ls > 1e-10:
            for t in triangles:
                if t[2]==k: X[ti[t]] /= ls
    return X, n, ti

def generate_non_member_point(n, seed=42):
    triangles, ti = build_all_triangles(n); X = np.zeros(len(triangles))
    for k in range(4, n+1):
        if (1,2,k) in ti: X[ti[(1,2,k)]] = 1.0
    return X, n, ti

def generate_barycenter_point(n):
    triangles, ti = build_all_triangles(n); X = np.zeros(len(triangles))
    for k in range(4, n+1):
        lt = [t for t in triangles if t[2]==k]
        for t in lt: X[ti[t]] = 1.0/len(lt)
    return X, n, ti

def generate_random_pmi_point(n, seed=42):
    rng = np.random.RandomState(seed)
    triangles, ti = build_all_triangles(n); X = np.zeros(len(triangles))
    for k in range(4, n+1):
        lt = [t for t in triangles if t[2]==k]
        w = rng.dirichlet(np.ones(len(lt)))
        for i, t in enumerate(lt): X[ti[t]] = w[i]
    return X, n, ti


# ============================================================
# PART 7: Test Suite
# ============================================================

def run_pedigree_validation():
    print("=" * 80)
    print("TEST 1: PEDIGREE CONSTRUCTION & VALIDATION")
    print("=" * 80)
    tours = [(5,[1,2,3,4,5]),(5,[1,3,5,2,4]),(6,[1,2,3,4,5,6]),
             (7,[1,2,4,6,3,5,7]),(8,[1,5,2,6,3,7,4,8]),
             (10,[1,6,2,7,3,8,4,9,5,10])]
    for n,tour in tours:
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        v,msg = validate_pedigree(ped, n)
        X,ti = pedigree_to_characteristic_vector(ped, n)
        pmi,_ = is_in_PMI(X, n, ti)
        ce = [f"{{{ped[i][0]},{ped[i][1]}}}" for i in range(1,len(ped))]
        print(f"  n={n:2d} tour={str(tour):30s} valid={v} PMI={pmi} edges={ce}")
    print("  --- Random tours ---")
    for n in [5,6,8,10,12,15]:
        X,_,ti,tour = generate_valid_pedigree_point(n, seed=42)
        ped = hamiltonian_cycle_to_pedigree(tour, n)
        v,_ = validate_pedigree(ped, n); pmi,_ = is_in_PMI(X, n, ti)
        print(f"  n={n:2d} valid={v} PMI={pmi}")


def run_lp_vs_mcf_validation():
    """Cross-validate MCF vs LP on small instances."""
    print("\n" + "=" * 80)
    print("TEST 2: LP vs MCF CROSS-VALIDATION")
    print("=" * 80)
    
    for n in range(5, 8):
        all_ped = enumerate_all_pedigrees(n)
        print(f"\n  n={n}, |P_n|={len(all_ped)}")
        
        test_points = []
        for seed in range(5):
            X,_,ti,_ = generate_valid_pedigree_point(n, seed=seed)
            test_points.append(('ValidPed', X, ti))
        for seed in range(5):
            X,_,ti = generate_convex_combination_pedigree(n, seed=seed)
            test_points.append(('ConvComb', X, ti))
        X,_,ti = generate_barycenter_point(n)
        test_points.append(('Bary', X, ti))
        X,_,ti = generate_non_member_point(n)
        test_points.append(('NonMemb', X, ti))
        for seed in range(10):
            X,_,ti = generate_random_pmi_point(n, seed=seed)
            test_points.append(('RandPMI', X, ti))
        
        matches = 0; mismatches = 0; total = 0
        for label, X, ti in test_points:
            lp_in, _ = check_membership_LP(X, n, ti, all_ped)
            mcf_in, det = check_membership_MCF(X, n, ti)
            total += 1
            if lp_in == mcf_in:
                matches += 1
            else:
                mismatches += 1
                fail_info = ""
                for li in det.get('layers', []):
                    if not li['feasible']:
                        fail_info = f" (MCF fail k={li['k']}: {li.get('error','')[:60]})"
                        break
                print(f"    MISMATCH! {label}: LP={'IN' if lp_in else 'OUT'}, "
                      f"MCF={'IN' if mcf_in else 'OUT'}{fail_info}")
        
        print(f"    Agreement: {matches}/{total} ({100*matches/total:.0f}%)")


def run_mcf_membership_tests():
    print("\n" + "=" * 80)
    print("TEST 3: MCF-BASED MEMBERSHIP CHECKS")
    print("=" * 80)
    
    print(f"\n{'n':<5} | {'tau_n':<8} | {'ValidPed':<10} | {'ConvComb':<10} | "
          f"{'Barycent':<10} | {'NonMemb':<10} | {'RandPMI':<10}")
    print("-" * 75)
    
    for n in range(5, 18):
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
                X,_,ti = gen()
                is_in,_ = check_membership_MCF(X, n, ti)
                results[name] = 'IN' if is_in else 'OUT'
            except: results[name] = 'ERR'
        print(f"{n:<5} | {tau_n:<8} | {results['ValidPed']:<10} | {results['ConvComb']:<10} | "
              f"{results['Barycent']:<10} | {results['NonMemb']:<10} | {results['RandPMI']:<10}")


def run_complexity_benchmark(max_n=20):
    print("\n" + "=" * 80)
    print("TEST 4: EMPIRICAL COMPLEXITY (MCF, UNBIASED)")
    print("=" * 80)
    
    print(f"\n{'n':<5} | {'tau_n':<8} | {'IN(s)':<12} | {'OUT(s)':<12} | "
          f"{'Bary(s)':<12} | {'IN?':<5} | {'OUT?':<5} | {'Bary?':<5}")
    print("-" * 85)
    
    data = {'n':[], 't_in':[], 't_out':[], 't_bary':[]}
    prev_t = None; prev_n = None
    
    for n in range(5, max_n+1):
        tau_n = n*(n-1)*(n-2)//6 - 1
        try:
            t0 = time.perf_counter()
            X,_,ti,_ = generate_valid_pedigree_point(n, seed=42)
            r_in,_ = check_membership_MCF(X, n, ti)
            t_in = time.perf_counter() - t0
        except: t_in = float('nan'); r_in = None
        try:
            t0 = time.perf_counter()
            X,_,ti = generate_non_member_point(n)
            r_out,_ = check_membership_MCF(X, n, ti)
            t_out = time.perf_counter() - t0
        except: t_out = float('nan'); r_out = None
        try:
            t0 = time.perf_counter()
            X,_,ti = generate_barycenter_point(n)
            r_bary,_ = check_membership_MCF(X, n, ti)
            t_bary = time.perf_counter() - t0
        except: t_bary = float('nan'); r_bary = None
        
        data['n'].append(n); data['t_in'].append(t_in)
        data['t_out'].append(t_out); data['t_bary'].append(t_bary)
        
        deg = ""
        if prev_t and prev_t > 0 and t_in > 0:
            deg = f" d≈{np.log(t_in/prev_t)/np.log(n/prev_n):.1f}"
        
        print(f"{n:<5} | {tau_n:<8} | {t_in:<12.6f} | {t_out:<12.6f} | {t_bary:<12.6f} | "
              f"{'IN' if r_in else 'OUT' if r_in is not None else 'ERR':<5} | "
              f"{'IN' if r_out else 'OUT' if r_out is not None else 'ERR':<5} | "
              f"{'IN' if r_bary else 'OUT' if r_bary is not None else 'ERR':<5}{deg}")
        prev_t = t_in; prev_n = n
    
    return data


def run_sat_reduction_tests():
    print("\n" + "=" * 80)
    print("TEST 5: SAT → M3P REDUCTION CHAIN")
    print("=" * 80)
    
    print("\n--- Satisfiable 3-SAT ---")
    for clauses,nv,label in [
        ([[1,2,3],[-1,-2,3],[1,-2,-3]], 3, "simple SAT"),
        ([[1,2],[-1,3],[2,3]], 3, "2-3 mixed"),
    ]:
        X,n,ti,_ = sat_to_m3p(clauses, nv)
        is_in,_ = check_membership_MCF(X, n, ti)
        print(f"  {label}: n={n}, {'IN' if is_in else 'OUT'}")
    
    print("\n--- Unsatisfiable ---")
    for clauses,nv,label in [
        ([[1],[-1]], 1, "trivial UNSAT"),
        ([[1,2],[-1,-2],[1,-2],[-1,2]], 2, "2-var UNSAT"),
    ]:
        X,n,ti,_ = sat_to_m3p(clauses, nv)
        is_in,_ = check_membership_MCF(X, n, ti)
        print(f"  {label}: n={n}, {'IN' if is_in else 'OUT'}")
    
    print("\n--- Pigeonhole (UNSAT) ---")
    for nh in [2,3,4]:
        clauses,nv = generate_pigeonhole_sat(nh)
        try:
            X,n,ti,_ = sat_to_m3p(clauses, nv)
            is_in,_ = check_membership_MCF(X, n, ti)
            print(f"  PHP({nh}): {len(clauses)} clauses, {nv} vars, n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  PHP({nh}): ERROR {str(e)[:60]}")
    
    print("\n--- SHA-3 Pinhole ---")
    for seed in ["pinhole","BESTIALES","sha3_keccak","qu1349tnbg","avalanche"]:
        for nv in [6,10]:
            nc = nv*4; clauses,_ = generate_sha3_sat(nv, nc, seed_phrase=seed)
            try:
                t0 = time.perf_counter()
                X,n,ti,_ = sat_to_m3p(clauses, nv)
                is_in,_ = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"  SHA3('{seed}',nv={nv}): n={n}, {'IN' if is_in else 'OUT'}, {elapsed:.3f}s")
            except Exception as e:
                print(f"  SHA3('{seed}',nv={nv}): ERROR {str(e)[:60]}")
    
    print("\n--- XOR-SAT ---")
    for cl in [3,5]:
        for nv in [6,10]:
            clauses,_ = generate_xorsat_chain(nv, cl)
            try:
                X,n,ti,_ = sat_to_m3p(clauses, nv)
                is_in,_ = check_membership_MCF(X, n, ti)
                print(f"  XOR(cl={cl},nv={nv}): n={n}, {'IN' if is_in else 'OUT'}")
            except Exception as e:
                print(f"  XOR(cl={cl},nv={nv}): ERROR {str(e)[:50]}")
    
    print("\n--- Tseitin ---")
    for nv_ in [6,8]:
        clauses,nv = generate_tseitin_sat(nv_)
        try:
            X,n,ti,_ = sat_to_m3p(clauses, nv)
            is_in,_ = check_membership_MCF(X, n, ti)
            print(f"  Tseitin({nv_}): n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  Tseitin({nv_}): ERROR {str(e)[:50]}")


def run_brutal_sha3():
    print("\n" + "=" * 80)
    print("TEST 6: BRUTAL SHA-3 PINHOLE BENCHMARK")
    print("=" * 80)
    seeds = ["pinhole","BESTIALES","sha3_keccak_f1600","qu1349tnbg",
             "hash_collision_attack","preimage_resistance","avalanche_criterion",
             "sponge_construction","theta_rho_pi_chi_iota","keccak_p_1600_24r"]
    print(f"\n{'Seed':<28} | {'nv':<5} | {'nc':<6} | {'n':<5} | {'Time(s)':<10} | Result")
    print("-" * 75)
    for seed in seeds:
        for nv in [6,10]:
            nc = nv*5; clauses,_ = generate_sha3_sat(nv, nc, seed_phrase=seed)
            try:
                t0 = time.perf_counter()
                X,n,ti,_ = sat_to_m3p(clauses, nv)
                is_in,_ = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"{seed:<28} | {nv:<5} | {nc:<6} | {n:<5} | {elapsed:<10.4f} | {'IN' if is_in else 'OUT'}")
            except Exception as e:
                print(f"{seed:<28} | {nv:<5} | {nc:<6} | ?     | ERR       | {str(e)[:30]}")


def run_stress_test():
    print("\n" + "=" * 80)
    print("TEST 7: STRESS TEST — EXTREME INSTANCES")
    print("=" * 80)
    
    print("\n--- Phase Transition 3-SAT (α ≈ 4.267) ---")
    for nv in [5,8,10]:
        nc = int(nv * 4.267)
        for trial in range(3):
            clauses,_ = generate_random_sat(nv, nc, seed=42+trial)
            try:
                t0 = time.perf_counter()
                X,n,ti,_ = sat_to_m3p(clauses, nv)
                is_in,_ = check_membership_MCF(X, n, ti)
                elapsed = time.perf_counter() - t0
                print(f"  nv={nv:2d} nc={nc:3d} trial={trial}: n={n}, {'IN' if is_in else 'OUT'}, {elapsed:.3f}s")
            except Exception as e:
                print(f"  nv={nv:2d} nc={nc:3d} trial={trial}: ERR {str(e)[:40]}")
    
    print("\n--- Overconstrained (α=10) ---")
    for nv in [4,6,8]:
        nc = nv*10; clauses,_ = generate_random_sat(nv, nc, seed=999)
        try:
            X,n,ti,_ = sat_to_m3p(clauses, nv)
            is_in,_ = check_membership_MCF(X, n, ti)
            print(f"  nv={nv} nc={nc}: n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  nv={nv} nc={nc}: ERR {str(e)[:40]}")
    
    print("\n--- Underconstrained (α=1) ---")
    for nv in [5,10]:
        nc = max(1, nv); clauses,_ = generate_random_sat(nv, nc, seed=777)
        try:
            X,n,ti,_ = sat_to_m3p(clauses, nv)
            is_in,_ = check_membership_MCF(X, n, ti)
            print(f"  nv={nv} nc={nc}: n={n}, {'IN' if is_in else 'OUT'}")
        except Exception as e:
            print(f"  nv={nv} nc={nc}: ERR {str(e)[:40]}")


def run_dantzig_42():
    print("\n" + "=" * 80)
    print("TEST 8: DANTZIG 42-CITY (paper Section 4.6.2)")
    print("=" * 80)
    n = 42; tau_n = n*(n-1)*(n-2)//6 - 1
    print(f"  n={n}, tau_n={tau_n}")
    
    print(f"\n  Valid tour:")
    try:
        t0 = time.perf_counter()
        X,_,ti,tour = generate_valid_pedigree_point(n, seed=42)
        is_in,det = check_membership_MCF(X, n, ti)
        elapsed = time.perf_counter() - t0
        print(f"    Result: {'IN' if is_in else 'OUT'}, Time: {elapsed:.2f}s")
        for li in det.get('layers', [])[:3]:
            print(f"    k={li['k']}: feasible={li['feasible']}, rigid={li['num_rigid_new']}")
    except Exception as e:
        print(f"    ERROR: {e}")
    
    print(f"\n  Non-member (duplicate edge {{1,2}}):")
    try:
        t0 = time.perf_counter()
        X,_,ti = generate_non_member_point(n)
        is_in,det = check_membership_MCF(X, n, ti)
        elapsed = time.perf_counter() - t0
        print(f"    Result: {'IN' if is_in else 'OUT'}, Time: {elapsed:.2f}s")
        for li in det.get('layers', []):
            if not li['feasible']:
                print(f"    Failed at k={li['k']}: {li.get('error','')[:100]}")
                break
    except Exception as e:
        print(f"    ERROR: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ARTHANARI PEDIGREE POLYTOPE MEMBERSHIP SOLVER v4              ║")
    print("║  Corrected FAT-LP + LP Validation + SAT→M3P + Brutal Tests    ║")
    print("║  arXiv:2606.03194 — Claim: M3P ∈ P (strongly poly O(n^14))    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    run_pedigree_validation()
    run_lp_vs_mcf_validation()
    run_mcf_membership_tests()
    data = run_complexity_benchmark(max_n=16)
    run_sat_reduction_tests()
    run_brutal_sha3()
    run_stress_test()
    run_dantzig_42()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
