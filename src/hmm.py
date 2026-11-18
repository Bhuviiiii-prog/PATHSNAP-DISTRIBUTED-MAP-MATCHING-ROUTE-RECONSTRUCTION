"""HMM + Viterbi route reconstruction.

Hidden state at GPS point i = which of its K candidate arcs the car is on.
Emission   = how well the arc fits the GPS point = the classifier's probability,
             normalised over the point's K candidates.
Transition = can the car get from arc a (point i-1) to arc b (point i)?
             same arc            -> cost 0
             b starts where a ends (1 hop)  -> cost HOP
             reachable in 2 hops            -> cost 2 * HOP
             U-turn onto the reverse of the same road -> cost UTURN
               (cars rarely U-turn; without this the decoder flipped to the
               opposite-direction twin arc far too often)
             anything else (not connected)  -> cost FAR
             After a GPS gap longer than GAP_S seconds the car could be anywhere,
             so all transitions get the same cost (no constraint).
Viterbi finds the sequence of candidates with the highest total
log-probability (emission + transition), step by step, keeping the best
path into every state (dynamic programming).
"""
import numpy as np

DEFAULTS = {"HOP": 1.0, "FAR": 6.0, "UTURN": 10.0, "GAP_S": 10.0}


def road_links(arcs):
    """For each node, the set of nodes reachable in one arc (used for the 2-hop test)."""
    nxt = {}
    for u, v in zip(arcs.u.to_numpy(), arcs.v.to_numpy()):
        nxt.setdefault(u, set()).add(v)
    return nxt


def transition_cost(a_prev, b_next, arc_u, arc_v, nxt, hop, far, uturn):
    """K x K matrix of transition costs from previous candidates to next candidates."""
    k1, k2 = len(a_prev), len(b_next)
    cost = np.full((k1, k2), far)
    for i, a in enumerate(a_prev):
        end = arc_v[a]
        two_hop = nxt.get(end, set())
        for j, b in enumerate(b_next):
            if a == b:
                cost[i, j] = 0.0
            elif arc_u[b] == end and arc_v[b] == arc_u[a]:
                cost[i, j] = uturn          # A->B then B->A: a U-turn
            elif arc_u[b] == end:
                cost[i, j] = hop
            elif arc_u[b] in two_hop:
                cost[i, j] = 2 * hop
    return cost


def viterbi(prob, cand, t, arcs, params=None):
    """prob, cand: (n, K) arrays; t: (n,) timestamps. Returns (n,) chosen candidate index."""
    p = {**DEFAULTS, **(params or {})}
    arc_u, arc_v = arcs.u.to_numpy(), arcs.v.to_numpy()
    nxt = road_links(arcs)
    emit = np.log(np.clip(prob / prob.sum(1, keepdims=True), 1e-9, 1.0))
    n, k = prob.shape
    score = emit[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    for i in range(1, n):
        if t[i] - t[i - 1] > p["GAP_S"]:
            trans = np.zeros((k, k))      # long gap: no connectivity constraint
        else:
            trans = -transition_cost(cand[i - 1], cand[i], arc_u, arc_v, nxt, p["HOP"], p["FAR"], p["UTURN"])
        total = score[:, None] + trans    # rows: previous state, columns: next state
        back[i] = total.argmax(0)
        score = total.max(0) + emit[i]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def route_continuity(arcs_seq, arcs):
    """Share of consecutive points whose predicted arcs are the same arc or
    directly connected (the next arc starts where the previous one ends)."""
    arc_u, arc_v = arcs.u.to_numpy(), arcs.v.to_numpy()
    a, b = arcs_seq[:-1], arcs_seq[1:]
    ok = (a == b) | (arc_u[b] == arc_v[a])
    return float(ok.mean()) if len(ok) else 1.0
