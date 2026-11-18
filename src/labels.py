"""Per-GPS-point labels from the .route file.

The .route file is a list of arcs in driving order, not one arc per GPS row.
To get a label for every GPS point we align the two sequences.

Why not just "nearest route arc" for every point?
  Routes often pass the same place twice (U-turns, loops, out-and-back) and
  run next to themselves in both directions. A plain nearest-arc lookup can
  then jump backwards and forwards along the route. The vehicle cannot do that,
  so we force the alignment to be monotonic: point i+1 may stay on the same
  route position as point i or move forward, never back.

Method (dynamic programming, like a simple DTW / Viterbi over route positions):
  D[i, k]    = distance (m) from GPS point i to route arc k, capped at CAP_M
  cost[i, k] = D[i, k] + min over k' <= k of cost[i-1, k']
  The capped distance stops one wild GPS fix from dragging the whole path.
  We back-track from the cheapest end state to read off route position k(i).
Arcs that no GPS point lands on (very short arcs passed within one second)
are simply skipped, which the "k' <= k" rule allows.
"""
import numpy as np
from .geo import arc_endpoints, point_segment

CAP_M = 100.0


def route_distance_matrix(trip):
    """D[i, k]: distance of every GPS point to every arc of the route (in route order)."""
    ax, ay, bx, by = arc_endpoints(trip)
    route = trip["route"]
    px = trip["track"]["x"].to_numpy()[:, None]
    py = trip["track"]["y"].to_numpy()[:, None]
    d, _ = point_segment(px, py, ax[route][None, :], ay[route][None, :], bx[route][None, :], by[route][None, :])
    return d


def align_monotonic(D, cap=CAP_M):
    """Returns route position k(i) for every point i (non-decreasing)."""
    n, K = D.shape
    D = np.minimum(D, cap)
    back = np.zeros((n, K), dtype=np.int32)
    cost = D[0].copy()
    idx = np.arange(K)
    for i in range(1, n):
        # best previous position k' <= k, found with a running minimum
        run_min = np.minimum.accumulate(cost)
        # index where that running minimum was reached (latest one on ties)
        run_arg = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
        back[i] = run_arg
        cost = D[i] + run_min
    pos = np.empty(n, dtype=np.int32)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos


def label_trip(trip):
    """Adds label columns to trip["track"]:
    route_pos  - position in the .route list
    label_arc  - the arc id at that position
    label_dist - distance (m) from the GPS point to that arc
    """
    D = route_distance_matrix(trip)
    pos = align_monotonic(D)
    track = trip["track"]
    track["route_pos"] = pos
    track["label_arc"] = trip["route"][pos]
    track["label_dist"] = D[np.arange(len(pos)), pos]
    return trip


def nearest_route_arc(D):
    """Plain nearest-route-arc label (no order constraint), only used to show
    how often it disagrees with the monotonic alignment."""
    return np.argmin(D, axis=1)
