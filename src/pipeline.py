"""The full prediction pipeline for ONE trip: used by run_hmm.py and by the API.

raw files -> metres -> candidates + features -> classifier probabilities
          -> HMM/Viterbi -> one road arc per GPS point

The ground-truth route is only used AFTER prediction, to compute accuracy.
It never enters the features, the model or the decoder.
"""
import os
import time
import numpy as np

from .data_io import _read
from .geo import project_trip
from .features import FEATURES, candidate_features
from .labels import route_distance_matrix, align_monotonic
from .hmm import viterbi, route_continuity


def load_trip_from_folder(folder, trip_id):
    """Load a trip from any folder laid out like the dataset: <folder>/<trip_id>/<trip_id>.<ext>.
    The .route file is optional (only needed to score the prediction)."""
    base = os.path.join(folder, trip_id, trip_id)
    trip = {
        "id": trip_id,
        "track": _read(base + ".track", ["lon", "lat", "t"], float),
        "nodes": _read(base + ".nodes", ["lon", "lat"], float),
        "arcs": _read(base + ".arcs", ["u", "v"], int),
    }
    if os.path.exists(base + ".route"):
        trip["route"] = _read(base + ".route", ["arc"], int)["arc"].to_numpy()
    return trip


def predict_trip(trip, model, hmm_params=None):
    """Returns a dict with predicted arcs (classifier-only and HMM), timings and, if the
    route is available, the accuracy of both."""
    t0 = time.time()
    project_trip(trip)
    rows, cand = candidate_features(trip)
    t_feat = time.time() - t0

    t1 = time.time()
    prob = model.predict_proba(rows[FEATURES])[:, 1].reshape(cand.shape)
    t_model = time.time() - t1

    t2 = time.time()
    n = len(cand)
    clf_arcs = cand[np.arange(n), prob.argmax(1)]
    hmm_arcs = cand[np.arange(n), viterbi(prob, cand, trip["track"]["t"].to_numpy(), trip["arcs"], hmm_params)]
    t_hmm = time.time() - t2

    out = {
        "trip_id": trip["id"], "n_points": int(n),
        "clf_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
        "continuity_clf": route_continuity(clf_arcs, trip["arcs"]),
        "continuity_hmm": route_continuity(hmm_arcs, trip["arcs"]),
        "seconds": {"features": t_feat, "model": t_model, "hmm": t_hmm, "total": time.time() - t0},
    }
    # scoring only: the route is used after all predictions are made
    if "route" in trip:
        label = trip["route"][align_monotonic(route_distance_matrix(trip))]
        out["top1_clf"] = float((clf_arcs == label).mean())
        out["top1_hmm"] = float((hmm_arcs == label).mean())
    return out


def arcs_to_lines(arc_seq, trip):
    """Turn a per-point arc sequence into map lines [[lat, lon], [lat, lon]], one per arc visit."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    keep = np.r_[True, arc_seq[1:] != arc_seq[:-1]]
    lines = []
    for a in arc_seq[keep]:
        u, v = arcs.u[a], arcs.v[a]
        lines.append([[float(nodes.lat[u]), float(nodes.lon[u])], [float(nodes.lat[v]), float(nodes.lon[v])]])
    return lines
