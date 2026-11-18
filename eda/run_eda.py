"""Quick EDA (both partitions, descriptive only): trip counts, rows per trip,
time-gap stats, network sizes, plus one Folium map of a train trip.
Run: .venv/bin/python eda/run_eda.py
"""
import os, sys
import numpy as np
import pandas as pd
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_io import PARTITIONS, list_trips, load_trip

OUT = os.path.dirname(os.path.abspath(__file__))


def trip_stats(trip):
    t = trip["track"]["t"].to_numpy()
    dt = np.diff(t)
    arcs, n_nodes = trip["arcs"], len(trip["nodes"])
    g = coo_matrix((np.ones(len(arcs)), (arcs.u, arcs.v)), shape=(n_nodes, n_nodes))
    n_comp, _ = connected_components(g, directed=True, connection="weak")
    deg = np.bincount(np.r_[arcs.u, arcs.v], minlength=n_nodes)
    return {
        "partition": trip["partition"], "trip": trip["id"],
        "gps_rows": len(t), "nan_values": int(trip["track"].isna().sum().sum()),
        "time_sorted": bool((dt > 0).all()), "dt_share_1s": float((dt == 1).mean()),
        "dt_max_s": float(dt.max()), "gaps_over_10s": int((dt > 10).sum()),
        "nodes": n_nodes, "arcs": len(arcs), "isolated_nodes": int((deg == 0).sum()),
        "weak_components": int(n_comp), "route_arcs": len(trip["route"]),
    }


def make_map(trip, path):
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14)
    # only draw network arcs near the track, otherwise the html gets huge
    pad = 0.003
    box = nodes.lon.between(tr.lon.min() - pad, tr.lon.max() + pad) & nodes.lat.between(tr.lat.min() - pad, tr.lat.max() + pad)
    near = arcs[box.to_numpy()[arcs.u] & box.to_numpy()[arcs.v]]
    net = folium.FeatureGroup("Road network (grey)")
    for u, v in zip(near.u, near.v):
        folium.PolyLine([(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])], color="grey", weight=1).add_to(net)
    route = folium.FeatureGroup("Reference route (green)")
    for a in trip["route"]:
        u, v = arcs.u[a], arcs.v[a]
        folium.PolyLine([(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])], color="green", weight=5).add_to(route)
    gps = folium.FeatureGroup("GPS points (blue)")
    for lat, lon in zip(tr.lat, tr.lon):
        folium.CircleMarker((lat, lon), radius=2, color="blue").add_to(gps)
    for layer in (net, route, gps):
        layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(path)


if __name__ == "__main__":
    rows = []
    for part in PARTITIONS:
        for tid in list_trips(part):
            rows.append(trip_stats(load_trip(part, tid)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "trip_inventory.csv"), index=False)
    print(df.groupby("partition").size().rename("trips"))
    summary = df.groupby("partition")[["gps_rows", "dt_share_1s", "dt_max_s", "gaps_over_10s", "nodes", "arcs",
                                       "isolated_nodes", "weak_components", "route_arcs"]].describe().T
    summary.to_csv(os.path.join(OUT, "summary_stats.csv"))
    print(summary.to_string())
    print("NaN values total:", df.nan_values.sum(), "| unsorted trips:", (~df.time_sorted).sum())
    make_map(load_trip("train", "00000003"), os.path.join(OUT, "map_train_00000003.html"))
    print("saved eda/trip_inventory.csv, eda/summary_stats.csv, eda/map_train_00000003.html")
