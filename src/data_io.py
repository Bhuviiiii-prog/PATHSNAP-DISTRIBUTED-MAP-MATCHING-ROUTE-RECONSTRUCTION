"""Reading the raw trip files.

Every trip folder (e.g. data/raw/Student folder/train/00000001/) holds four
whitespace-separated text files with no header:
  .track  lon lat t        (one GPS fix per row, t in seconds)
  .nodes  lon lat          (row index = node_id)
  .arcs   from_node to_node (row index = arc_id)
  .route  arc_id           (ground-truth arcs in driving order, NOT one per GPS row)
"""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw", "Student folder")
PARTITIONS = ["train", "student_test"]


def list_trips(partition):
    """Trip ids (folder names) of one partition, sorted."""
    folder = os.path.join(RAW_DIR, partition)
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def trip_path(partition, trip_id, ext):
    return os.path.join(RAW_DIR, partition, trip_id, f"{trip_id}.{ext}")


def _read(path, names, dtype):
    # sep=r"\s+" because the files mix tabs/spaces; header=None because there is none.
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_track(partition, trip_id):
    return _read(trip_path(partition, trip_id, "track"), ["lon", "lat", "t"], float)


def load_nodes(partition, trip_id):
    return _read(trip_path(partition, trip_id, "nodes"), ["lon", "lat"], float)


def load_arcs(partition, trip_id):
    return _read(trip_path(partition, trip_id, "arcs"), ["u", "v"], int)


def load_route(partition, trip_id):
    """Ground-truth arc ids in order. Only used for labels / evaluation, never as a model input."""
    return _read(trip_path(partition, trip_id, "route"), ["arc"], int)["arc"].to_numpy()


def load_trip(partition, trip_id, with_route=True):
    trip = {
        "id": trip_id,
        "partition": partition,
        "track": load_track(partition, trip_id),
        "nodes": load_nodes(partition, trip_id),
        "arcs": load_arcs(partition, trip_id),
    }
    if with_route:
        trip["route"] = load_route(partition, trip_id)
    return trip
