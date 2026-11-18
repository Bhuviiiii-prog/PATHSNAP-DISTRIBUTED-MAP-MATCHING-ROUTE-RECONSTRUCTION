"""Turning lon/lat (degrees) into flat x/y in metres, plus small geometry helpers.

Why a per-trip azimuthal equidistant (AEQD) projection centred on the trip's own
GPS centroid, instead of one UTM zone or a plain equirectangular formula:
  * The 90 trips are spread over the whole world (Europe, Russia, USA, Japan,
    Vietnam, Argentina, ...). There is no single zone that fits all of them.
  * AEQD is exact for distances from its centre and the scale error grows
    only slowly with distance (well under 0.1% inside ~50 km, which covers every
    trajectory we have). We only ever measure short distances near the GPS track.
  * Unlike picking a UTM zone, there is no edge case at zone borders or at the
    equator (trip 57 sits at latitude 0.3 N; trip 71 and 98 are south of it).
"""
import numpy as np
from pyproj import Transformer


def make_projector(lon_centre, lat_centre):
    proj = f"+proj=aeqd +lat_0={lat_centre} +lon_0={lon_centre} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def project_trip(trip):
    """Adds x/y (metres) to track and nodes, using one projection per trip."""
    track, nodes = trip["track"], trip["nodes"]
    lon0, lat0 = track["lon"].mean(), track["lat"].mean()
    tf = make_projector(lon0, lat0)
    track["x"], track["y"] = tf.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    nodes["x"], nodes["y"] = tf.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    trip["centre"] = (lon0, lat0)
    return trip


def arc_endpoints(trip):
    """Arrays (ax, ay, bx, by) for every arc: start node and end node in metres."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    return nx[u], ny[u], nx[v], ny[v]


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (element-wise) and the projection
    parameter s in [0, 1] (0 = at the start node, 1 = at the end node)."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    # zero-length arcs exist in some networks; avoid dividing by zero
    s = np.where(seg_len2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(seg_len2, 1e-12), 0.0)
    s = np.clip(s, 0.0, 1.0)
    qx, qy = ax + s * dx, ay + s * dy
    return np.hypot(px - qx, py - qy), s


def bearing_deg(dx, dy):
    """Compass-style bearing in degrees [0, 360): 0 = north (+y), 90 = east (+x)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
