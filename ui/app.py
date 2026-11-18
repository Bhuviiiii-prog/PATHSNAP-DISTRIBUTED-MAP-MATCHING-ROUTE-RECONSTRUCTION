"""PathSnap evaluator UI (runs on Laptop 2, the Linux laptop).

Start:  streamlit run ui/app.py
Then type the API address of Laptop 1 in the sidebar, e.g. http://192.168.1.23:8000
The UI only talks to the API. It never reads data files or the model itself.
"""
import requests
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching and route reconstruction")

api = st.sidebar.text_input("API address (Laptop 1)", "http://127.0.0.1:8000").rstrip("/")
show_clf = st.sidebar.checkbox("Also show classifier-only route", value=False)

try:
    info = requests.get(f"{api}/datasets", timeout=5).json()
except Exception:
    st.sidebar.error(f"Cannot reach the API at {api}")
    st.stop()

names = info["datasets"]
dataset = st.sidebar.radio("Dataset", names, index=names.index(info["default"]),
                           format_func=lambda n: {"evaluator_test": "Evaluator trips",
                                                  "student_test": "Student-test trips"}.get(n, n))
trips = requests.get(f"{api}/trips", params={"dataset": dataset}, timeout=5).json()["trips"]
st.sidebar.success(f"Connected: {len(trips)} trips in {dataset}")

if st.session_state.get("result", {}).get("dataset") not in (None, dataset):
    st.session_state.pop("result")          # switching dataset clears the old map

trip_id = st.selectbox("Choose a trip", trips)
if st.button("Reconstruct route", type="primary"):
    with st.spinner("Sending the trip to the API..."):
        resp = requests.post(f"{api}/predict", json={"trip_id": trip_id, "dataset": dataset}, timeout=300)
    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()
    st.session_state["result"] = resp.json()

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (HMM)", pct(m["top1_hmm"]),
              None if m["top1_hmm"] is None else f"{100 * (m['top1_hmm'] - m['top1_classifier']):+.1f} pts vs classifier only")
    c2.metric("Route continuity (HMM)", pct(m["continuity_hmm"]))
    c3.metric("GPS points", f"{res['n_points']:,}")
    c4.metric("API time", f"{res['seconds']['total']:.2f} s")
    st.caption(f"Dataset {res['dataset']}, trip {res['trip_id']}. Classifier only: accuracy {pct(m['top1_classifier'])}, continuity {pct(m['continuity_classifier'])}. "
               "Accuracy is computed on the API server; the true route never leaves the server.")

    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14)
    pts = folium.FeatureGroup("GPS points (blue)")
    for p in gps:
        folium.CircleMarker(p, radius=1.5, color="blue").add_to(pts)
    pts.add_to(fmap)
    if show_clf:
        clf = folium.FeatureGroup("Classifier only (orange)")
        for line in res["route_classifier"]:
            folium.PolyLine(line, color="orange", weight=3).add_to(clf)
        clf.add_to(fmap)
    hmm = folium.FeatureGroup("Predicted route, HMM (red)")
    for line in res["route_hmm"]:
        folium.PolyLine(line, color="red", weight=4).add_to(hmm)
    hmm.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st_folium(fmap, height=560, use_container_width=True, returned_objects=[])
