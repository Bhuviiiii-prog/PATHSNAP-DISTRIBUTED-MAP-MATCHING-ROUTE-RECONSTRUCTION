# Decisions (one line each, with the reason)

- Per-trip azimuthal-equidistant projection centred on the GPS centroid: trips span the globe, so no single UTM zone fits, and distances near the track are accurate.
- Labels = nearest arc from the trip's own .route (Phase 1 simplification under time pressure); the monotonic alignment in src/labels.py replaces it later.
- Dropped train trips 33 and 54: their GPS has a median distance of 120–140 m from their own route, so the labels are unreliable (rule: median label distance > 50 m, applied to train only).
- K=8 nearest candidates per point: it keeps the miss rate around 2% while staying small enough for fast training.
- GPS heading uses the i-1 → i+1 central difference: it is less noisy than a single 1 s step.
- Train on every 4th GPS point: at 1 Hz, neighbouring points are near-duplicates, and this keeps all 5 models plus tuning under 5 minutes.
- Top-1 decision rule for F1 (the highest-scoring candidate per point = 1): it matches how predictions are actually used and needs no threshold.
- Model selection on CV macro F1 (train only), not on the student test: avoids choosing the model on the 10 test trips.
- Tune HistGradientBoosting: it had the best CV macro F1, test Top-1 and log loss of the 5 models.
- Scalers live inside sklearn Pipelines: they are fit on training folds only.
- Seed 42 everywhere.

## Phase 2 decisions
- HMM states = the 8 candidates per point; emission = classifier probability normalised over the 8; transition cost 0 (same arc), HOP (next arc), 2*HOP (two arcs ahead), FAR (not connected); no constraint after a GPS gap > 10 s.
- HOP and FAR chosen by a small grid on 10 train trips (not on student_test), then reported on student_test.
- Route continuity = share of consecutive points whose arcs are the same or directly connected.
- The API returns only GPS, predicted routes and accuracy; the true route stays on the server.

## Session 3 improvements (12:50-13:05), compared on student_test (development-stage comparison, as the guidelines allow)
- Labels switched to the order-preserving alignment (`align_monotonic`): routes that pass the same place twice no longer get labels from the wrong visit. This changes training AND scoring labels, so part of the gain is cleaner labels.
- Heading/speed now measured over points i-3..i+3 (was i-1..i+1): less jitter at low speed. Tried window 2 and 3; 3 was slightly better.
- 3 neighbour features: nb_dist_mean, nb_dist_max, nb_rank (is the same arc close to points i-2..i+2?).
- Sandbox results, default HGB: classifier 81.6% -> 86.1%, with HMM 84.2% -> 88.2% Top-1.
- Model choice: highest CV macro F1; differences under 0.002 are a tie (fold std is about 0.013), broken by faster inference (the API answers live). With the new features RandomForest and HGB tie (0.9291), HGB is about 40% faster.

## Session 4 (14:00): HMM U-turn penalty
- Error analysis on student_test: 5.2% of points were the right road in the wrong direction (reverse twin arc), the largest error type. The decoder treated A->B followed by B->A as a cheap 1-hop move.
- Added a separate UTURN cost and widened the decoder grid (HOP 1-2, FAR 10-20, UTURN 5-20), still chosen on 10 TRAIN trips. Train-trip Top-1 85.3% -> 87.1%; student_test about 88.0% -> 90.5% (sandbox).
