# Nav2CAN tuning changelog (`inspect` world, point-nav)

Reverse-chronological record of parameter and code changes during the
Nav2CAN proxemic-tuning campaign on 2026-05-25.  One row per
`metrics_data/inspect/Nav2CAN/inspect_nav2_*.csv` run.

Primary metric: **`goals_count.max() / count(encounter_min_dist < 0.5 m)`**
across all pedestrian actors (`diag_actor_dist`, `linear_actor_dist`,
`triangle_actor_dist`), as printed by
`plot_scripts/ratio_table.py` and the `goals per 0.5 encounter` line of
`plot_scripts/inspect_multi_plot_new.py`.  Higher is better.

Source of truth for pre-tuning baseline:
`ros2_ws/src/roverrobotics_ros2/roverrobotics_gazebo/config/leo_nav2_lidar_params.yaml`
at commit `46c5876` (2026-03-10, unmodified since).  Working copy:
`ros2_ws/src/context_aware_navigation/params/nav2_params_leo.yaml`
(untracked — has no git history of its own).

---

## Per-run summary

| # | CSV start | dur (min) | goals | enc <1.2 | 0.5–0.8 | **<0.5** | **ratio** | diag file |
|---|-----------|-----------|-------|----------|---------|----------|-----------|-----------|
| 1 | 11:05 | 30 | **70** | 111 | 42 | 27 | **2.59** *(best)* | — |
| 2 | 11:43 | 30 | 59 | 130 | 35 | 32 | 1.84 | — |
| 3 | 12:26 | 30 | 47 | 125 | 36 | 36 | 1.31 | — |
| 4 | 13:59 | 30 | 59 | 135 | 37 | 41 | 1.44 | `diag_out3.txt` (14:00) |
| 5 | 14:59 | 30 | 39 | 107 | 22 | 24 | 1.62 | `diag_out4.txt` (15:00) → `analysis4.txt` |
| 6 | 15:35 | 30 | 47 | 134 | 34 | 40 | **1.18** *(worst)* | — |
| 7 | 16:10 | 30 | 48 | 130 | 34 | 34 | 1.41 | — |
| 8 | 18:07 | 30 | 40 | 131 | 33 | 30 | 1.33 | `diag_out8.txt` → `analysis8.txt` |
| 9 | 19:23 | 30 | 63 | 117 | 43 | 41 | 1.54 | `diag_out9.txt` → `analysis9.txt` |
| 10 | 20:09 | 30 | 42 | 115 | 32 | 23 | 1.83 | — |
| 11 | 20:44 | 30 | 53 | 113 | 39 | 29 | 1.83 | — |
| 12 | 21:25 | 3 | 5 | 16 | 3 | 5 | 1.00 | partial run; ignore |
| 13 | 21:47 (05-25) | 30 | 43 | 118 | — | — | **1.26** | ablation A: `min_vel_x: −0.7`, `PreferForward` removed — **failed**, see below |
| 14 | 06:40 (05-26) | 30 | 48 | 132 | — | 33 | **1.45** | ablation A′: `min_vel_x: −0.7`, `PreferForward.penalty: 15.0` kept — **failed** (< 1.83), see below |
| 15 | 07:28 (05-26) | 30 | 50 | 128 | 32 | 28 | **1.79** | run-11 config restored; first run with enriched `[CLOSE_PED]` / new `[CLOSE_PED_END]` instrumentation — see "Perception coverage" below |
| 16 | ~09:30 (05-26) | 25 *(early kill)* | 2 *(per diag)* | — | — | — | **n/a — failed** | footprint half-extent `0.23 → 0.27`: predicted to fix wheel-clipping, instead made the inspect world unnavigable.  `ROSOUT_NO_VALID_TRAJ` 4 → **27**, `STUCK_START` 2 → 14, `GOAL_FAIL` 25 → 37, only 2 successful goals.  Reverted to 0.23 for run 17. |
| 17 | *(pending)* (05-26) | 30 | — | — | — | — | (variance baseline) | footprint **reverted to 0.23**; otherwise identical to run 15.  Diagnostic-only run to (a) confirm rollback restored run-15-level performance and (b) gather more data on the FollowPath-internal pause/reverse pattern (95 such events in run 16; 57 in run 15) ahead of a targeted ablation. |

Diag files `diag_out2.txt` (13:21), `diag_out5.txt` (16:26),
`diag_out6.txt` (16:58), and `diag_out7.txt` (17:31) do not have a
matching CSV — they appear to be `nav2_diag.py` development runs that
were not paired with the collector.  Diag-numbered "run N" in the chat
transcripts refers to `diag_outN.txt` / `analysisN.txt`, **not** CSV
position.

---

## Configuration delta vs upstream baseline

Below: every parameter that differs between
`leo_nav2_lidar_params.yaml` (baseline) and the current
`nav2_params_leo.yaml`.  Direction marked `↓ safety` when the change
likely degrades the proxemic ratio, `↑ safety` when it likely improves
it, `~` when neutral or mixed.  These are *hypotheses* from first
principles; ablation results will refine them.

### controller_server.FollowPath

| param | baseline | current | dir |
|-------|----------|---------|-----|
| `controller_frequency` (server) | 15.0 | 20.0 | ↑ |
| `progress_checker.required_movement_radius` | 0.20 | 0.15 | ~ |
| `progress_checker.movement_time_allowance` | 20.0 | 5.0 | ~ |
| `min_vel_x` | **−0.7** | **−0.15** | **↓** large |
| `acc_lim_theta` | 2.5 | 1.5 | ↓ small |
| `decel_lim_theta` | −2.5 | −1.5 | ↓ small |
| `critics` list | no `PreferForward` | + `PreferForward` | **↓** large |
| `BaseObstacle.scale` | 0.15 | 0.5 | ↑ |
| `PreferForward.penalty` | absent | 15.0 | **↓** large |
| `PreferForward.strafe_x` | absent | 0.1 | ~ |
| `PreferForward.strafe_theta` | absent | 0.2 | ~ |
| `PreferForward.theta_scale` | absent | 10.0 | ~ |
| `PathAlign.forward_point_distance` | 0.1 | 0.325 | ~ |
| `GoalAlign.forward_point_distance` | 0.1 | 0.325 | ~ |
| `Oscillation.scale` | default (1.0) | 1.0 (explicit) | ~ |
| `Oscillation.oscillation_reset_dist` | default (0.05) | 0.10 | ~ |
| `Oscillation.oscillation_reset_angle` | default (0.2) | 0.4 | ~ |
| `Oscillation.x_only_threshold` | default | 0.05 (explicit) | ~ |

Removed compared to baseline: `min_y_velocity_threshold`,
`debug_trajectory_details`, `min_vel_y`, `max_vel_y`, `acc_lim_y`,
`decel_lim_y`, `vy_samples`, `linear_granularity`,
`angular_granularity`, `trans_stopped_velocity`,
`short_circuit_trajectory_evaluation`.  All are 2D-holonomic vestiges
not relevant to a diff-drive robot.

### planner_server (`GridBased` / SmacPlanner2D)

| param | baseline | current | dir |
|-------|----------|---------|-----|
| `expected_planner_frequency` | 20.0 | 5.0 | ~ |
| `max_planning_time` | absent | 2.0 | ~ |
| `cost_travel_multiplier` | 2.0 | 3.5 | ↑ |
| `downsample_costmap`, `downsampling_factor` | set | removed | ~ |

### local_costmap

| param | baseline | current | dir |
|-------|----------|---------|-----|
| `transform_tolerance` | default 0.3 | 0.5 | ~ |
| robot shape | `robot_radius: 0.4` | `footprint: [[0.23, 0.23], …]` *(reverted from 0.27 after run 16 broke navigability)* | ↓ medium |
| `plugins` | `[obstacle_layer, inflation_layer]` | + `social_layer` | ↑ |
| `obstacle_layer.scan.observation_persistence` | 0.5 | 0.3 | ~ |
| `obstacle_layer.scan.obstacle_max_range` | **8.0** | **4.0** | **↓** medium |
| `inflation_layer.cost_scaling_factor` | 5.0 | 3.0 | ~ |
| `inflation_layer.inflation_radius` | 0.4 | 0.55 | ↑ |

### global_costmap

| param | baseline | current | dir |
|-------|----------|---------|-----|
| `transform_tolerance` | default 0.3 | 0.5 | ~ |
| robot shape | `robot_radius: 0.3` | `footprint: [[0.23, 0.23], …]` *(reverted from 0.27 after run 16 broke navigability)* | ↓ medium |
| `plugins` | `[obstacle_layer, inflation_layer]` | + `social_layer` | ↑ |
| `obstacle_layer.scan.observation_persistence` | 1.0 | 0.3 | ~ |
| `obstacle_layer.scan.obstacle_max_range` | **8.0** | **4.0** | **↓** medium |
| `inflation_layer.inflation_radius` | 0.5 | 0.65 | ↑ |

---

## Code changes (non-YAML)

### `ros2_ws/src/rover_metrics/diagnostics/nav2_diag.py`
Built during this campaign; no baseline.  Iterated:
- Per-UUID `goal_start_time` so `GOAL_FAIL` `duration=` reflects the
  actual lifetime of each goal.
- `_active_bt_node` walks `bt_recent` backward past scaffolding
  (`RateController`, `RoundRobin`, etc.) to find the active driving
  node (`FollowPath`, `BackUp`, `Spin`).  Falls back to
  `no_driver_<scaffolding-name>` for visibility.
- `bt_recent` deque grown from `maxlen=20` to `maxlen=100` so the
  walk-back can reach past one BACKUP debounce interval.
- `STUCK_START` guard `self.stuck_since != -1` added so the latched
  stuck state cannot re-emit `STUCK_START` for the same episode.
- **2026-05-26 proxemic instrumentation**: `[CLOSE_PED]` enriched
  with `cmd_vx`, `cmd_wz`, and `active_bt` at encounter entry, so
  the approach type and BT state are visible without grepping back
  through `[STATE]`.  New `[CLOSE_PED_END]` event paired with each
  `[CLOSE_PED]` carries the encounter `duration` and
  `min_ped_dist` — distinguishing 0.49 m grazes from 0.15 m
  near-collisions.  These pairs match the per-encounter semantics
  of `plot_scripts/inspect_multi_plot_new.py`, so the diag
  `[CLOSE_PED]` count and the plot's `<0.5 m` count should agree.

### `ros2_ws/src/rover_metrics/nav2_lidar_metrics_collector.py`
- Removed the `/follow_path/_action/status` subscription and
  `controller_abort_count` / `controller_abort_hard` state — these
  produced premature respawns on routine controller aborts during
  Nav2's recovery sequence.
- Added module-level `ABORT_GRACE_S: float = 2.0` and a per-goal
  `goal_send_sim_time` so `ABORTED`/`CANCELED` messages on
  `/navigate_to_pose` that arrive within `ABORT_GRACE_S` of the most
  recent `send_goal()` are ignored.  This eliminated the
  "spawn-cluster" false-failure burst seen in run 4 / `analysis4.txt`.
- Failure path now respects only the BT-root `nav2_goal_failed`
  signal ("Nav2 BT aborted — respawning").

### `ros2_ws/src/rover_metrics/plot_scripts/ratio_table.py`
New helper, added 2026-05-25.  Per-CSV table of goals, encounters
binned by min-distance, and the `goals / <0.5 m` ratio.  Imports the
encounter-detection from `inspect_multi_plot_new.py` so the numbers
match the existing plot caption.

### `ros2_ws/src/rover_metrics/diagnostics/analyze_diag.sh`
- Added `CLOSE_PED_END` to the tag-count list (section 1).
- **New section 8 (CLOSE_PED analysis)**: total entry/exit counts,
  `min_ped_dist` distribution with 10 cm severity bins (0.0-0.5 m),
  encounter duration statistics, approach-type breakdown
  (forward / slow_fwd / reverse / stationary based on `cmd_vx` at
  entry), active-BT-node breakdown at entry, and spatial clustering
  of close-pass start positions (1 m × 1 m bins, mirrors section 4).
- **New section 9 (events before each CLOSE_PED)**: the 8 most
  recent `BT` / `ROSOUT_*` / `STUCK_*` / `FWD_COLLISION` /
  `HIGH_COST` / `LETHAL_FOOT` / `OBSTACLE_INVISIBLE` / `BACKUP_*` /
  `CLOSE_PED_END` events before each `CLOSE_PED`.  Mirrors
  section 3 (events-before-GOAL_FAIL) but rooted on close passes,
  so we can see what BT state and costmap regime led into each
  proxemic violation.  Gracefully degrades on logs from before the
  CLOSE_PED enrichment (no `cmd_vx` / `active_bt` fields → those
  sub-sections render empty but the spatial clustering and
  events-before sub-sections still work).

---

## Hypothesized regression sources (highest-impact first)

1. **`min_vel_x: −0.7 → −0.15` and `PreferForward.penalty` added at 15.0.**
   Together, these prevent DWB from reversing to evade a pedestrian who
   has walked into the robot's forward path.  Reverse-to-evade was the
   robot's primary close-pass-avoidance maneuver in run 1.  Combined
   expected impact: large.

2. **`obstacle_max_range: 8.0 → 4.0` (local + global).**  Pedestrians
   enter the costmap at 4 m instead of 8 m, halving the time the
   `social_layer` and `inflation_layer` have to paint cost before the
   robot is committed to a path past the person.  Expected impact:
   medium.

3. **`footprint: robot_radius:0.4 → 0.23 × 0.23` rectangle.**  Modeled
   robot is now smaller in both planning and control, so passes
   through gaps that were previously rejected — including
   pedestrian-adjacent gaps.  Expected impact: medium.

4. **`acc_lim_theta: 2.5 → 1.5`.**  Halves the robot's ability to yaw
   away from a sudden close approach.  Expected impact: small (this
   change was made between runs 8 and 9, and ratio went *up* over that
   interval, so probably not a dominant factor; keep watching).

Changes that likely help the ratio (keep):
- `BaseObstacle.scale 0.15 → 0.5`
- `cost_travel_multiplier 2.0 → 3.5`
- `inflation_radius local 0.4 → 0.55`, `global 0.5 → 0.65`
- `controller_frequency 15 → 20`
- adding `social_layer` to both costmaps

---

## What we *don't* know

- Exactly which YAML changes were in effect for runs 1–3 (no diag
  files, and `nav2_params_leo.yaml` is untracked so we can't diff
  against its history).  We only know the ratio was 2.59 / 1.84 / 1.31
  for those runs, in that order.  The 2.59 → 1.84 step is the
  single-biggest regression in the campaign and we cannot directly
  attribute it without the bisection ablations below.
- Which subset of the four hypothesized-regression changes is actually
  responsible.  Ablation plan below.

---

## Ablation results

### A. `min_vel_x: −0.7` + remove `PreferForward` (run 13, 21:47)

**Result: ratio 1.26 — failed.** Second-worst of the day; predicted
≥ 2.3, observed 1.26.  Hypothesis falsified: re-opening the reverse
band and dropping `PreferForward` does *not* recover the run-1 ratio.

Interpretation: free reverse appears to *hurt* the proxemic metric in
this configuration.  Best explanation is that the front-only YOLO
camera cannot see what the robot is reversing into, so an unrestricted
`min_vel_x` lets DWB choose reverse trajectories that move the robot
*toward* pedestrians that are behind/behind-and-to-the-side.  The
narrow `−0.15` band capped at `0.15 × sim_time(1.7) = 0.255 m`
constrains reverse to ground the robot saw seconds earlier, which is
why run 11 (with `−0.15` and `PreferForward.penalty: 15.0`) was
**1.83** — meaningfully better than run 13's 1.26.

Reverting to the run-11 config and looking elsewhere for the gap from
1.83 → 2.59.

### A′. `min_vel_x: −0.7` + `PreferForward.penalty: 15.0` kept (run 14, 06:40 05-26)

**Result: ratio 1.45 — failed.** Predicted [1.26, 1.83]; observed 1.45,
clearly below run-11's 1.83.

Conclusion of the three-point bisection on the reverse parameters:

| run | `min_vel_x` | `PreferForward.penalty` | ratio |
|-----|-------------|--------------------------|-------|
| 13  | −0.7 | (removed) | 1.26 |
| 14  | −0.7 | 15.0 | 1.45 |
| 11  | **−0.15** | **15.0** | **1.83** |

PreferForward.penalty=15 does *not* override DWB's GoalDist/PathDist
pull when the full reverse band is available — DWB still picks
occasional reverses that hit the proxemic metric.  The `-0.15` band
provides a hard cap (0.15 × sim_time = 0.255 m max reverse) that the
penalty alone cannot enforce.

**Run-11 config (`min_vel_x: -0.15` + `PreferForward.penalty: 15.0`)
is now locked as the project baseline for proxemic safety.**  Future
ablations branch from this state, not from upstream.

## Perception coverage finding (run 15, 2026-05-26)

The enriched `[CLOSE_PED]` / `[CLOSE_PED_END]` instrumentation
revealed a major source-of-truth mismatch that had been hidden in
every prior run:

| source | data origin | <0.5 m count, run 15 |
|--------|-------------|-----------------------|
| `inspect_multi_plot_new.py` on the CSV | per-actor ground-truth `Pose` topics (`actor_topics` in `nav2_lidar_metrics_collector.py`) | **28** |
| `nav2_diag.py` `CLOSE_PED` count | `/people` from `multi_person_tracker` (front-camera YOLO) | **10** |

**The robot perceived only 10 of 28 actual close passes (36 %).**
The remaining 18 were pedestrians behind, off-FOV-edge, or occluded
by the front-only camera mounting.  The `social_layer` only paints
cost around *perceived* pedestrians; in 64 % of close passes the
costmap was clean where the pedestrian actually was.

### Consequences

- Every parameter we have tuned (DWB critics, `min_vel_x`,
  `PreferForward`, inflation, social-layer cost weighting) only
  affects the controller's response to *perceived* pedestrians.  The
  unperceived 64 % are invisible to all of it.
- The achievable ratio ceiling by tuning alone, given today's
  perception, is approximately `goals / unperceived_close_passes`.
  Run 15: `50 / 18 ≈ 2.78`.  Run 1's 2.59 fits inside this ceiling
  comfortably — it doesn't require a mystery config.
- Best controller tuning can attack the **10 perceived close passes
  per run**.  Section 8 / 9 of `analysis10.txt` classifies them:

  | type | count | actionability |
  |------|-------|---------------|
  | robot stationary in `Wait` recovery, ped walks in | 5 | low — controller can't move; ped behavior unbounded |
  | robot driving forward at max speed (`cmd_vx ≈ +0.70`) | 3 | **high** — DWB / social_layer should slow it down |
  | reverse-aftermath (just finished a small reverse) | 1 | low |
  | slow forward (cautious turning) | 1 | low |

### Implications for next steps

Two paths, in priority order:

1. **Investigate the perception coverage gap (high upside).**  Possible
   approaches: rear or 360° camera, LiDAR-based pedestrian clustering
   to supplement YOLO, increase `obstacle_max_range` on the LiDAR
   layer to push perception range out, or compare runs with vs
   without YOLO occlusion analysis.  Even halving the unperceived
   share (18 → 9) would move the controllable+uncontrollable ceiling
   from `50/18 = 2.78` to `50/9 = 5.56`.

2. **Tune the 3 forward-at-max-speed close passes (small upside).**
   Candidate knobs: raise `BaseObstacle.scale` (currently 0.5) so
   ped-adjacent cells outweigh GoalDist; increase the
   `social_layer` cost amplitude; add a velocity-limit critic that
   forces lower `max_vel_x` near social cost.  Best case this drops
   close-pass count from 10 to 7 perceived — corresponding ratio
   improvement maybe 1.79 → 1.95.

### Open question

Is there a way to bisect *without* burning 30 min per ablation?
Worth investigating whether the encounter-detection on a shorter run
(e.g. 10 min) is statistically meaningful, or if it introduces too
much variance.

## Open experiments (revised)

Each runs for 30 minutes in the `inspect` world with the standard 3
pedestrian actors (`linear`, `diag`, `triangle`).  One ablation at a
time.  All ablations branch from the **run-11 config** as the new
baseline (ratio 1.83), not from the upstream YAML.

Candidate ablations to bisect the remaining 1.83 → 2.59 gap:

| change | rationale |
|--------|-----------|
| `obstacle_max_range: 4.0 → 8.0` (both maps) | restores 8 m pedestrian lookahead; the hill-ghost issue is a goal-success-rate concern, may be worth the trade if the metric improves |
| `BaseObstacle.scale: 0.5 → 0.15` (baseline value) | confirm whether raising it actually helped or whether it pushes DWB into ped-adjacent cells indirectly |
| `inflation_radius local 0.55 → 0.4`, `global 0.65 → 0.5` | confirm the inflation gradient changes are doing what we think |
| `cost_travel_multiplier 3.5 → 2.0` | confirm the planner-detour change helps |
| `footprint → robot_radius: 0.3` | confirm the modeled footprint isn't squeezing the robot past peds |

These are now hypotheses, not predictions.  Each one will be tested
individually.  The ratio-table after each ablation goes in the
**per-run summary** above.

After any ablation, re-run `python3 plot_scripts/ratio_table.py
metrics_data/inspect/Nav2CAN/` and append a row to the **per-run
summary** table above.

---

## Going forward

1. Append a row to the per-run summary for every new CSV produced by
   `nav2_lidar_metrics_collector.py`.
2. When changing a YAML or code parameter, add a brief entry under
   **Configuration delta** or **Code changes** with the run number it
   first took effect in.
3. Open question: at some point we should commit the
   `context_aware_navigation/` directory (currently untracked).  Once
   committed, `git log -p params/nav2_params_leo.yaml` becomes the
   authoritative timeline and this document can summarize rather than
   document.
