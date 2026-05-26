#!/usr/bin/env bash
# analyze_diag.sh — Post-run analysis of nav2_diag.py output.
#
# Usage:
#   bash analyze_diag.sh diag_out.txt
#   bash analyze_diag.sh diag_out.txt > analysis_$(date +%H%M).txt
#
# Outputs seven sections in order, all to stdout:
#   1. Tag counts (which failure modes are happening)
#   2. Goal lifecycle summary (success/fail rate, durations)
#   3. The 8 most recent BT / ROSOUT / STUCK / HIGH_COST / LETHAL_FOOT /
#      CLOSE_PED / OBSTACLE_INVISIBLE / BACKUP_* events before each GOAL_FAIL
#   4. Spatial clustering of failure positions
#   5. Diagnostic node sanity: how many [STATE] lines, time span
#   6. BACKUP analysis: classifies BACKUP_END events by the BT node that was
#      active when the reverse started — i.e. DWB-chosen reverse during
#      FollowPath vs recovery-issued reverse during BackUp/Spin/Wait
#   7. OBSTACLE_INVISIBLE analysis: how many invisible-obstacle events
#      happened in the 5 s window before each GOAL_FAIL, indicating that
#      the scan saw a close obstacle that the local costmap didn't have
# Keep set -u (catches typos), drop -e and pipefail.  This script is
# read-only analysis; tolerating "no matches" exits from grep/awk inside
# pipes is essential — otherwise sections after a no-match (e.g. zero
# GOAL_OK) silently stop.
set -u

LOG="${1:-}"
if [[ -z "$LOG" ]]; then
    echo "Usage: $0 <diag_out.txt>" >&2
    exit 1
fi
if [[ ! -f "$LOG" ]]; then
    echo "Error: file not found: $LOG" >&2
    exit 1
fi

echo "=== nav2_diag analysis: $LOG ==="
echo

echo "--- 1. Tag counts ---"
for tag in INIT STATE GOAL_START GOAL_OK GOAL_FAIL GOAL_STATE \
           STUCK_START STUCK_END FWD_COLLISION \
           HIGH_COST LETHAL_FOOT CLOSE_PED \
           OBSTACLE_INVISIBLE BACKUP_START BACKUP_END BT \
           ROSOUT_LETHAL_START ROSOUT_NO_VALID_TRAJ ROSOUT_PATIENCE \
           ROSOUT_PLANNER_RATE ROSOUT_PLAN_FAILED ROSOUT_CONTROLLER_ABORT \
           ROSOUT_TRANSFORM_OLD CONTEXT_STATE CONTEXT_BT CONTEXT_END; do
    # grep -c already prints "0" on no-match (but exits 1).  || true keeps
    # set -e happy without appending a second "0".
    n=$(grep -c "\[${tag}\]" "$LOG" || true)
    printf '  %-25s %s\n' "$tag" "$n"
done
echo

echo "--- 2. Goal lifecycle summary ---"
ok=$(grep -c '\[GOAL_OK\]' "$LOG" || true)
fail=$(grep -c '\[GOAL_FAIL\]' "$LOG" || true)
total=$((ok + fail))
if [[ "$total" -gt 0 ]]; then
    printf '  successes:    %s\n' "$ok"
    printf '  failures:     %s\n' "$fail"
    printf '  total:        %s\n' "$total"
    printf '  fail rate:    %s%%\n' "$((100 * fail / total))"
else
    echo '  (no completed goals in this log)'
fi

echo
# Helper to print n/mean/min/max for whatever durations are piped in.
# Reads numeric durations (one per line) on stdin.
duration_stats() {
    awk '
      { v = $1 + 0
        if (n++ == 0) { mn = v; mx = v } else {
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        sum += v
      }
      END {
        if (n > 0)
            printf "    n=%d  mean=%.1fs  min=%.1fs  max=%.1fs\n",
                   n, sum / n, mn, mx
        else
            print "    (none)"
      }
    '
}

echo '  Goal durations (success):'
grep '\[GOAL_OK\]' "$LOG" \
  | grep -oE 'duration=[0-9.]+s' \
  | tr -d 'durations=' \
  | duration_stats

echo '  Goal durations (failure):'
grep '\[GOAL_FAIL\]' "$LOG" \
  | grep -oE 'duration=[0-9.]+s' \
  | tr -d 'durations=' \
  | duration_stats
echo

echo "--- 3. Events in the 8 lines before each GOAL_FAIL ---"
awk '
  /\[GOAL_FAIL\]/ {
    print "  --- failure ---"
    for (i=1; i<=n; i++) print "  " buf[i]
    print "  " $0
    delete buf; n = 0; next
  }
  /\[(BT|ROSOUT_|STUCK|FWD_COLLISION|HIGH_COST|LETHAL_FOOT|CLOSE_PED|OBSTACLE_INVISIBLE|BACKUP_)/ {
    if (n >= 8) { for (i=1; i<8; i++) buf[i] = buf[i+1]; n = 7 }
    n++; buf[n] = $0
  }
' "$LOG"
echo

echo "--- 4. Spatial clustering of failure positions ---"
fail_lines=$(grep '\[GOAL_FAIL\]' "$LOG" 2>/dev/null || true)
if [[ -z "$fail_lines" ]]; then
    echo '  (no GOAL_FAIL events to cluster)'
else
    # bucket robot positions into 1 m × 1 m bins so near-misses cluster
    echo "$fail_lines" \
      | grep -oE 'robot=\([^)]+\)' \
      | sed -E 's/robot=\(([^,]+),([^,]+)\)/\1 \2/' \
      | awk '{ printf "(%d, %d)\n", int($1), int($2) }' \
      | sort | uniq -c | sort -rn | head -10 \
      | awk '{ printf "  %s failures near %s\n", $1, $2" "$3 }'
fi
echo

echo "--- 5. Sanity: was the diag node receiving data? ---"
first_state=$(grep '\[STATE\]' "$LOG" 2>/dev/null | head -1 || true)
last_state=$(grep '\[STATE\]' "$LOG" 2>/dev/null | tail -1 || true)
n_state=$(grep -c '\[STATE\]' "$LOG" 2>/dev/null || echo 0)
echo "  first [STATE]: $first_state"
echo "  last  [STATE]: $last_state"
echo "  total [STATE] lines: $n_state"
if [[ "$n_state" -gt 5 ]]; then
    t0=$(echo "$first_state" | grep -oE 't=[0-9.]+' | head -1 | tr -d 't=')
    t1=$(echo "$last_state"  | grep -oE 't=[0-9.]+' | head -1 | tr -d 't=')
    if [[ -n "$t0" && -n "$t1" ]]; then
        printf '  approximate run duration: %.1f s\n' "$(echo "$t1 - $t0" | bc -l)"
    fi
fi
echo

echo "--- 6. BACKUP analysis ---"
# Count BACKUP_END events by the BT node that was active at backup start.
# This is the key question for the "excessive backing up" complaint:
#   FollowPath  -> DWB scored a reverse trajectory higher than any forward
#                  trajectory.  PreferForward penalty too low, or DWB
#                  legitimately had no forward escape.
#   BackUp      -> classic Nav2 recovery behavior firing.  Expected after a
#                  controller failure, but a high count means controller
#                  failures are frequent.
#   Spin        -> Nav2 spin recovery.  Doesn't actually drive reverse, but
#                  shows up if cmd_vel sees brief negatives.
#   ComputePathToPose / NavigateRecovery / unknown -> usually transient
#                  state changes; harder to interpret.
n_backup=$(grep -c '\[BACKUP_END\]' "$LOG" || true)
echo "  total backups: $n_backup"
if [[ "$n_backup" -gt 0 ]]; then
    echo "  by active BT node at start:"
    grep '\[BACKUP_END\]' "$LOG" \
      | grep -oE 'active_bt=[A-Za-z_]+' \
      | sort | uniq -c | sort -rn \
      | awk '{ printf "    %4d  %s\n", $1, $2 }'

    echo "  reverse distance distribution:"
    # NOTE: `tr -d` deletes a *set* of characters, not a substring.  Using
    # `tr -d 'reverse_dist='` would silently miss any character not in the
    # set (e.g. d, _) and corrupt values to garbage like "d_d0.05".  Use
    # sed with an anchored substring instead.
    grep '\[BACKUP_END\]' "$LOG" \
      | grep -oE 'reverse_dist=[0-9.]+' \
      | sed 's/^reverse_dist=//' \
      | awk '
          { v = $1 + 0
            if (n++ == 0) { mn = v; mx = v } else {
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            sum += v
            if (v > 0.30) far++
          }
          END {
            if (n > 0)
                printf "    n=%d  mean=%.2fm  min=%.2fm  max=%.2fm  >0.30m=%d\n",
                       n, sum/n, mn, mx, (far ? far : 0)
          }'

    echo "  duration distribution:"
    grep '\[BACKUP_END\]' "$LOG" \
      | grep -oE 'duration=[0-9.]+s' \
      | tr -d 'durations=' \
      | duration_stats
fi
echo

echo "--- 7. OBSTACLE_INVISIBLE near failures ---"
# For each GOAL_FAIL, look back at the [STATE] timestamps to find the failure
# time, then count [OBSTACLE_INVISIBLE] events within 5 s before that
# timestamp.  This is the smoking-gun correlation for "robot crashed because
# costmap didn't see the obstacle".
total_inv=$(grep -c '\[OBSTACLE_INVISIBLE\]' "$LOG" || true)
echo "  total OBSTACLE_INVISIBLE events: $total_inv"
if [[ "$total_inv" -gt 0 ]]; then
    echo "  closest-return range distribution:"
    grep '\[OBSTACLE_INVISIBLE\]' "$LOG" \
      | grep -oE 'range=[0-9.]+' \
      | tr -d 'range=' \
      | awk '
          { v = $1 + 0
            if (n++ == 0) { mn = v; mx = v } else {
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            sum += v
          }
          END {
            if (n > 0)
                printf "    n=%d  mean=%.2fm  min=%.2fm  max=%.2fm\n",
                       n, sum/n, mn, mx
          }'

    # Per-failure correlation: was an OBSTACLE_INVISIBLE within 5 s before
    # the failure?  Pure awk so we don't need to invoke external tools per
    # failure.
    awk '
      function extract_t(line,   out) {
        if (match(line, /t=[0-9.]+/) > 0) {
            out = substr(line, RSTART + 2, RLENGTH - 2) + 0
            return out
        }
        return -1
      }
      /\[OBSTACLE_INVISIBLE\]/ {
        ti = extract_t($0)
        if (ti >= 0) { invis[++n_inv] = ti }
      }
      /\[GOAL_FAIL\]/ {
        tf = extract_t($0)
        if (tf < 0) next
        c = 0
        for (i = 1; i <= n_inv; i++) {
            if (invis[i] >= tf - 5.0 && invis[i] <= tf) c++
        }
        n_fail++
        if (c > 0) n_with++
        total += c
        if (c > max) max = c
      }
      END {
        if (n_fail > 0)
            printf "  failures with >=1 invisible-obstacle event in the 5 s before: %d / %d (%.0f%%)\n",
                   n_with, n_fail, 100.0 * n_with / n_fail
        if (n_with > 0)
            printf "  per-failure invisible-event counts in that window: mean=%.1f  max=%d\n",
                   total / n_with, max
      }
    ' "$LOG"
fi
