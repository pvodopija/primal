#!/bin/bash
# Pavle's night session (helmet HERO7, unstabilised) steadied by its own gyro
# (experiments/gyro_stabilise.py), against a control with the same lens remap and zoom and no
# turn: the pitch model replayed as the app runs it (session_bars.py: the delta at the line and
# the lap times against the transponder) and along whole laps against the camera's GPS
# (gps_compare.py). Light (0.08 s: shake only) and full (0.25 s: nods too) smoothing.
cd /Users/pvodopija/code/primal
PY="env PYTHONPATH=. ./.venv/bin/python"
RUN=newtracks_pitch_s0
for spec in "control 0" "s008 0.08" "s025 0.25"; do
  name=${spec% *}; sigma=${spec#* }
  [[ -f data/real-footage/night1/stream_$name.mp4 ]] || $PY experiments/gyro_stabilise.py render night1 $name $sigma 2>&1 | tail -1
done
for name in control s008 s025; do
  [[ -f data/real-footage/night1/gps_compare_${RUN}_$name.json ]] && continue  # scored
  echo "=== $name"
  REAL_FOOTAGE=night1 REAL_STREAM=$name $PY experiments/session_bars.py $RUN 2>&1 | grep -E "median|share|lap times" | head -6
  REAL_FOOTAGE=night1 REAL_STREAM=$name $PY experiments/gps_compare.py $RUN 2>&1 | tail -1
  $PY experiments/gyro_stabilise.py check night1 $name 2>&1 | tail -2
done
# the same at the lens's own width (STAB_ZOOM=1.0): the turned edges grey where they enter the crop
for spec in "control_z10 0" "s025_z10 0.25"; do
  name=${spec% *}; sigma=${spec#* }
  [[ -f data/real-footage/night1/stream_$name.mp4 ]] || STAB_ZOOM=1.0 $PY experiments/gyro_stabilise.py render night1 $name $sigma 2>&1 | tail -1
  [[ -f data/real-footage/night1/gps_compare_${RUN}_$name.json ]] && continue
  echo "=== $name"
  REAL_FOOTAGE=night1 REAL_STREAM=$name $PY experiments/session_bars.py $RUN 2>&1 | grep -E "median|share|lap times" | head -6
  REAL_FOOTAGE=night1 REAL_STREAM=$name $PY experiments/gps_compare.py $RUN 2>&1 | tail -1
done
