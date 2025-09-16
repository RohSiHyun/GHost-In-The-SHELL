#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

cd ../../../../
./apply-no-mitigation.sh
cd -

nvidia-smi > /dev/null

if [ $# -ne 1 ]; then
  echo "Usage: $0 N"
  echo "Example: $0 10   # runs ./baseline 2025 2025 ./binary_1hr_all/ 10 times"
  exit 1
fi

REPS="$1"
if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N must be a positive integer." >&2
  exit 2
fi

# === HARDCODED COMMAND (do not change this script's args) ===
CMD=( "./baseline" "2025" "2025" "./binary_1hr_all/" )
# ===========================================================

sum=0
min=""
max=""
fail_count=0

for i in $(seq 1 "$REPS"); do
  echo "=== Run #$i / $REPS ==="
  echo "Command: ${CMD[*]}"
  # Run command, capture stdout+stderr; don't abort the loop if command fails
  out="$("${CMD[@]}" 2>&1 || true)"

  # Optionally show the full run output (comment this out if noisy)
  printf "%s\n" "$out"

  # Extract numeric value from a line containing "Total time:"
  # Primary method: take the second-last field from the line (e.g. "Total time: 4.05643 seconds")
  time_val=$(printf "%s\n" "$out" | awk '/Total time:/{ if (NF>=2) print $(NF-1) }' | head -n1 || true)

  # Fallback: find any numeric token on the "Total time" line
  if [ -z "$time_val" ]; then
    time_val=$(printf "%s\n" "$out" | awk '/Total time/{ for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) print $i }' | head -n1 || true)
  fi

  if [ -z "$time_val" ]; then
    echo "Warning: couldn't parse 'Total time' on run #$i" >&2
    fail_count=$((fail_count+1))
    continue
  fi

  # Validate numeric form
  if ! printf "%s" "$time_val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    echo "Warning: parsed value is not numeric: '$time_val' (run #$i)" >&2
    fail_count=$((fail_count+1))
    continue
  fi

  # Update min/max
  if [ -z "$min" ] || awk "BEGIN{exit !($time_val < $min)}"; then
    min="$time_val"
  fi
  if [ -z "$max" ] || awk "BEGIN{exit !($time_val > $max)}"; then
    max="$time_val"
  fi

  # Sum using awk for floating point precision
  sum=$(awk -v a="$sum" -v b="$time_val" 'BEGIN{printf "%.14f", a + b}')
  echo "Parsed Total time: $time_val seconds"
done

successful_runs=$((REPS - fail_count))

echo
echo "Requested runs: $REPS"
echo "Successful parsed runs: $successful_runs"
echo "Failed parses / failures: $fail_count"

if [ "$successful_runs" -eq 0 ]; then
  echo "No successful measurements to average." >&2
  exit 3
fi

avg=$(awk -v s="$sum" -v n="$successful_runs" 'BEGIN{printf "%.8f", s / n}')
echo "Sum of parsed times: $sum seconds"
echo "Average Total time: $avg seconds"
echo "Min: $min seconds"
echo "Max: $max seconds"

