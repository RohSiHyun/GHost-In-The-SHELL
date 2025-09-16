#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

cd ../../../../../
./apply-no-mitigation.sh
cd -

nvidia-smi > /dev/null

if [ $# -ne 1 ]; then
  echo "Usage: $0 N"
  echo "Example: $0 50   # run ./mmap-eval 50 times and average 'time taken' (ns)"
  exit 1
fi

REPS="$1"
if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N must be a positive integer." >&2
  exit 2
fi

# === HARDCODED COMMAND ===
CMD=( "./mmap-eval" )
# ========================

sum=0            # total in nanoseconds (as integer with awk)
min=""
max=""
fail_count=0
declare -a values

for i in $(seq 1 "$REPS"); do
  echo "=== Run #$i / $REPS ==="
  out="$("${CMD[@]}" 2>&1 || true)"
  # Optionally show program output — comment out next line if noisy:
  printf "%s\n" "$out"

  # Extract integer from line like: "time taken: 21475 ns"
  # This finds the first integer on a line containing "time taken"
  time_ns=$(printf "%s\n" "$out" | awk 'tolower($0) ~ /time taken/ { for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { print $i; exit } }' | head -n1 || true)

  if [ -z "$time_ns" ]; then
    echo "Warning: couldn't parse 'time taken' on run #$i" >&2
    fail_count=$((fail_count+1))
    continue
  fi

  # Validate integer
  if ! printf "%s" "$time_ns" | grep -qE '^[0-9]+$'; then
    echo "Warning: parsed value not integer: '$time_ns' (run #$i)" >&2
    fail_count=$((fail_count+1))
    continue
  fi

  values+=("$time_ns")

  # update min/max
  if [ -z "$min" ] || [ "$time_ns" -lt "$min" ]; then
    min="$time_ns"
  fi
  if [ -z "$max" ] || [ "$time_ns" -gt "$max" ]; then
    max="$time_ns"
  fi

  # sum using awk to keep it safe for large sums
  sum=$(awk -v a="$sum" -v b="$time_ns" 'BEGIN{printf "%d", a + b}')
  echo "Parsed time taken: ${time_ns} ns"
done

successful_runs=$((REPS - fail_count))

echo
echo "Requested runs: $REPS"
echo "Successful parsed runs: $successful_runs"
echo "Failed parses: $fail_count"

if [ "$successful_runs" -eq 0 ]; then
  echo "No successful measurements to average." >&2
  exit 3
fi

# average may be fractional -> compute with awk, show 3 decimal places
avg=$(awk -v s="$sum" -v n="$successful_runs" 'BEGIN{printf "%.3f", s / n}')

echo "Sum: ${sum} ns"
echo "Average: ${avg} ns"
echo "Min: ${min} ns"
echo "Max: ${max} ns"

