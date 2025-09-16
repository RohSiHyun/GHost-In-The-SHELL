#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

cd ../../../../../
./apply-mitigation.sh
cd -

nvidia-smi > /dev/null

if [ $# -ne 1 ]; then
  echo "Usage: $0 N"
  echo "Example: $0 50   # run ./pf-patch-eval 50 times, drop low/high 10%, average ns"
  exit 1
fi

REPS="$1"
if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N must be a positive integer." >&2
  exit 2
fi

CMD=( "./pf-patch-eval" )

values=()
fail_count=0

for i in $(seq 1 "$REPS"); do
  echo "=== Run #$i / $REPS ==="
  "${CMD[@]}"

  line=$(sudo dmesg | grep "page fault handling took for mitigation" | tail -n1)
  echo "dmesg: $line"

  time_ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}')

  if [ -z "$time_ns" ] || ! [[ "$time_ns" =~ ^[0-9]+$ ]]; then
    echo "Warning: failed to parse run #$i" >&2
    fail_count=$((fail_count+1))
    continue
  fi

  values+=("$time_ns")
done

successful_runs=$((REPS - fail_count))
if [ "$successful_runs" -eq 0 ]; then
  echo "No successful values parsed." >&2
  exit 3
fi

# Sort values numerically
sorted=($(printf "%s\n" "${values[@]}" | sort -n))

# Compute how many to trim
trim=$(awk -v n="$successful_runs" 'BEGIN{printf "%d", n*0.1}')
if [ "$trim" -gt 0 ]; then
  trimmed=("${sorted[@]:$trim:$((successful_runs - 2*trim))}")
else
  trimmed=("${sorted[@]}")
fi

# Sum and average the trimmed values
sum=0
for v in "${trimmed[@]}"; do
  sum=$(awk -v a="$sum" -v b="$v" 'BEGIN{printf "%d", a+b}')
done

count=${#trimmed[@]}
avg=$(awk -v s="$sum" -v n="$count" 'BEGIN{printf "%.3f", s/n}')

echo
echo "Requested runs: $REPS"
echo "Successful: $successful_runs"
echo "Trimmed away: $((successful_runs - count))"
echo "Kept for average: $count"
echo "Average (trimmed 10% tails): ${avg} ns"
echo "Min kept: ${trimmed[0]} ns"
echo "Max kept: ${trimmed[-1]} ns"

