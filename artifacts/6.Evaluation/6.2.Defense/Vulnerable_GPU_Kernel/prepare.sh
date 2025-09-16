#!/bin/bash

cd ../../../../
./apply-mitigation.sh
cd -
cd attacker

export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

./make_payload.sh > /dev/null
