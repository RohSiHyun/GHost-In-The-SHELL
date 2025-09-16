#!/bin/bash

cd ../../../../
./apply-mitigation.sh
cd -

export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

make

sudo mkdir -p /opt/cudahook/

sudo cp libattack.so /opt/cudahook/
