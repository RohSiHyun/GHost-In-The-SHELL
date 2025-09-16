#!/bin/bash

ROOT_PATH=$(pwd)

echo "[+] patch /usr/include/x86_64-linux-gnu/sys/mman.h"
pushd /usr/include/x86_64-linux-gnu/sys/ > /dev/null
sudo patch -N -p0 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/LLVM/mman.patch
popd > /dev/null


echo "[+] GPU kernel module"
pushd $ROOT_PATH/third_party/open-gpu-kernel-modules-SHELL/ > /dev/null
make modules -j$(nproc) > /dev/null
sudo modprobe -r nvidia_uvm
sudo make modules_install
popd > /dev/null

nvidia-smi > /dev/null


echo "Done. If you want to work in python env, type below: "
echo "source ${ROOT_PATH}/third_party/pytorch/.pytorch/bin/activate"
