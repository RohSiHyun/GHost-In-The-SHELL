#!/bin/bash

ROOT_PATH=$(pwd)

# Package installation
echo "install all dependencies"
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update --fix-missing

sudo apt install -y gcc-12 g++-12 python3.12 python3.12-venv python3.12-dev python3-pip build-essential git curl wget libeigen3-dev gperf libjemalloc-dev vulkan-tools\
    python3-typing-extensions cmake ninja-build clang libomp-dev gdb > /dev/null

bash -c "$(curl -fsSL https://gef.blah.cat/sh)"
sudo bash -c "$(curl -fsSL https://gef.blah.cat/sh)"

# Git submodule initialization
echo "initialize all git submodules"
echo "[+] initialize llvm-project (17.0.6)"
git submodule update --init --recursive third_party/llvm-project-17.0.6 > /dev/null
echo "[+] initialize llvm-project-patch (17.0.6)"
git submodule update --init --recursive third_party/llvm-project-17.0.6-patch > /dev/null
echo "[+] initialize llvm-project (19.1.7)"
git submodule update --init --recursive third_party/llvm-project-19.1.7 > /dev/null
echo "[+] initialize open-gpu-kernel-modules"
git submodule update --init --recursive third_party/open-gpu-kernel-modules > /dev/null
echo "[+] initialize open-gpu-kernel-modules-SHELL"
git submodule update --init --recursive third_party/open-gpu-kernel-modules-SHELL > /dev/null
echo "[+] initialize pytorch"
git submodule update --init --recursive third_party/pytorch > /dev/null
echo "[+] initialize depot_tools"
git submodule update --init --recursive third_party/depot_tools > /dev/null

sudo modprobe -r nouveau

sudo systemctl stop gdm
sudo modprobe -r nouveau

export CC=gcc-12
export CXX=g++-12


# GPU execution environment preparation
# Install Driver & CUDA
echo "preparing for GPU kernel execution environment"

sudo systemctl stop gdm3

echo "[+] install nvidia display driver"
pushd $ROOT_PATH/third_party/ > /dev/null

if [ -f "NVIDIA-Linux-x86_64-570.144.run" ]; then
  echo "[+] already exists, skip"
else
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/570.144/NVIDIA-Linux-x86_64-570.144.run
  chmod +x NVIDIA-Linux-x86_64-570.144.run
  sudo sh NVIDIA-Linux-x86_64-570.144.run --silent --no-kernel-modules --accept-license --disable-nouveau
fi

popd > /dev/null

echo "[+] building open GPU kernel modules"
pushd $ROOT_PATH/third_party/open-gpu-kernel-modules/ > /dev/null
git checkout 8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f

patch -p1 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.3.Runtime_Enforcement/baseline-driver-8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f.patch

make modules -j$(nproc) > /dev/null
sudo make modules_install > /dev/null
popd > /dev/null


#confirm installation of NVIDIA Driver
pushd $ROOT_PATH/third_party/ > /dev/null
sudo sh NVIDIA-Linux-x86_64-570.144.run --silent --no-kernel-modules --accept-license --disable-nouveau

cd open-gpu-kernel-modules
sudo make modules_install

popd > /dev/null


echo "[+] install cuda toolkit (12.8)"
pushd $ROOT_PATH/third_party > /dev/null

if [ -f "cuda_12.8.0_570.86.10_linux.run" ]; then
  echo "[+] already exists, skip"
else
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
  chmod +x cuda_12.8.0_570.86.10_linux.run
  sudo sh ./cuda_12.8.0_570.86.10_linux.run --silent --toolkit --no-opengl-libs > /dev/null
fi

popd > /dev/null

echo "[+] Environment Variables Setup for CUDA"
export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

# LLVM Installation
echo "LLVM build"
echo "[+] build llvm 17.0.6"
pushd $ROOT_PATH/third_party/llvm-project-17.0.6/ > /dev/null
git checkout 6009708b4367171ccdbf4b5905cb6a803753fe18
mkdir -p build
cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_TERMINFO=ON \
../llvm
ninja
popd > /dev/null



# Attack preparation
# PyTorch Patch & Build
echo "preparing modules for attack poc"

echo "[+] build libSHELL"
pushd $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ > /dev/null
make
popd > /dev/null

echo "[+] patch pyTorch"
export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

pushd $ROOT_PATH/third_party/pytorch

git checkout 1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340
# See if .pytorch exist
if [ -d ".pytorch" ]; then
  echo "[+] .pytorch directory already exists, skipping venv creation"
else
  python3.12 -m venv .pytorch
  echo "[+] creating virtual environment for pyTorch"
fi

source .pytorch/bin/activate

patch -p1 < $ROOT_PATH/artifacts/6.Evaluation/6.1.Attack/Vulnerable_GPU_Kernel/pytorch-patch/pytorch-1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340-v2.patch
pip3 install -r requirements.txt

cd third_party
git submodule update --init --recursive
cd ..

mkdir build
cd build
echo "[+] configuring pyTorch"

PREFIX="${VIRTUAL_ENV:-$PWD/_cmake_prefix}"
mkdir -p "$PREFIX"

ORIG_LDFLAGS="${LDFLAGS-}"
export TORCH_NVCC_FLAGS="-G"
export CLANG_BIN="$ROOT_PATH/third_party/llvm-project-17.0.6/build/bin"
export CC="$CLANG_BIN/clang"
export CXX="$CLANG_BIN/clang++"
export TORCH_LIBDIR="$PREFIX/lib"
export LDFLAGS="-L${TORCH_LIBDIR} -Wl,-rpath,${TORCH_LIBDIR} ${LDFLAGS-}"
export LIBRARY_PATH="${TORCH_LIBDIR}:${LIBRARY_PATH-}"

cmake ..  \
  -G Ninja \
  -DUSE_NCCL=0 \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DCMAKE_INSTALL_RPATH=${PREFIX}/lib \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON ${CMAKE_ARGS:-} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_CUSTOM_PROTOBUF=OFF \
  -DCMAKE_C_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6/build/bin/clang \
  -DCMAKE_CXX_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6/build/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6/build/bin/clang++ \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_FLAGS="-Wno-vla-extension \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6/openmp/build/runtime/src \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6/offload/include/OpenMP/ \
    -O0 -Wno-unknown-warning-option \
    -Wno-extra-semi \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/usr/include/c++/12/ \
    -Wno-c++98-compat-extra-semi \
    -Wno-dev \
    -Wno-deprecated-literal-operator \
    -Wno-deprecated-copy \
    -Wno-sign-compare \
    -Wno-unused-function \
    -Wno-unused-command-line-argument" \
  -DCMAKE_CXX_FLAGS="-Wno-vla-extension \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6/openmp/build/runtime/src \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6/offload/include/OpenMP/ \
    -O0 -Wno-unknown-warning-option \
    -Wno-extra-semi \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/usr/include/c++/12/ \
    -Wno-c++98-compat-extra-semi \
    -Wno-dev \
    -Wno-deprecated-literal-operator \
    -Wno-deprecated-copy \
    -Wno-sign-compare \
    -Wno-unused-function \
    -Wno-unused-command-line-argument" \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler \
                     -Xcompiler -Wno-vla-extension \
                     -Wno-extra-semi \
                     -I$ROOT_PATH/third_party/llvm-project-17.0.6/openmp/build/runtime/src \
                     -O0 -Wno-unknown-warning-option \
                     -I/usr/include/x86_64-linux-gnu/c++/12/ \
                     -Xcompiler -O0 \
                     -Xcompiler -mavx512f \
                     -Xcompiler -mavx512bw \
                     -Xcompiler -mavx512vl \
                     -I/usr/include/c++/12/ \
                     -Xcompiler -Wno-c++98-compat-extra-semi \
                     -Xcompiler -Wno-dev \
                     -Xcompiler -Wno-deprecated-literal-operator \
                     -Xcompiler -Wno-deprecated-copy \
                     -Xcompiler -Wno-unused-command-line-argument \
                     -Xcompiler -Wno-unused-function" \
  -DUSE_FBGEMM=0 \
  -DUSE_KINETO=0 \
  -DUSE_XNNPACK=0 \
  -DUSE_GLOO=0 \
  -DUSE_TENSORPIPE=0 \
  -DUSE_OBSERVERS=0 \
  -DUSE_DISTRIBUTED=0 \
  -DUSE_SYSTEM_XNNPACK=0 \
  -DUSE_CUDA=1 \
  -DUSE_MKLDNN=0 \
  -DUSE_OPENMP=0 \
  -DUSE_PTHREADPOOL=0 \
  -DUSE_CUDNN=0 \
  -DUSE_FLASH_ATTENTION=0 \
  -DUSE_MEM_EFF_ATTENTION=0 \
  -DUSE_ITT=0 \
  -DUSE_SYSTEM_FLATBUFFERS=0 \
  -DUSE_LITE_PROTO=0 \
  -DINTERN_DISABLE_ONNX=0 \
  -DTORCH_CUDA_ARCH_LIST="8.0" \
  -DOpenMP_C_FLAGS="-fopenmp" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/usr/lib/x86_64-linux-gnu/libomp.so.5"

echo "[+] building pyTorch"

ninja

cd ../

USE_CUDA=1 \
USE_CUDNN=0 \
USE_ROCM=0 \
python setup.py develop

echo "[+] installing pytorch"
pip install -e . --no-build-isolation
pip install netCDF4 tqdm
pip install pwntools

mkdir -p torch/bin
mkdir -p torch/lib

cp -r .pytorch/bin/* torch/bin/
cp -r .pytorch/lib/* torch/lib/

deactivate

popd > /dev/null

export CC=gcc-12
export CXX=g++-12
export LDFLAGS="${ORIG_LDFLAGS-}"

# Build Chrome
echo "[+] patch Chrome"
export PATH="$ROOT_PATH/third_party/depot_tools:$PATH"
mkdir -p $ROOT_PATH/third_party/chromium

pushd $ROOT_PATH/third_party/chromium

fetch --no-history chromium

gclient sync
sleep 10
gclient sync

sleep 2

cd src/
git fetch origin 0230ae0e970eca9af194287c0c05b6ef1e0283bd
git checkout 0230ae0e970eca9af194287c0c05b6ef1e0283bd
./build/install-build-deps.sh
gclient sync

export EDITOR=true
gn args out/Release --args="is_debug=false is_component_build=false symbol_level=0 use_suid_sandbox=true"
gn gen out/Release


echo "[+] patch chrome"
pushd $ROOT_PATH/artifacts/6.Evaluation/6.1.Attack/Attacker_Controlled_GPU_Kernel/ > /dev/null

make
sudo mkdir -p /opt/cudahook
sudo cp libattack.so /opt/cudahook/

popd > /dev/null

cd $ROOT_PATH/third_party/chromium/src/

patch -p1 < $ROOT_PATH/artifacts/6.Evaluation/6.1.Attack/Attacker_Controlled_GPU_Kernel/chrome-patch/chrome.patch

autoninja -C out/Release chrome

# for turning on sandbox on 24.04
# echo 0 | sudo tee /proc/sys/kernel/apparmor_restrict_unprivileged_userns

#out/Release/Chrome --enable-features=Vulkan

popd > /dev/null

export CC=gcc-12
export CXX=g++-12

# MITIGATION
# LLVM patch
echo "Patch and Build Mitgation"
echo "[+] patch llvm-17.0.6 for mitigation"
pushd $ROOT_PATH/third_party/llvm-project-17.0.6-patch/ > /dev/null
git checkout 6009708b4367171ccdbf4b5905cb6a803753fe18
patch -p1 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/LLVM/mitigation-llvm-6009708b4367171ccdbf4b5905cb6a803753fe18-17-forPyTorch-v4.patch
mkdir -p build
cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;llvm;" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_TERMINFO=ON \
../llvm
ninja
popd > /dev/null

echo "[+] patch llvm-19.1.7 for mitigation"
pushd $ROOT_PATH/third_party/llvm-project-19.1.7/ > /dev/null
git checkout cd708029e0b2869e80abe31ddb175f7c35361f90
patch -p1 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/LLVM/mitigation-llvm-cd708029e0b2869e80abe31ddb175f7c35361f90-v4.patch
mkdir -p build
cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;llvm;" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_TERMINFO=ON \
../llvm
ninja
popd > /dev/null


# OPEN GPU KERNEL MODULES patch
echo "[+] patch open-gpu-kernel-modules for mitigation"
pushd $ROOT_PATH/third_party/open-gpu-kernel-modules-SHELL/ > /dev/null
git checkout 8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f
patch -p1 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.3.Runtime_Enforcement/mitigation-driver-8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f.patch
make modules -j$(nproc) > /dev/null
popd > /dev/null



echo "[+] Preparing performance evaluation codes"


pushd $ROOT_PATH/artifacts/6.Evaluation/6.3.Performance/end-to-end > /dev/null

cd raw_1hr_all/
cat e5.acca* | tar zxvf -
cd ../

mkdir -p binary_1hr_all
python era5_preicp_nc_to_bin.py

popd > /dev/null


echo "[+] patch /usr/include/x86_64-linux-gnu/sys/mman.h"
pushd /usr/include/x86_64-linux-gnu/sys/ > /dev/null
sudo patch -p0 < $ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/LLVM/mman.patch
popd > /dev/null


pushd $ROOT_PATH/third_party/pytorch


export CC="$CLANG_BIN/clang"
export CXX="$CLANG_BIN/clang++"

# See if .pytorch exist
if [ -d ".pytorch_patch" ]; then
  echo "[+] .pytorch directory already exists, skipping venv creation"
else
  python3.12 -m venv .pytorch_patch
  echo "[+] creating virtual environment for pyTorch"
fi

source .pytorch_patch/bin/activate

pip3 install -r requirements.txt

mkdir build-patch
cd build-patch
echo "[+] configuring pyTorch"

PREFIX="${VIRTUAL_ENV:-$PWD/_cmake_prefix}"
mkdir -p "$PREFIX"

ORIG_LDFLAGS="${LDFLAGS-}"
export TORCH_NVCC_FLAGS="-G"
export CLANG_BIN="$ROOT_PATH/third_party/llvm-project-17.0.6-patch/build/bin"
export CC="$CLANG_BIN/clang"
export CXX="$CLANG_BIN/clang++"
export TORCH_LIBDIR="$PREFIX/lib"
export LDFLAGS="-L${TORCH_LIBDIR} -Wl,-rpath,${TORCH_LIBDIR} ${LDFLAGS-}"
export LIBRARY_PATH="${TORCH_LIBDIR}:${LIBRARY_PATH-}"

cmake ..  \
  -G Ninja \
  -DUSE_NCCL=0 \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DCMAKE_INSTALL_RPATH=${PREFIX}/lib \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON ${CMAKE_ARGS:-} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_CUSTOM_PROTOBUF=OFF \
  -DCMAKE_C_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6-patch/build/bin/clang \
  -DCMAKE_CXX_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6-patch/build/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=$ROOT_PATH/third_party/llvm-project-17.0.6-patch/build/bin/clang++ \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_FLAGS="-Wno-vla-extension \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6-patch/openmp/build/runtime/src \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6-patch/offload/include/OpenMP/ \
    -O0 -Wno-unknown-warning-option \
    -Wno-extra-semi \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/usr/include/c++/12/ \
    -Wno-c++98-compat-extra-semi \
    -Wno-dev \
    -Wno-deprecated-literal-operator \
    -Wno-deprecated-copy \
    -Wno-sign-compare \
    -Wno-unused-function \
    -Wl,-rpath,$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/\
    -L$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ -lshell \
    -Wno-unused-command-line-argument \
    -ljemalloc -O0" \
  -DCMAKE_CXX_FLAGS="-Wno-vla-extension \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6-patch/openmp/build/runtime/src \
    -I/$ROOT_PATH/third_party/llvm-project-17.0.6-patch/offload/include/OpenMP/ \
    -O0 -Wno-unknown-warning-option \
    -Wno-extra-semi \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/usr/include/c++/12/ \
    -Wno-c++98-compat-extra-semi \
    -Wno-dev \
    -Wno-deprecated-literal-operator \
    -Wno-deprecated-copy \
    -Wno-sign-compare \
    -Wno-unused-function \
    -Wl,-rpath,$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/\
    -L$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ -lshell \
    -Wno-unused-command-line-argument \
    -ljemalloc -O0" \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler \
                     -Xcompiler -Wno-vla-extension \
                     -Xcompiler \"-Wl,-rpath,$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/\"\
                     -L$ROOT_PATH/artifacts/4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ -lshell \
                     -ljemalloc -O0 \
                     -Wno-extra-semi \
                     -I$ROOT_PATH/third_party/llvm-project-17.0.6-patch/openmp/build/runtime/src \
                     -O0 -Wno-unknown-warning-option \
                     -I/usr/include/x86_64-linux-gnu/c++/12/ \
                     -Xcompiler -O0 \
                     -Xcompiler -mavx512f \
                     -Xcompiler -mavx512bw \
                     -Xcompiler -mavx512vl \
                     -I/usr/include/c++/12/ \
                     -Xcompiler -Wno-c++98-compat-extra-semi \
                     -Xcompiler -Wno-dev \
                     -Xcompiler -Wno-deprecated-literal-operator \
                     -Xcompiler -Wno-deprecated-copy \
                     -Xcompiler -Wno-unused-command-line-argument \
                     -Xcompiler -Wno-unused-function" \
  -DUSE_FBGEMM=0 \
  -DUSE_KINETO=0 \
  -DUSE_XNNPACK=0 \
  -DUSE_GLOO=0 \
  -DUSE_TENSORPIPE=0 \
  -DUSE_OBSERVERS=0 \
  -DUSE_DISTRIBUTED=0 \
  -DUSE_SYSTEM_XNNPACK=0 \
  -DUSE_CUDA=1 \
  -DUSE_MKLDNN=0 \
  -DUSE_OPENMP=0 \
  -DUSE_PTHREADPOOL=0 \
  -DUSE_CUDNN=0 \
  -DUSE_SYSTEM_FLATBUFFERS=0 \
  -DUSE_FLASH_ATTENTION=0 \
  -DUSE_MEM_EFF_ATTENTION=0 \
  -DUSE_ITT=0 \
  -DUSE_LITE_PROTO=0 \
  -DINTERN_DISABLE_ONNX=0 \
  -DTORCH_CUDA_ARCH_LIST="8.0" \
  -DOpenMP_C_FLAGS="-fopenmp" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/usr/lib/x86_64-linux-gnu/libomp.so.5"

echo "[+] building pyTorch"

ninja

cd ../

USE_CUDA=1 \
USE_CUDNN=0 \
USE_ROCM=0 \
python setup.py develop

echo "[+] installing pytorch"
pip install -e . --no-build-isolation
pip install netCDF4 tqdm
pip install pwntools

mkdir -p torch/bin
mkdir -p torch/lib

cp -r .pytorch_patch/bin/* torch/bin/
cp -r .pytorch_patch/lib/* torch/lib/

# need LD_PRELOAD of jemalloc when run pytorch
#LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
#python
popd > /dev/null

deactivate


sudo systemctl restart gdm

'''
export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda-12.8/bin:$PATH"

to your .bashrc
'''
