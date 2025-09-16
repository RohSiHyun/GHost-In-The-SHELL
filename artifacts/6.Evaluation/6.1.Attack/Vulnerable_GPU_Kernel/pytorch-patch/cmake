git clone https://github.com/pytorch/pytorch.git -b v2.6.0
cd pytorch
git submodule update --init --recursive
mkdir build
cd build

sudo apt install libeigen3-dev



cmake ..  \
  -G Ninja \
  -DUSE_NCCL=0 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_CUSTOM_PROTOBUF=OFF \
  -DCMAKE_C_COMPILER=/home/lafi/git/llvm-17/llvm-project/build/bin/clang \
  -DCMAKE_CXX_COMPILER=/home/lafi/git/llvm-17/llvm-project/build/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/home/lafi/git/llvm-17/llvm-project/build/bin/clang++ \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_C_FLAGS="-Wno-vla-extension \
    -Wno-error=vla-cxx-extension \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/home/lafi/git/llvm-project-patch/openmp/build/runtime/src \
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
    -Wl,-rpath,/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/\
    -L/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/ -lshell \
    -Wno-unused-command-line-argument \
    -ljemalloc -O0" \
  -DCMAKE_CXX_FLAGS="-Wno-vla-extension \
    -Wno-error=vla-cxx-extension \
    -I/usr/include/x86_64-linux-gnu/c++/12/ \
    -I/home/lafi/git/llvm-project-patch/openmp/build/runtime/src \
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
    -Wl,-rpath,/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/\
    -L/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/ -lshell \
    -Wno-unused-command-line-argument \
    -ljemalloc -O0" \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler \
                     -Xcompiler -Wno-vla-extension \
                     -Xcompiler -Wno-error=vla-cxx-extension \
                     -Xcompiler \"-Wl,-rpath,/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/\"\
                     -L/home/lafi/git/hmm-attack/poc/hmm-mitigation/mitigation_3/libshell/ -lshell \
                     -ljemalloc -O0 \
                     -Wno-extra-semi \
                     -I/home/lafi/git/llvm-project-patch/openmp/build/runtime/src \
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
  -DUSE_XNNPACK=1 \
  -DUSE_SYSTEM_XNNPACK=0 \
  -DUSE_CUDA=1 \
  -DUSE_PTHREADPOOL=0 \
  -DUSE_CUDNN=1 \
  -DUSE_OPENMP=1 \
  -DUSE_LITE_PROTO=1 \
  -DINTERN_DISABLE_ONNX=0 \
  -DTORCH_CUDA_ARCH_LIST="8.0" \
  -DOpenMP_C_FLAGS="-fopenmp" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/usr/lib/x86_64-linux-gnu/libomp.so.5"

  ninja