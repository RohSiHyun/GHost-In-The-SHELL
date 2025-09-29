# Makefile to build mmap-eval and mmap-patch-eval with different clang++

CLANG_PATCH := $(pwd)/../../../../../third_party/llvm-project-17.0.6-patch/build/bin/clang++
CLANG_BASE  := $(pwd)/../../../../../third_party/llvm-project-17.0.6/build/bin/clang++
SRC         := pf.cu
TARGETS     := pf-eval pf-patch-eval

CXXFLAGS    := -x cuda -std=c++14 -O0
CUDAFLAGS   := --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_89 \
               -I/usr/local/cuda-12.8/include
LDFLAGS     := -Wl,-rpath,$(pwd)/../../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L$(pwd)/../../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L/usr/local/cuda-12.8/lib64
LIBS        := -lshell -lcudart -ljemalloc

all: $(TARGETS)

pf-eval: $(SRC)
	$(CLANG_BASE) $(SRC) -o $@ $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $(LIBS)

pf-patch-eval: $(SRC)
	$(CLANG_PATCH) $(SRC) -o $@ $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(TARGETS)
