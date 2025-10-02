# Makefile to build mmap-eval and mmap-patch-eval with different clang++

CLANG_PATCH := ${PWD}/../../../../../third_party/llvm-project-17.0.6-patch/build/bin/clang++
CLANG_BASE  := ${PWD}/../../../../../third_party/llvm-project-17.0.6/build/bin/clang++
SRC         := mmap.cu
TARGETS     := mmap-eval mmap-patch-eval

CXXFLAGS    := -x cuda -std=c++14 -O0
CUDAFLAGS   := --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_89 \
               -I/usr/local/cuda-12.8/include
LDFLAGS     := -Wl,-rpath,${PWD}/../../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L${PWD}/../../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L/usr/local/cuda-12.8/lib64
LIBS        := -lshell -lcudart -ljemalloc

all: $(TARGETS)

mmap-eval: $(SRC)
	$(CLANG_BASE) $(SRC) -o $@ $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $(LIBS)

mmap-patch-eval: $(SRC)
	$(CLANG_PATCH) $(SRC) -o $@ $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(TARGETS)
