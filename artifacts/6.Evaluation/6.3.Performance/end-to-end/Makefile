# Makefile to build SHELL and baseline with clang++

CLANG       := ../../../../third_party/llvm-project-17.0.6/build/bin/clang++
CLANG-PATCH := ../../../../third_party/llvm-project-17.0.6-patch/build/bin/clang++
SRC_SHELL   := weather_app-SHELL.cu
SRC_BASE    := weather_app-baseline.cu
TARGETS     := SHELL baseline

# Common flags
CXXFLAGS    := -x cuda -std=c++14 -O0
CUDAFLAGS   := --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_89 \
               -I/usr/local/cuda-12.8/include
LIBS        := -lcudart -ljemalloc

# Special linker flags for SHELL build
SHELL_LDFLAGS := -Wl,-rpath,../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
                 -L../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
                 -L/usr/local/cuda-12.8/lib64 \
                 -lshell
# Baseline only needs CUDA lib dir
BASE_LDFLAGS  := -L/usr/local/cuda-12.8/lib64

all: $(TARGETS)

SHELL: $(SRC_SHELL)
	$(CLANG-PATCH) $(SRC_SHELL) -o $@ $(CXXFLAGS) -g $(CUDAFLAGS) $(SHELL_LDFLAGS) $(LIBS)

baseline: $(SRC_BASE)
	$(CLANG) $(SRC_BASE) -o $@ $(CXXFLAGS) $(CUDAFLAGS) $(BASE_LDFLAGS) $(LIBS)

clean:
	$(RM) $(TARGETS)
