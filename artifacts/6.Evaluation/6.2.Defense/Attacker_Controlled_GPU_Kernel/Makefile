# Makefile to build libattack.so with clang++

CLANG       := ../../../../third_party/llvm-project-17.0.6-patch/build/bin/clang++
SRC         := attacker_kernel.cu
TARGET      := libattack.so

CXXFLAGS    := -x cuda -std=c++14 -O0 \
               -fno-stack-protector -fPIC -fcf-protection=none -fno-omit-frame-pointer -shared \
               -I/usr/local/cuda-12.8/include

LDFLAGS     := -Wl,-rpath,../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L../../../4.SHELL_Mitigation/4.2.Static_Instrumentation/libSHELL/ \
               -L/usr/local/cuda-12.8/lib64

LIBS        := -lshell -lcudart -ljemalloc

CUDAFLAGS   := --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_89

all: $(TARGET)

$(TARGET): $(SRC)
	$(CLANG) $(SRC) -o $(TARGET) $(CXXFLAGS) $(CUDAFLAGS) $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(TARGET)
