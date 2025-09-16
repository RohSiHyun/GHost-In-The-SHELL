# Makefile to build libmycudahook.so

# Compiler
NVCC        := nvcc

# Flags
NVCCFLAGS   := -Xcompiler "-fno-stack-protector -fPIC -fcf-protection=none -fno-omit-frame-pointer"
LDFLAGS     := -shared
LIBS        := -lcudart

# Sources and output
SRC         := attacker_kernel.cu
TARGET      := libattack.so

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -arch=sm_89 -o $@ $^ $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)

