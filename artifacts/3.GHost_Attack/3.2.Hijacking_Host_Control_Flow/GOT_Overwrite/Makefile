NVCC := nvcc

# Source file and output
SRC    := got-overwrite.cu
TARGET := got-overwrite

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $<

clean:
	rm -f $(TARGET)
