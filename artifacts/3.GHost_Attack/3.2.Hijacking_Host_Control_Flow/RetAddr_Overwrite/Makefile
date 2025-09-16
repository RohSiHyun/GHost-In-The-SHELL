NVCC := nvcc

# Source file and output
SRC    := retAddr-overwrite.cu
TARGET := retAddr-overwrite

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $<

clean:
	rm -f $(TARGET)
