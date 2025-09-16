NVCC := nvcc

# Source file and output
SRC    := hmm-aslr-break.cu
TARGET := hmm-aslr-break

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $<

clean:
	rm -f $(TARGET)
