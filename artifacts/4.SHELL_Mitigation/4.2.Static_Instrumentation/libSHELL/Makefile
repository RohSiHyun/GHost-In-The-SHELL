# Name of the shared library to build
TARGET = libshell.so

# Source files
SRC = libshell.c

# Compiler and flags
CC = gcc
CFLAGS = -Wall -fPIC -O2
LDFLAGS = -shared
LDLIBS = -ljemalloc

# Output
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c instru.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
