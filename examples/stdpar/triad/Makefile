# Makefile for compiling and running the C++ program with nvc++ for NVIDIA GPUs

# Compiler and flags
CXX = nvc++
#CXXFLAGS = -stdpar -std=c++17 -mp -gpu=cuda12.6 -Wall -Wextra
CXXFLAGS = -stdpar -std=c++17 -mp -Wall -Wextra

# Source file and executable name
SRCS = triad.cc
OBJS = $(SRCS:%.cc=%.o)
DEPS = $(SRCS:%.cc=%.d)
TARGET = triad

# Default target: build the executable
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.d: %.cc
	$(CXX) $(CXXFLAGS) -M $< > $@

# Rule to run the executable
run: $(TARGET)
	$(CURDIR)/$(TARGET)

# Rule to clean up (remove the executable)
clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

# Phony targets (not actual files)
.PHONY: all run clean

-include $(DEPS)
