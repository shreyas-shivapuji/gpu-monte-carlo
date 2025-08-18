# Compiler: NVIDIA CUDA C++ Compiler
CXX = /export/apps/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/bin/nvcc

# Compiler flags
CXXFLAGS = -O3 -std=c++17 -arch=sm_80 --extended-lambda

# Target executable name
TARGET = monte_carlo_gpu

# Source file (now a .cu file)
SRCS = main.cu

# Libraries to link against
LIBS = -lcurand

# Default target
all: $(TARGET)

# Rule to compile and link the CUDA code
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Rule to clean up the build directory
clean:
	rm -f $(TARGET)
