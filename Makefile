# Compiler: NVIDIA CUDA C++ Compiler
CXX = /export/apps/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/bin/nvcc

# compiler flags
CXXFLAGS = -O3 -std=c++17 -arch=sm_80 --extended-lambda

TARGET = monte_carlo_gpu

SRCS = main.cu

# libraries
LIBS = -lcurand

# default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
