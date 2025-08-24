# Compiler: NVIDIA CUDA C++ Compiler
CXX = /export/apps/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/bin/nvcc
CXXFLAGS = -O3 -std=c++17 -arch=sm_80 --extended-lambda
TARGET = monte_carlo_gpu
SRCS = main.cu
LIBS = -lcurand

CPU_CXX = g++
CPU_CXXFLAGS = -O3 -std=c++11
CPU_TARGET = monte_carlo_cpu
CPU_SRCS = main_cpu.cpp

MPI_CXX = mpicxx
MPI_CXXFLAGS = -O3 -std=c++11
MPI_TARGET = monte_carlo_mpi
MPI_SRCS = main_mpi.cpp

# Default target builds all
all: gpu cpu mpi

gpu: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

cpu: $(CPU_SRCS)
	$(CPU_CXX) $(CPU_CXXFLAGS) -o $(CPU_TARGET) $(CPU_SRCS)

mpi: $(MPI_SRCS)
	$(MPI_CXX) $(MPI_CXXFLAGS) -o $(MPI_TARGET) $(MPI_SRCS)

clean:
	rm -f $(TARGET) $(CPU_TARGET) $(MPI_TARGET)
