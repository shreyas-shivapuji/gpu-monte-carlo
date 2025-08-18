#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// helper function to check for CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t err_code = (err); \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel to initialize cuRAND states
__global__ void setup_kernel(curandState* states, unsigned long long seed, int num_simulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_simulations) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel for MC simulation
__global__ void monte_carlo_kernel(double S, double K, double T, double r, double sigma, int num_simulations, double* d_payoffs, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_simulations) {
        curandState* localState = &states[idx];
        double Z = curand_normal_double(localState);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        d_payoffs[idx] = fmax(ST - K, 0.0);
    }
}

// host fn to start the GPU MC simulation
double monte_carlo_vanilla_call_gpu(double S, double K, double T, double r, double sigma, int num_simulations) {
    double* d_payoffs;
    curandState* d_states;
    CUDA_CHECK(cudaMalloc((void**)&d_payoffs, num_simulations * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_states, num_simulations * sizeof(curandState)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_simulations + threadsPerBlock - 1) / threadsPerBlock;

    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL), num_simulations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    monte_carlo_kernel<<<blocksPerGrid, threadsPerBlock>>>(S, K, T, r, sigma, num_simulations, d_payoffs, d_states);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_payoffs(num_simulations);
    CUDA_CHECK(cudaMemcpy(h_payoffs.data(), d_payoffs, num_simulations * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_payoffs));
    CUDA_CHECK(cudaFree(d_states));

    double total_payoff = 0.0;
    for (int i = 0; i < num_simulations; ++i) {
        total_payoff += h_payoffs[i];
    }
    return exp(-r * T) * (total_payoff / num_simulations);
}

// calculate the price of an option using BS
double black_scholes_vanilla_call(double S, double K, double T, double r, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return (S * 0.5 * (1 + erf(d1 / sqrt(2))) - K * exp(-r * T) * 0.5 * (1 + erf(d2 / sqrt(2))));
}

int main() {
    double S, K, T, r, sigma;

    // --- Manual User Input for Market Data ---
    std::cout << "Enter market data:" << std::endl;
    std::cout << "  - Current Stock Price (S): ";
    std::cin >> S;
    std::cout << "  - Risk-Free Interest Rate (r): ";
    std::cin >> r;
    std::cout << "  - Volatility (sigma): ";
    std::cin >> sigma;

    // --- Manual User Input for Option Parameters ---
    std::cout << "\nEnter option parameters:" << std::endl;
    std::cout << "  - Strike Price (K): ";
    std::cin >> K;
    std::cout << "  - Time to Maturity in Years (T): ";
    std::cin >> T;

    std::vector<int> simulations = {100000, 1000000, 10000000, 50000000, 100000000};
    
    double bs_price = black_scholes_vanilla_call(S, K, T, r, sigma);
    std::cout << "\n--- Black-Scholes Analytical Price ---" << std::endl;
    std::cout << "Call Option Price: " << std::fixed << std::setprecision(4) << bs_price << std::endl << std::endl;
    
    std::cout << "--- GPU Monte Carlo Simulation (CUDA C++) ---" << std::endl;
    std::cout << std::setw(15) << "Simulations" << " | " 
              << std::setw(12) << "MC Price" << " | " 
              << std::setw(20) << "Execution Time (s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int n_sims : simulations) {
        auto start = std::chrono::high_resolution_clock::now();
        double mc_price_gpu = monte_carlo_vanilla_call_gpu(S, K, T, r, sigma, n_sims);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_time = end - start;
        
        std::cout << std::setw(15) << n_sims << " | " 
                  << std::setw(12) << std::fixed << std::setprecision(4) << mc_price_gpu << " | " 
                  << std::setw(20) << std::fixed << std::setprecision(4) << gpu_time.count() << std::endl;
    }

    return 0;
}
