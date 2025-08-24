#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

__global__ void monte_carlo_kernel(double S, double K, double T, double r, double sigma, int n, double *results, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curandState state;
    curand_init(seed, tid, 0, &state);
    double Z = curand_normal_double(&state);
    double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
    results[tid] = fmax(ST - K, 0.0);
}

double black_scholes_vanilla_call(double S, double K, double T, double r, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return (S * 0.5 * (1 + erf(d1 / sqrt(2))) - K * exp(-r * T) * 0.5 * (1 + erf(d2 / sqrt(2))));
}

double monte_carlo_vanilla_call_gpu(double S, double K, double T, double r, double sigma, int num_simulations, float& milliseconds) {
    double *d_results;
    double *h_results = new double[num_simulations];
    cudaMalloc((void**)&d_results, num_simulations * sizeof(double));

    int blockSize = 256;
    int gridSize = (num_simulations + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    monte_carlo_kernel<<<gridSize, blockSize>>>(S, K, T, r, sigma, num_simulations, d_results, time(NULL));
    cudaMemcpy(h_results, d_results, num_simulations * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    double sum_payoff = 0.0;
    for (int i = 0; i < num_simulations; ++i) {
        sum_payoff += h_results[i];
    }
    double price = exp(-r * T) * (sum_payoff / num_simulations);

    cudaFree(d_results);
    delete[] h_results;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return price;
}

int main() {
    double S, K, T, r, sigma;

    std::cout << "Enter market data:" << std::endl;
    std::cout << "  - Current Stock Price (S): ";
    std::cin >> S;
    std::cout << "  - Risk-Free Interest Rate (r): ";
    std::cin >> r;
    std::cout << "  - Volatility (sigma): ";
    std::cin >> sigma;

    std::cout << "\nEnter option parameters:" << std::endl;
    std::cout << "  - Strike Price (K): ";
    std::cin >> K;
    std::cout << "  - Time to Maturity in Years (T): ";
    std::cin >> T;

    std::vector<int> simulations = {100000, 1000000, 10000000, 50000000, 100000000};

    double bs_price = black_scholes_vanilla_call(S, K, T, r, sigma);
    std::cout << "\n--- Black-Scholes Analytical Price ---" << std::endl;
    std::cout << "Call Option Price: " << std::fixed << std::setprecision(4) << bs_price << std::endl << std::endl;

    std::cout << "--- GPU Monte Carlo Simulation (CUDA) ---" << std::endl;
    std::cout << std::setw(15) << "Simulations" << " | "
              << std::setw(12) << "MC Price" << " | "
              << std::setw(25) << "GPU Execution Time (s)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (int n_sims : simulations) {
        float milliseconds = 0.0;
        double price = monte_carlo_vanilla_call_gpu(S, K, T, r, sigma, n_sims, milliseconds);
        std::cout << std::setw(15) << n_sims << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << price << " | "
                  << std::setw(25) << std::fixed << std::setprecision(4) << milliseconds / 1000.0 << std::endl;
    }

    return 0;
}
