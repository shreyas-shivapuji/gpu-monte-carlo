#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <mpi.h>

double black_scholes_vanilla_call(double S, double K, double T, double r, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return (S * 0.5 * (1 + erf(d1 / sqrt(2))) - K * exp(-r * T) * 0.5 * (1 + erf(d2 / sqrt(2))));
}

double monte_carlo_vanilla_call_mpi(double S, double K, double T, double r, double sigma, int num_simulations, int rank, int size) {
    std::mt19937_64 rng(rank + time(NULL));
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    int local_sim = num_simulations / size;
    int remainder = num_simulations % size;
    if (rank < remainder) local_sim++;

    double local_total_payoff = 0.0;
    for (int i = 0; i < local_sim; ++i) {
        double Z = norm_dist(rng);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        local_total_payoff += std::max(ST - K, 0.0);
    }
    double global_total_payoff = 0.0;
    MPI_Reduce(&local_total_payoff, &global_total_payoff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_total_payoff; // Only valid on rank 0
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double S, K, T, r, sigma;

    if (rank == 0) {
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
    }
    MPI_Bcast(&S, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> simulations = {100000, 1000000, 10000000, 50000000, 100000000};

    if (rank == 0) {
        double bs_price = black_scholes_vanilla_call(S, K, T, r, sigma);
        std::cout << "\n--- Black-Scholes Analytical Price ---" << std::endl;
        std::cout << "Call Option Price: " << std::fixed << std::setprecision(4) << bs_price << std::endl << std::endl;

        std::cout << "--- MPI Monte Carlo Simulation (C++/MPI) ---" << std::endl;
        std::cout << std::setw(15) << "Simulations" << " | "
                  << std::setw(12) << "MC Price" << " | "
                  << std::setw(20) << "Execution Time (s)" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
    }

    for (int n_sims : simulations) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        double global_total_payoff = monte_carlo_vanilla_call_mpi(S, K, T, r, sigma, n_sims, rank, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mpi_time = end - start;

        if (rank == 0) {
            double mc_price_mpi = exp(-r * T) * (global_total_payoff / n_sims);
            std::cout << std::setw(15) << n_sims << " | "
                      << std::setw(12) << std::fixed << std::setprecision(4) << mc_price_mpi << " | "
                      << std::setw(20) << std::fixed << std::setprecision(4) << mpi_time.count() << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
