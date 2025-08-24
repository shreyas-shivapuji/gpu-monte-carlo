#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>

double black_scholes_vanilla_call(double S, double K, double T, double r, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return (S * 0.5 * (1 + erf(d1 / sqrt(2))) - K * exp(-r * T) * 0.5 * (1 + erf(d2 / sqrt(2))));
}

double monte_carlo_vanilla_call_cpu(double S, double K, double T, double r, double sigma, int num_simulations) {
    std::mt19937_64 rng;
    rng.seed(std::random_device{}());
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    double total_payoff = 0.0;
    for (int i = 0; i < num_simulations; ++i) {
        double Z = norm_dist(rng);
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        total_payoff += std::max(ST - K, 0.0);
    }
    return exp(-r * T) * (total_payoff / num_simulations);
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

    std::cout << "--- CPU Monte Carlo Simulation (C++) ---" << std::endl;
    std::cout << std::setw(15) << "Simulations" << " | "
              << std::setw(12) << "MC Price" << " | "
              << std::setw(20) << "Execution Time (s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int n_sims : simulations) {
        auto start = std::chrono::high_resolution_clock::now();
        double mc_price_cpu = monte_carlo_vanilla_call_cpu(S, K, T, r, sigma, n_sims);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = end - start;

        std::cout << std::setw(15) << n_sims << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << mc_price_cpu << " | "
                  << std::setw(20) << std::fixed << std::setprecision(4) << cpu_time.count() << std::endl;
    }

    return 0;
}
