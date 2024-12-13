import numpy as np
import matplotlib.pyplot as plt
from mainv11 import *
import math
def bootstrap_sample(data, seed=None):

    if seed is not None:
        random.seed(seed)

    n = len(data)
    indices = list(range(n))
    train_indices = [random.choice(indices) for _ in range(n)]
    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in indices if i not in set(train_indices)]
    return train_set, test_set

def plot_initial_iteration(fractions, pair_quality, pair_completeness, f1_scores, title="Initial Iteration Results"):
    """
    Plots PQ, PC, and F1-score vs Fraction of Comparisons Made for the initial iteration.
    """
    plt.figure(figsize=(12, 8))

    # PQ Plot
    plt.subplot(3, 1, 1)
    plt.plot(fractions, pair_quality, marker='o', label="Pair Quality (PQ)")
    plt.xlabel("Fraction of Comparisons Made")
    plt.ylabel("Pair Quality (PQ)")
    plt.title(f"{title} - Pair Quality vs Fraction of Comparisons")
    plt.legend()

    # PC Plot
    plt.subplot(3, 1, 2)
    plt.plot(fractions, pair_completeness, marker='o', label="Pair Completeness (PC)")
    plt.xlabel("Fraction of Comparisons Made")
    plt.ylabel("Pair Completeness (PC)")
    plt.title(f"{title} - Pair Completeness vs Fraction of Comparisons")
    plt.legend()

    # F1-Score Plot
    plt.subplot(3, 1, 3)
    plt.plot(fractions, f1_scores, marker='o', label="F1-Score")
    plt.xlabel("Fraction of Comparisons Made")
    plt.ylabel("F1-Score")
    plt.title(f"{title} - F1-Score vs Fraction of Comparisons")
    plt.legend()

    plt.tight_layout()
    plt.show()

def bootstrap_and_plot_with_fractions(products, num_hashes, n_bootstrap_samples, shingle_sizes, thresholds, weight_combinations):
    bands_list = [5, 10, 15, 30, 50]  # Updated band configurations

    results_table = []  # To store results for the table
    score_details = []  # To track metrics and their configurations
    initial_fractions, initial_pqs, initial_pcs, initial_f1_scores = [], [], [], []  # To store first iteration data

    for shingle_size in shingle_sizes:
        for threshold in thresholds:
            for idx, weights in enumerate(weight_combinations):
                for num_bands in bands_list:
                    pqs_temp, pcs_temp, f1_scores_temp, fractions_temp = [], [], [], []
                    f1_star_scores_temp = []  # For F1* scores

                    for bootstrap_iter in range(n_bootstrap_samples):
                        train_set, test_set = bootstrap_sample(products)
                        true_pairs = get_official_modelID_pairs(test_set)
                        vocabulary = build_vocabulary(test_set)
                        binary_matrix = create_binary_matrix(test_set, vocabulary)
                        hash_functions = generate_hash_functions(num_hashes, len(vocabulary))
                        signature_matrix = compute_minhash_signature(binary_matrix, hash_functions)
                        candidate_pairs = banding_with_shop_and_brand_filter(signature_matrix, num_bands, test_set)
                        fraction = len(candidate_pairs) / math.comb(len(test_set), 2)
                        filtered_pairs = filter_candidate_pairs_with_shingles(test_set, candidate_pairs, shingle_size, threshold)
                        metrics = compute_evaluation_metrics(true_pairs, filtered_pairs, candidate_pairs)
                        pqs_temp.append(metrics["pair_quality"])
                        pcs_temp.append(metrics["pair_completeness"])
                        f1_scores_temp.append(metrics["f1_score"])
                        f1_star_scores_temp.append(metrics["f1_star"])
                        fractions_temp.append(fraction)

                        # Store data for the first iteration
                        if bootstrap_iter == 0:
                            initial_fractions.append(fraction)
                            initial_pqs.append(metrics["pair_quality"])
                            initial_pcs.append(metrics["pair_completeness"])
                            initial_f1_scores.append(metrics["f1_score"])

                    avg_f1_score = np.mean(f1_scores_temp)
                    avg_f1_star_score = np.mean(f1_star_scores_temp)
                    avg_pq = np.mean(pqs_temp)
                    avg_pc = np.mean(pcs_temp)
                    avg_fraction = np.mean(fractions_temp)
                    combined_metric = (2 * avg_f1_score * avg_f1_star_score) / (avg_f1_score + avg_f1_star_score) if (avg_f1_score + avg_f1_star_score) > 0 else 0

                    config_label = f"{num_bands} bands, Shingle: {shingle_size}, Threshold: {threshold}, Weights: {weights}"
                    results_table.append([
                        config_label,
                        f"{avg_f1_score:.4f}",
                        f"{avg_f1_star_score:.4f}",
                        f"{avg_pq:.4f}",
                        f"{avg_pc:.4f}",
                        f"{avg_fraction:.4f}"
                    ])

                    score_details.append({
                        "f1_score": avg_f1_score,
                        "f1_star_score": avg_f1_star_score,
                        "combined_metric": combined_metric,
                        "config": config_label,
                        "fractions": fractions_temp,
                        "pair_quality": pqs_temp,
                        "pair_completeness": pcs_temp,
                        "f1_score_values": f1_scores_temp
                    })

    best_f1 = max(score_details, key=lambda x: x["f1_score"])
    best_f1_star = max(score_details, key=lambda x: x["f1_star_score"])
    best_combined = max(score_details, key=lambda x: x["combined_metric"])

    print("\nSummary of Results:")
    headers = ["Configuration", "Average F1 Score", "Average F1* Score", "Pair Quality (PQ)",
               "Pair Completeness (PC)", "Fraction of Comparisons"]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

    print("\nBest Configurations:")
    print(f"Highest F1 Score: {best_f1['f1_score']:.4f}, Configuration: {best_f1['config']}")
    print(f"Highest F1* Score: {best_f1_star['f1_star_score']:.4f}, Configuration: {best_f1_star['config']}")
    print(f"Highest Combined Metric: {best_combined['combined_metric']:.4f}, Configuration: {best_combined['config']}")

    # Plot results for the initial iteration
    plot_initial_iteration(initial_fractions, initial_pqs, initial_pcs, initial_f1_scores, title="Initial Iteration Results")


    #Discarded version of plotting.
    # Plot results for the best configurations
    #plot_metrics_for_best_configuration(best_f1, "Highest F1 Score")
    #plot_metrics_for_best_configuration(best_f1_star, "Highest F1* Score")
    #plot_metrics_for_best_configuration(best_combined, "Highest Combined Metric")

if __name__ == "__main__":
    # Define parameters
    num_hashes = 150
    #shingle_sizes = [2, 3, 4]
    shingle_sizes = [3]
    #thresholds = [0.6, 0.7, 0.8]
    thresholds = [ 0.7]
    n_bootstrap_samples = 5

    weight_combinations = [
       # (0.5, 0.3, 0.15, 0.05),
       #(0.4, 0.35, 0.2, 0.05),
        (0.3, 0.5, 0.15, 0.05),
        #(0.35, 0.4, 0.2, 0.05),
       # (0.4, 0.3, 0.25, 0.05)
    ]

    products = load_data("TVs-all-merged.json")
    bootstrap_and_plot_with_fractions(products, num_hashes, n_bootstrap_samples, shingle_sizes, thresholds, weight_combinations)
