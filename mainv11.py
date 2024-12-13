import json
from class_tv_product import TVProduct
import random
import re
import numpy as np
from collections import defaultdict
from tabulate import tabulate


# Functions for data processing and analysis
#Loads the JSON data from a file and structures it into TVProduct instances.
def load_data(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)

    products = []
    for model_id, product_list in data.items():
        for product_data in product_list:
            tv_product = TVProduct(
                shop=product_data["shop"],
                url=product_data["url"],
                modelID=product_data["modelID"],
                features_map=product_data["featuresMap"],
                title=product_data["title"]
            )
            products.append(tv_product)
    return products


    #Finds pairs of TVProducts with the same modelID.
def get_official_modelID_pairs(tv_products):
    model_pairs = set()
    model_to_indices = {}

    for i, tv in enumerate(tv_products):
        if tv.modelID in model_to_indices:
            model_to_indices[tv.modelID].append(i)
        else:
            model_to_indices[tv.modelID] = [i]

    #print("\nModel to Indices Mapping (Sample):")
    # for model_id, indices in list(model_to_indices.items())[:10]:
    #     print(f"{model_id}: {indices}")

    for indices in model_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    model_pairs.add((indices[i], indices[j]))

    return model_pairs



    #Extracts a set of unique brands from the TV products using the 'brand' attribute.
def extract_unique_brands(tv_products):
    return {product.brand.lower() for product in tv_products if product.brand and product.brand != "Unknown"}


    #Extracts a set of unique sizes from the TV products using the 'size' attribute.
def extract_unique_sizes(tv_products):
    return {product.size for product in tv_products if product.size}


    #Extracts potential model-related words from a product title.
def extract_model_words_title(title):
    #note this only takes the title of a single product.
    regex = r'(?=.*[a-zA-Z])(?=.*[0-9])[\w\-]{4,}'
    model_words = re.findall(regex, title)
    model_words = [word for word in model_words if
                   not any(irrelevant in word for irrelevant in ['inch', 'hz', '1080p', '720p', '2160p', '3d-ready', '480hz'])]
    return set(model_words)

    #Extracts potential model-related words from the features_map of TV products.
def extract_model_words_features(tv_products):
    #note this gets all the features values from all products in tv_products
    model_words = set()
    for product in tv_products:
        for key, value in product.features_map.items():
            model_words.update(key.lower().split())
            if isinstance(value, str):
                model_words.update(value.lower().split())
    return model_words

    #Generate shingles (substrings of fixed length) from the given cleaned text.
def generate_shingles(text, shingle_size=3):
    # Generate overlapping substrings (shingles)
    shingles = {text[i:i + shingle_size] for i in range(len(text) - shingle_size + 1)}
    return shingles


    #Builds a vocabulary of unique words from: Unique brands, Unique sizes, Potential model numbers, Model words  titles
def build_vocabulary(tv_products):
    vocabulary = set()

    # Add unique brands
    vocabulary.update(extract_unique_brands(tv_products))

    # Add unique sizes
    vocabulary.update(str(size) for size in extract_unique_sizes(tv_products))

    # Add potential model numbers
    for product in tv_products:
        vocabulary.update(product.potential_model_numbers)

    # Add model words from titles
    for product in tv_products:
        vocabulary.update(extract_model_words_title(product.cleaned_title))

    # adding vocab of the features does nto increase f1 or f1star score.
    # # Add model words from features
    # vocabulary.update(extract_model_words_features(tv_products))

    return sorted(vocabulary)

    #Creates a dense binary matrix for TV products and vocabulary.
def create_binary_matrix(tv_products, vocabulary):
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    binary_matrix = np.zeros((len(tv_products), len(vocabulary)), dtype=int)

    for row_idx, tv in enumerate(tv_products):
        words = set(tv.cleaned_title.split())
        for word in words:
            if word in vocab_index:
                col_idx = vocab_index[word]
                binary_matrix[row_idx, col_idx] = 1

    return binary_matrix

    #Find the next prime number greater than or equal to n.
def next_prime(n):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    while not is_prime(n):
        n += 1
    return n

    #Generate hash functions for MinHash.
def generate_hash_functions(num_hashes, vocab_size):
    p = next_prime(vocab_size)
    hash_functions = [
        (random.randint(1, p - 1), random.randint(0, p - 1), p)
        for _ in range(num_hashes)
    ]
    return hash_functions

    #Compute the MinHash signature matrix for a binary matrix.
def compute_minhash_signature(binary_matrix, hash_functions):
    num_products, vocab_size = binary_matrix.shape
    num_hashes = len(hash_functions)
    signature_matrix = np.full((num_hashes, num_products), np.inf)

    for col_idx in range(vocab_size):
        rows_with_1 = np.where(binary_matrix[:, col_idx] == 1)[0]
        for hash_idx, (a, b, p) in enumerate(hash_functions):
            hash_value = (a * col_idx + b) % p
            signature_matrix[hash_idx, rows_with_1] = np.minimum(signature_matrix[hash_idx, rows_with_1], hash_value)

    return signature_matrix

    #eventually not used, we also want to filter see def below
def banding(signature_matrix, num_bands):

    num_hashes, num_products = signature_matrix.shape
    rows_per_band = num_hashes // num_bands
    assert num_hashes % num_bands == 0, "Number of hash rows must be divisible by num_bands."

    candidate_pairs = set()
    for band_idx in range(num_bands):
        # Extract rows for the current band
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        band = signature_matrix[start_row:end_row, :]

        # Use a dictionary to hash products into buckets
        buckets = {}
        for product_idx in range(num_products):
            # Create a hashable tuple of the band's rows for the product
            band_signature = tuple(band[:, product_idx])
            if band_signature in buckets:
                buckets[band_signature].append(product_idx)
            else:
                buckets[band_signature] = [product_idx]

        # Add all pairs from the same bucket to candidate pairs
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidate_pairs.add((bucket[i], bucket[j]))

    return candidate_pairs


    #Splits the MinHash signature matrix into bands and generates candidate pairs, avoiding intra-shop comparisons.
def banding_with_shop_filter(signature_matrix, num_bands, tv_products):
    num_hashes, num_products = signature_matrix.shape
    rows_per_band = num_hashes // num_bands
    assert num_hashes % num_bands == 0, "Number of hash rows must be divisible by num_bands."

    candidate_pairs = set()
    product_shops = [tv.shop for tv in tv_products]

    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        band = signature_matrix[start_row:end_row, :]

        buckets = defaultdict(list)
        for product_idx in range(num_products):
            band_signature = tuple(band[:, product_idx])
            buckets[band_signature].append(product_idx)

        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        prod_idx1, prod_idx2 = bucket[i], bucket[j]
                        # Only include pairs from different shops
                        if product_shops[prod_idx1] != product_shops[prod_idx2]:
                            candidate_pairs.add((prod_idx1, prod_idx2))

    return candidate_pairs


    #Splits the MinHash signature matrix into bands and generates candidate pairs,
    #avoiding intra-shop comparisons and ensuring brand matches. Tracks potential pairs
    #after the signature matrix, each band, and the final filtering step.
def banding_with_shop_and_brand_filter(signature_matrix, num_bands, tv_products):
    num_hashes, num_products = signature_matrix.shape
    rows_per_band = num_hashes // num_bands
    assert num_hashes % num_bands == 0, "Number of hash rows must be divisible by num_bands."

    # Initial count of potential pairs based on the signature matrix (all combinations)
    total_combinations = (num_products * (num_products - 1)) // 2
    #print(f"Total potential pairs after signature matrix: {total_combinations}")

    candidate_pairs = set()
    product_shops = [tv.shop for tv in tv_products]
    product_brands = [tv.brand.lower() for tv in tv_products]

    # Track pairs after each banding step
    for band_idx in range(num_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band
        band = signature_matrix[start_row:end_row, :]

        buckets = defaultdict(list)
        for product_idx in range(num_products):
            band_signature = tuple(band[:, product_idx])
            buckets[band_signature].append(product_idx)

        # Add candidate pairs from the current band
        band_pairs = set()
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        prod_idx1, prod_idx2 = bucket[i], bucket[j]
                        band_pairs.add((prod_idx1, prod_idx2))

        # Update candidate pairs and track progress
        candidate_pairs.update(band_pairs)
        #print(f"Potential pairs after band {band_idx + 1}: {len(candidate_pairs)}")

    # Apply filtering
    filtered_pairs = set()
    for prod_idx1, prod_idx2 in candidate_pairs:
        if (
            product_shops[prod_idx1] != product_shops[prod_idx2] and
            product_brands[prod_idx1] == product_brands[prod_idx2]
        ):
            filtered_pairs.add((prod_idx1, prod_idx2))

    # Print the number of pairs after filtering
    #print(f"Potential pairs after filtering: {len(filtered_pairs)}")

    return filtered_pairs


    #Computes the Jaccard similarity between two lists of tokens.
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# Use when you want to remove spaces, eventually not used.
# def remove_spaces(text):
#     return text.replace(" ", "")

    #Computes the similarity between two products using an MSM approach, including shingle-based similarity.
def compute_msm_similarity_with_shingles(product1, product2, shingle_size=3, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    # use this to clean the titles from spaces, gives lower F scores so we do not use it.
    #shingles1 = generate_shingles(remove_spaces(product1.cleaned_title), shingle_size)
    #shingles2 = generate_shingles(remove_spaces(product2.cleaned_title), shingle_size)

    # Title similarity using shingles
    shingles1 = generate_shingles(product1.cleaned_title, shingle_size)
    shingles2 = generate_shingles(product2.cleaned_title, shingle_size)
    title_similarity = jaccard_similarity(shingles1, shingles2)

    # Potential model number similarity
    model_similarity = 1 if set(product1.potential_model_numbers) & set(product2.potential_model_numbers) else 0

    # Brand similarity
    brand_similarity = 1 if product1.brand == product2.brand else 0

    # Size similarity
    size_similarity = 1 if product1.size == product2.size else 0

    # Weighted total similarity
    total_similarity = (
        alpha * title_similarity +
        beta * model_similarity +
        gamma * brand_similarity +
        delta * size_similarity
    )

    return total_similarity




#Deprecated: we use other filter_candidate_pairs_with_shingles.
    #Filter candidate pairs based on Jaccard Similarity.
def filter_candidate_pairs_jaccard(binary_matrix, candidate_pairs, threshold):
    filtered_pairs = []

    for pair in candidate_pairs:
        similarity = compute_jaccard_similarity(binary_matrix, pair)
        if similarity >= threshold:
            filtered_pairs.append((pair, similarity))

    return filtered_pairs


    #    Filter candidate pairs using shingle-based MSM similarity.
def filter_candidate_pairs_with_shingles(tv_products, candidate_pairs, shingle_size=3, threshold=0.75):
    filtered_pairs = []

    for pair in candidate_pairs:
        product1, product2 = tv_products[pair[0]], tv_products[pair[1]]
        similarity = compute_msm_similarity_with_shingles(product1, product2, shingle_size)

        if similarity >= threshold:
            filtered_pairs.append((pair, similarity))

    return filtered_pairs

    #    Compute Pair Quality (PQ), Pair Completeness (PC), F1*, and traditional evaluation metrics.
def compute_evaluation_metrics(ground_truth_pairs, predicted_pairs, candidate_pairs):
    # Convert predicted pairs to a set of just the pair indices
    predicted_set = {pair for pair, _ in predicted_pairs}

    # True positives (TP): predicted to be duplicates and are real duplicates.
    true_positives = ground_truth_pairs & predicted_set

    # False positives (FP): Predicted as duplicates but are real non-duplicates
    false_positives = predicted_set - ground_truth_pairs

    # False negatives (FN): Real duplicates but not predicted as duplicates
    false_negatives = ground_truth_pairs - predicted_set

    # True positives count
    duplicates_found = len(true_positives)

    # Metrics calculation (predicted set: = TP + FP) (ground_thuth pairs: = TP+ TN)
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0.0
    recall = len(true_positives) / len(ground_truth_pairs) if ground_truth_pairs else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # Pair Quality (PQ): Duplicates found / Comparisons made
    pair_quality = duplicates_found / len(candidate_pairs) if candidate_pairs else 0.0

    # Pair Completeness (PC): Duplicates found / Total duplicates
    pair_completeness = duplicates_found / len(ground_truth_pairs) if ground_truth_pairs else 0.0

    # F1*: Harmonic mean of PQ and PC
    f1_star = (
        2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness)
        if pair_quality + pair_completeness > 0
        else 0.0
    )

    # Return all metrics
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "pair_quality": pair_quality,
        "pair_completeness": pair_completeness,
        "f1_star": f1_star,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
    }
    #Compute the fraction of comparisons performed out of the total possible combinations.
def compute_fraction_of_comparisons(num_products, candidate_pairs):
    total_pairs = (num_products * (num_products - 1)) // 2  # Total possible pairs
    fraction = len(candidate_pairs) / total_pairs if total_pairs > 0 else 0
    return fraction, total_pairs


    # Displays a performance table with metrics.
def display_performance_table(candidate_pairs, true_pairs, metrics, stage_name, total_pairs, fraction):
    data = [
        ["Metric", "Value"],
        ["Stage", stage_name],
        ["Candidate Pairs", len(candidate_pairs)],
        ["True Pairs", len(true_pairs)],
        ["Total Possible Pairs", total_pairs],
        ["Fraction of Comparisons", f"{fraction:.4f}"],
        ["Pair Quality (PQ)", f"{metrics['pair_quality']:.4f}"],
        ["Pair Completeness (PC)", f"{metrics['pair_completeness']:.4f}"],
        ["F1* (Harmonic Mean)", f"{metrics['f1_star']:.4f}"],
        ["Precision", f"{metrics['precision']:.2f}"],
        ["Recall", f"{metrics['recall']:.2f}"],
        ["F1-Score", f"{metrics['f1_score']:.2f}"],
        ["True Positives", metrics["true_positives"]],
        ["False Positives", metrics["false_positives"]],
        ["False Negatives", metrics["false_negatives"]],
    ]
    table = tabulate(data, headers=["Measure", "Result"], tablefmt="plain")
    print("\nPerformance Metrics:\n")
    print(table)


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(3)
    np.random.seed(3)

    # File name
    file_name = "TVs-all-merged.json"

    #  Load the data into TVProduct instances
    products = load_data(file_name)
    print(f"Loaded {len(products)} TV products.")

    # you can uncheck this to check a few lines manually
    #  Print the first 5 products for verification
    # print("\nFirst 5 TV Products:")
    # for product in products[:5]:
    #     print(product)

    # Generates pairs based on official model IDs
    pairs = get_official_modelID_pairs(products)
    print(f"\nNumber of pairs with the same modelID: {len(pairs)}")
    #print(f"First 10 pairs: {list(pairs)[:10]}")

    ## You can print this to check if the actual pairs are correct.
    # Print titles of the first 5 pairs
    print("\nTitles of the first 5 pairs:")
    for i, (idx1, idx2) in enumerate(list(pairs)[:5]):
        product1, product2 = products[idx1], products[idx2]
        print(f"Pair {i + 1}:")
        print(f"  Product 1 Title: {product1.title}")
        print(f"  Product 2 Title: {product2.title}")

    # Get vocabulary and create the binary matrix
    vocabulary = build_vocabulary(products)
    binary_matrix = create_binary_matrix(products, vocabulary)
    print(f"\nBinary matrix shape: {binary_matrix.shape}")
    print("\ndim vocab: ")
    print(len(vocabulary))
    print(f"Total number of 1s: {np.sum(binary_matrix)}")

    # Generate MinHash signature matrix
    num_hashes = 150
    hash_functions = generate_hash_functions(num_hashes, len(vocabulary))
    signature_matrix = compute_minhash_signature(binary_matrix, hash_functions)
    print(f"\nMinHash signature matrix shape: {signature_matrix.shape}")
    print(f"Sample signature matrix:\n{signature_matrix[:5, :10]}")

    # Perform banding to generate candidate pairs
    ## IMPORTANT: num_hashes should be dividable by num_bands
    num_bands = 30

    # banding with shop AND brand filter works the best, we get hgiher Fstar score from this.
    # Perform banding to generate candidate pairs with shop filter
    # candidate_pairs = banding_with_shop_filter(signature_matrix, num_bands, products)

    # Perform banding to generate candidate pairs with shop and brand filter
    candidate_pairs = banding_with_shop_and_brand_filter(signature_matrix, num_bands, products)

    print(f"\nNumber of candidate pairs: {len(candidate_pairs)}")
    print("Sample candidate pairs:")
    print(list(candidate_pairs)[:10])

    # Filter candidate pairs using shingle-based MSM similarity
    shingle_size = 3  # You can adjust the shingle size
    threshold = 0.70  # Similarity threshold
    filtered_pairs = filter_candidate_pairs_with_shingles(products, candidate_pairs, shingle_size, threshold)

    print(f"\nNumber of filtered pairs (similarity >= {threshold}): {len(filtered_pairs)}")
    print(f"Sample filtered pairs with similarity scores:")
    for pair, similarity in filtered_pairs[:10]:
        print(f"Pair: {pair}, Similarity: {similarity:.2f}")

    # Compute Evaluation Metrics
    ground_truth_pairs = set(pairs)  # True pairs from model IDs
    metrics = compute_evaluation_metrics(ground_truth_pairs, filtered_pairs, candidate_pairs)

    # Compute fraction of comparisons
    fraction, total_pairs = compute_fraction_of_comparisons(len(products), candidate_pairs)
    # Print the fraction and total comparisons
    print(f"\nTotal possible pairs: {total_pairs}")
    print(f"Number of candidate pairs: {len(candidate_pairs)}")
    print(f"Fraction of comparisons: {fraction:.4f}")

    # Usage in main pipeline
    metrics = compute_evaluation_metrics(ground_truth_pairs, filtered_pairs, candidate_pairs)
    display_performance_table(candidate_pairs, ground_truth_pairs, metrics, stage_name="LSH", total_pairs=total_pairs,
                              fraction=fraction)
