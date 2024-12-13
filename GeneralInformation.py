import json
from collections import Counter, defaultdict
from class_tv_product import TVProduct  # Import the TVProduct class

# Load the dataset
with open("TVs-all-merged.json", "r") as f:
    data = json.load(f)

# Initialize variables
tv_products = []
model_ids = []
titles = []
all_features = []
all_brands = set()  # Use a set to store unique brands

# Iterate through the dataset
for product_id, product_info_list in data.items():
    for product_info in product_info_list:
        # Create a TVProduct instance
        tv_product = TVProduct(
            shop=product_info["shop"],
            url=product_info["url"],
            modelID=product_info["modelID"],
            features_map=product_info["featuresMap"],
            title=product_info["title"]
        )
        # Add to lists
        tv_products.append(tv_product)
        model_ids.append(tv_product.modelID)
        titles.append(tv_product.title)
        all_features.extend(tv_product.features_map.keys())  # Collect feature names

        # Add the brand to the set
        all_brands.add(tv_product.brand)  # Set ensures no duplicates

# Analyze features and duplicates
website_product_count = Counter(product.shop for product in tv_products)
model_id_counter = Counter(model_ids)
duplicates = [model for model, count in model_id_counter.items() if count > 1]

# Division of model mentions
category_counts = {1: 0, 2: 0, 3: 0, 4: 0, 'more_than_4': 0}
for model_id, count in model_id_counter.items():
    if count == 1:
        category_counts[1] += 1
    elif count == 2:
        category_counts[2] += 1
    elif count == 3:
        category_counts[3] += 1
    elif count == 4:
        category_counts[4] += 1
    else:
        category_counts['more_than_4'] += 1

# Generate official model ID pairs
def get_official_modelID_pairs(tv_products):
    """
    Generates pairs of products with the same official model ID.
    """
    model_pairs = set()
    model_to_indices = defaultdict(list)

    for i, product in enumerate(tv_products):
        model_to_indices[product.modelID].append(i)

    for indices in model_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    model_pairs.add((indices[i], indices[j]))

    return model_pairs

official_pairs = get_official_modelID_pairs(tv_products)

# Group products by extracted model numbers
def group_by_extracted_models(tv_products):
    """
    Groups products based on their extracted model numbers.
    """
    extracted_model_groups = defaultdict(set)

    for i, product in enumerate(tv_products):
        for model in product.potential_model_numbers:
            extracted_model_groups[model].add(i)

    # Convert sets of indices into pairs for comparison
    extracted_pairs = set()
    for indices in extracted_model_groups.values():
        indices = list(indices)
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    extracted_pairs.add((indices[i], indices[j]))

    return extracted_pairs

extracted_pairs = group_by_extracted_models(tv_products)

# Analyze overlap between official and extracted pairs
correctly_identified_pairs = official_pairs & extracted_pairs
total_official_pairs = len(official_pairs)
total_extracted_pairs = len(extracted_pairs)
total_correct_pairs = len(correctly_identified_pairs)

# Print analysis results
print("\nComparison of Model ID Sets:")
print(f"Total official model ID pairs: {total_official_pairs}")
print(f"Total extracted model ID pairs: {total_extracted_pairs}")
print(f"Total correctly identified pairs: {total_correct_pairs}")

# Calculate percentages
recall = (total_correct_pairs / total_official_pairs) * 100 if total_official_pairs > 0 else 0
precision = (total_correct_pairs / total_extracted_pairs) * 100 if total_extracted_pairs > 0 else 0

print(f"Recall: {recall:.2f}%")
print(f"Precision: {precision:.2f}%")

# Results
print("\nNumber of products from each website:")
for website, count in website_product_count.items():
    print(f"  {website}: {count}")

print("\nAll model IDs:")
print(model_ids)

print("\nAll titles:")
print(titles)

print("\nDistinct features:")
print(set(all_features))  # Set to avoid duplicates

print("\nNumber of duplicate model IDs:")
print(len(duplicates))

print("\nDuplicate model IDs:")
print(duplicates[:10])  # Print first 10 duplicates to avoid clutter

print("\nDivision of model mentions:")
for category, count in category_counts.items():
    print(f"  Models mentioned {category} time(s): {count}")

# Print all unique brands
print("\nAll unique brands:")
for brand in sorted(all_brands):  # Sorted for easier readability
    print(f"  {brand}")
