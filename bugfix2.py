from class_tv_product import TVProduct
from mainv6 import load_data  # Assuming you already have a load_data function to load products from JSON
from collections import Counter
import re
# Load the data
file_name = "TVs-all-merged.json"  # Adjust the file name if needed
products = load_data(file_name)




# check 1: def: extract_model_numbers
# check 1: def: extract_model_numbers.

# Initialize counters
total_products = len(products)
correctly_identified = 0
incorrectly_identified = 0
not_found = 0
total_possible_extractions = 0  # For recall calculation
incorrect_cases = []  # To store incorrectly identified cases

# Print header
print(f"{'Actual Model Number':<30} {'Extracted Possible Model Numbers':<50} {'Correct?':<10}")
print("=" * 90)

# Iterate over the TV products and compare model numbers
for product in products:
    actual_model = product.modelID
    extracted_models = product.potential_model_numbers
    total_possible_extractions += 1  # Count for recall (every product has a model ID to extract)

    # Check if any model ID was extracted
    if not extracted_models:
        not_found += 1
        print(f"{actual_model:<30} {'None':<50} {'False':<10}")
        print(f"FAILED TO IDENTIFY: {product.title}")
        continue

    # Check if the actual model ID is in the extracted potential models
    is_correct = actual_model in extracted_models

    # Update counters
    if is_correct:
        correctly_identified += 1
    else:
        incorrectly_identified += 1
        incorrect_cases.append((actual_model, extracted_models, product.title))

    # Print the results in the desired format
    print(f"{actual_model:<30} {', '.join(extracted_models):<50} {str(is_correct):<10}")

    # If false, print the product title for debugging
    if not is_correct:
        print(f"FAILED TO IDENTIFY: {product.title}")

# Calculate precision, recall, and accuracy
if (correctly_identified + incorrectly_identified) > 0:
    precision = (correctly_identified / (correctly_identified + incorrectly_identified)) * 100
else:
    precision = 0.0

if total_possible_extractions > 0:
    recall = (correctly_identified / total_possible_extractions) * 100
else:
    recall = 0.0

accuracy = (correctly_identified / total_products) * 100

# Print summary
print("\nSummary:")
print(f"Total Products: {total_products}")
print(f"Correctly Identified: {correctly_identified}")
print(f"Incorrectly Identified: {incorrectly_identified}")
print(f"Not Found: {not_found}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")

# Print incorrectly identified cases
if incorrect_cases:
    print("\nIncorrectly Identified Cases:")
    print(f"{'Actual Model Number':<30} {'Extracted Models':<50} {'Product Title':<50}")
    print("=" * 130)
    for actual, extracted, title in incorrect_cases:
        print(f"{actual:<30} {', '.join(extracted):<50} {title:<50}")
#
#
#
# ## check 2: def: cleans_title
# # # Print header
# # print(f"{'Cleaned Title':<50}")
# # print("=" * 50)
# #
# # # Iterate over the TV products and print the cleaned titles
# # for product in products:
# #     print(f"{product.cleaned_title:<50}")
#
#
# ##check 3: def get_brand_from_title
# # Print header
#
# print(f"{'Extracted?':<10} {'Brand':<20} {'Cleaned Title':<50}")
# print("=" * 80)
#
# # Initialize counters, list for titles with missing brands, and a counter for brand mentions
# titles_no_brand = []
# brand_counter = Counter()
# total_products = len(products)
# extracted_count = 0
# missing_count = 0
#
# # Iterate over the TV products
# for product in products:
#     cleaned_title = product.cleaned_title
#     brand = product.get_brand_from_title(cleaned_title)  # Pass the cleaned title to the method
#
#     # Determine if the brand was extracted
#     is_extracted = bool(brand)
#     if is_extracted:
#         extracted_count += 1
#         brand_counter[brand] += 1
#     else:
#         missing_count += 1
#         titles_no_brand.append(cleaned_title)
#
#     # Print the results
#     print(f"{str(is_extracted):<10} {brand if brand else 'N/A':<20} {cleaned_title:<50}")
#
# # Print summary
# print("\nSummary:")
# print(f"Total products processed: {total_products}")
# print(f"Successfully extracted brands: {extracted_count}")
# print(f"Missing brands: {missing_count}")
#
# # Print titles where brand was not identified
# if titles_no_brand:
#     print("\nTitles where brand couldn't be identified:")
#     print("=" * 50)
#     for title in titles_no_brand:
#         print(title)
#
# # Print all unique brands with their counts, sorted by count
# if brand_counter:
#     print("\nAll Unique Brands with Counts:")
#     print("=" * 50)
#     for brand, count in brand_counter.most_common():  # Sort by count, descending
#         print(f"{brand:<20} {count}")

# Step 4: check the get_size

# Initial size extraction
sizes_found = []
titles_without_sizes = []

# Extract sizes
for product in products:
    size = product.get_size()  # Use the updated get_size method
    if size:
        sizes_found.append((size, product.cleaned_title))
    else:
        titles_without_sizes.append((product.title, product.cleaned_title))

# Print results
print(f"{'Size (in inches)':<20} {'Cleaned Title':<80}")
print("=" * 100)
for size, title in sizes_found:
    print(f"{size:<20} {title:<80}")

# Print titles where no size was identified
print("\nTitles Where Size Could Not Be Identified:")
print("=" * 100)
for original_title, cleaned_title in titles_without_sizes:
    print(f"Original Title: {original_title}")
    print(f"Cleaned Title:  {cleaned_title}")
    print("-" * 100)

# Print a summary
print(f"\nTotal sizes identified: {len(sizes_found)}")
print(f"Total titles without sizes: {len(titles_without_sizes)}")


