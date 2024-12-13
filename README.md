# Scalable-Product-Duplicate-Detection

This project implements a scalable product duplicate detection system specifically for TVs, utilizing Locality-Sensitive Hashing (LSH) and advanced filtering techniques to efficiently identify duplicate listings across multiple vendors while balancing computational cost and detection accuracy.

File Structure:

mainv11.py:
This is the core of the calculations, it can be run independent, but can also be run via the bootstrappingv4 file.

class_tv_product.py:
This file determines the structure of the individial tv products. It converts the TVs-allmerged.json file to indivial 'tv' products. It also contains functions for data cleaning and extraction.

bootstrappingv4.py:
With this file, a grid search for optimal parameters can eb performed. With it we can also perform a few boothstrap runs. Note: we can also create performance evaluation plots with it showing PC, PQ, F1 scores vs fraction of comparisons made.

GeneralInformation.py:
With this file, the data was analysed, and you can create lists of the cleaned titled or descriptions, brand names, sizes and more.

bugfix2.py:
This file was created for small tests and bugfixxing, it does not serve a specific goal.
