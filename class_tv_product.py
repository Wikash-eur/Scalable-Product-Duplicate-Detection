import re

class TVProduct:
    # Class-level variable for tracking instance count
    instance_counter = 0

        #Initialize a TVProduct instance with attributes from the JSON data.
    def __init__(self, shop, url, modelID, features_map, title):
        self.shop = shop
        self.url = url
        self.modelID = modelID.lower()
        self.features_map = self.cleans_features(features_map)
        self.title = title

        # Assign a unique index to this instance
        self.index = TVProduct.instance_counter
        TVProduct.instance_counter += 1

        # Create a cleaned version of the title
        self.cleaned_title = self.cleans_title(title)

        # Extract the brand from the title
        self.brand = self.get_brand_from_title(self.cleaned_title)

        # Extract potential model numbers from the title
        self.potential_model_numbers = self.extract_model_numbers(self.title)

        # Extract the screen size from the cleaned title
        self.size = self.get_size()

    @staticmethod
    def extract_model_numbers(title):

        """
        Extracts potential model numbers from a given title.

        - Must contain at least one digit and one letter.
        - Must be at least 4 characters long.
        - Allows special characters '-' and '/'.
        - Ensures complete model numbers are extracted, including both parts around '-' or '/'.
        - Excludes irrelevant terms like '120hz', 'inch', etc.
        """
        # Convert to lowercase
        title = title.lower()

        # Remove irrelevant terms
        irrelevant_words = [
            "inch", "hz", "1080p", "720p", "2160p", "3dready", "4k", "hdtv", "smart",
            "series", "class", "diag", "diagonal", "120hz", "240hz", "60hz", "600hz",
            "3d-ready", "480hz"
        ]
        for word in irrelevant_words:
            # Use word boundaries to ensure whole word removal
            title = re.sub(rf"\b{word}\b", "", title)

        # Retain alphanumeric characters and special symbols '-' and '/'
        # Avoid splitting valid model numbers at '-'
        title = re.sub(r"[^\w\-\/]", " ", title)

        # Extract model numbers that are at least 4 characters long, have digits, and letters
        model_candidates = re.findall(r"\b[\w\-\/]{4,}\b", title)
        valid_models = [
            candidate for candidate in model_candidates
            if any(char.isdigit() for char in candidate) and any(char.isalpha() for char in candidate)
        ]

        # Return the list of valid model numbers
        return valid_models

    def cleans_title(self, title):
        """
        Cleans the title by:
        - Converting to lowercase.
        - Normalizing variations of 'inch' and 'hz'.
        - Removing unwanted characters ('-' and '/').
        """
        # Convert to lowercase
        title = title.lower()

        # Replace variations of 'inch' including quotes
        title = re.sub(r'(\b\d+)"', r'\1 inch', title)  # Replace number followed by a double quote with 'inch'
        title = re.sub(r'(\b\d+)\'\'', r'\1 inch', title)  # Replace number followed by two single quotes with 'inch'
        title = re.sub(r'\b(inch(es)?)\b', 'inch', title)  # Normalize the word 'inch'

        # Normalize variations of 'hz'
        title = re.sub(r'\b(hertz|hz)\b', 'hz', title, flags=re.IGNORECASE)

        # Remove unwanted characters like '-' and '/'
        title = re.sub(r'[-/]', '', title)

        return title

    def get_brand_from_title(self, title):
        # note should be used with cleaneded title!
        """
        Extracts the brand from the title. The brand is:
        - The first word in the title after removing special characters, store names, and some other words.
        - Consists only of letters (no numbers or special characters).
        """
        #the following names, could be seen as brand and are therefor deleted.
        store_names = ["newegg.com", "best buy", "amazon.com", "thenerds.net", "refurbished", "inch", "open", "plasma", "class", "led"]
        for store in store_names:
            title = title.replace(store, "")

        title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        words = title.split()

        for word in words:
            if word.isalpha():
                return word
        return "Unknown"

    def cleans_features(self, features_map):
        """
        Cleans the features map by:
        - Lowercasing all keys.
        - Lowercasing all values.
        - Normalizing variations of 'inch' and 'hz' in the values.
        """
        cleaned_features = {}
        for key, value in features_map.items():
            cleaned_key = key.lower()
            cleaned_value = str(value).lower()
            cleaned_value = re.sub(r'\b(inch(es)?|["\'`]|-inch)\b', 'inch', cleaned_value)
            cleaned_value = re.sub(r'\b(hertz|hz)\b', 'hz', cleaned_value, flags=re.IGNORECASE)
            cleaned_features[cleaned_key] = cleaned_value
        return cleaned_features

    def get_size(self):
        """
        Extracts the screen size (in inches) from the cleaned title.
        Uses fallback logic to infer size from the model ID if not found in the title.
        """
        # Step 1: Try extracting size directly from the cleaned title
        size_pattern = r'\b((?:[1-9]\d?|[1-9]\.\d{1,2}))\s*(?:inch|in\.|diag\.|class|lcd|led|tv)?\b'
        matches = re.findall(size_pattern, self.cleaned_title)
        sizes = [float(match) for match in matches]

        if sizes:
            rounded_matches = [size for size in sizes if size.is_integer()]
            decimal_matches = [size for size in sizes if not size.is_integer()]
            if rounded_matches:
                return int(rounded_matches[0])  # Return first rounded size
            elif decimal_matches:
                return int(round(decimal_matches[0]))  # Round half up

        # Step 2: Fallback to potential model number
        if self.potential_model_numbers:
            first_model = self.potential_model_numbers[0]
            model_size_match = re.search(r'(\d{2})', first_model)  # First 2 digits
            if model_size_match:
                inferred_size = int(model_size_match.group(1))
                return inferred_size  # Directly use the first 2 digits as size

        # Step 3: Fallback to directly extracting digits from the cleaned title
        fallback_model_pattern = r'(\d{2})'  # First occurrence of any 2 digits
        fallback_match = re.search(fallback_model_pattern, self.cleaned_title)
        if fallback_match:
            inferred_size = int(fallback_match.group(1))
            return inferred_size  # Directly return the first 2 digits

        # Step 4: Return None if no size is found
        return None

        #Returns a string representation of a single TVProduct object.
    def __repr__(self):
        return (
            f"TVProduct(index={self.index}, title='{self.cleaned_title}', "
            f"modelID='{self.modelID}', shop='{self.shop}', brand='{self.brand}', "
            f"size='{self.size}')"
        )
