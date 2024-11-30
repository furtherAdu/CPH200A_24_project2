import numpy as np
import json

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.
    """
    def __init__(self, feature_config, num_bins=5, feature_map_path=None):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.subgroups = {}
        self.is_fit = False
        self.feature_map = None

        # Load feature map if provided
        if feature_map_path:
            with open(feature_map_path, 'r') as f:
                self.feature_map = json.load(f)

    def get_numerical_vectorizer(self, values, verbose=False):
        values = np.array([float(v) if v not in [None, ''] else 0.0 for v in values])
        mean, std = np.nanmean(values), np.nanstd(values)

        def vectorizer(x):
            if x in [None, '']:
                return [0.0]
            else:
                x = float(x)
                return [(x - mean) / std]
        return vectorizer

    def get_histogram_vectorizer(self, values):
        values_filtered = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
        hist, bin_edges = np.histogram(values_filtered, bins='auto')

        if len(values_filtered) == 0:
            raise ValueError("No valid values available for histogram creation.")
            
        def vectorizer(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return [-1]
            bin_index = np.digitize(x, bin_edges) - 1
            if bin_index < 0 or bin_index >= len(bin_edges) - 1:
                return [-1]
            return [bin_index]
        return vectorizer

    def get_categorical_vectorizer(self, values):
        unique_values = list(set(values))
        index_map = {value: index for index, value in enumerate(unique_values)}
        missing_value_category = "missing"
        if missing_value_category not in index_map:
            index_map[missing_value_category] = len(unique_values)
            unique_values.append(missing_value_category)

        def vectorizer(x):
            if x not in unique_values or x is None:
                x = missing_value_category
            one_hot_vector = np.zeros(len(unique_values))
            one_hot_vector[index_map[x]] = 1
            return one_hot_vector
        return vectorizer

    def fit(self, X):
        for feature in self.feature_config:
            name = feature['name']
            type = feature['type']
            values = [x.get(name) for x in X]
            if type == 'numerical':
                self.feature_transforms[name] = self.get_numerical_vectorizer(values)
            elif type == 'categorical':
                self.feature_transforms[name] = self.get_categorical_vectorizer(values)
            elif type == 'histogram':
                self.feature_transforms[name] = self.get_histogram_vectorizer(values)
            else:
                raise ValueError(f"Unknown feature type: {type}")
        self.is_fit = True

    def transform(self, X, nlst=False):

        if not self.is_fit:
            raise Exception("Vectorizer not initialized! You must first call fit with a training set")
        transformed_data = []
        # If NLST, rename columns using the feature map
        if nlst and self.feature_map:
            X_mapped = []
            for row in X:
                mapped_row = {plco_feature: row[nlst_feature]
                            for plco_feature, nlst_feature in self.feature_map.items()
                            if nlst_feature in row}
                X_mapped.append(mapped_row)
            X = X_mapped

        for x in X:
            feature_vector = []
            for feature in self.feature_config:
                original_name = feature['name']
                mapped_name = self.feature_map.get(original_name, original_name) if self.feature_map else original_name
                type = feature['type']

                if mapped_name in x:
                    value = x[mapped_name]
                    transform_function = self.feature_transforms[original_name]
                    feature_vector.extend(transform_function(value))
                else:
                    # Handle missing values
                    feature_vector.extend(self.feature_transforms[original_name](None))

            transformed_data.append(feature_vector)
        return np.array(transformed_data)
