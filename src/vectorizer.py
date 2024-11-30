import numpy as np
from NLST_data_dict import feature_transforms
from collections import defaultdict

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5, str_input=False):
        self.feature_transforms = feature_transforms
        self.inv_feature_transforms = {v_i: k for k, v in self.feature_transforms.items() for v_i in v}
        self.feature_config = feature_config if feature_config else list(self.inv_feature_transforms.keys())
        self.skipped_features = []
        self.feature_levels = defaultdict(dict)
        self.features_fit = defaultdict(list)
        self.num_bins = num_bins
        self.mean = {}
        self.std = {}
        self.classes = {}
        self.bin_edges = {}
        self.str_input = str_input

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self. 
            
            X is a list of dict, where each dict is a sample of {feature: values}.
        """
        
        for feature_name in self.feature_config:

            # get features from each data point in X
            features_from_all_subjects = np.array([int(sample[feature_name]) for sample in X])

            # for numeric features, filter for numeric codes
            if feature_name in self.feature_transforms['numerical']: # select only numeric values
                if self.str_input:
                    features_from_all_subjects = list(filter(lambda i: i.isdigit(), features_from_all_subjects))

            # skip features with constant values
            if len(set(features_from_all_subjects)) == 1:
                print(f'Skipping fitting for feature "{feature_name}", values are constant.')
                self.skipped_features.append(feature_name)
                continue

            if feature_name in self.feature_transforms['numerical']:
                # calculate descriptive statistics
                self.mean[feature_name] = np.nanmean(np.array(features_from_all_subjects).astype(float))
                self.std[feature_name] = np.nanstd(np.array(features_from_all_subjects).astype(float))

                # save feature level
                self.feature_levels['numerical'][feature_name] = [feature_name]
                
                self.features_fit['numerical'].append(feature_name)
                
            if feature_name in self.feature_transforms['histogram']:
                # get bin edge values
                bin_edges = np.histogram_bin_edges(np.array(features_from_all_subjects).astype(float), bins=self.num_bins)
                self.bin_edges[feature_name] = bin_edges
                
                # save feature level
                consecutive_edge_pairs = list(zip(bin_edges, bin_edges[1:]))
                self.feature_levels['histogram'][feature_name] = [f'{str(x[0])} to {str(x[1])}' for x in consecutive_edge_pairs]

                self.features_fit['histogram'].append(feature_name)

            if feature_name in self.feature_transforms['categorical']:
                # get class values
                class_levels = np.unique(features_from_all_subjects)
                self.classes[feature_name] = class_levels

                # save feature level
                self.feature_levels['categorical'][feature_name] = [f'{feature_name}_{level}' for level in class_levels]

                self.features_fit['categorical'].append(feature_name)
        
        # remove skipped features from config
        self.feature_config = [x for x in self.feature_config if x not in self.skipped_features]

    def transform(self, X, feature_type):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: dict, where keys are feature names are values are an array of data points to be transformed
        """

        if not self.features_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )
        
        transformed_data = {}

        for feature, values in X.items():
            if feature in self.features_fit[feature_type]: # for each valid feature
                
                # vectorize the feature
                transformed_data[feature] = np.array(list(map(self.get_vectorizer(feature, feature_type), values)))

        return transformed_data
    
    def get_vectorizer(self, feature_name, feature_type):  
        match feature_type:
            case "numerical":
                return self.get_numerical_vectorizer(feature_name)
            case "histogram":
                return self.get_histogram_vectorizer(feature_name)
            case "categorical":
                return self.get_categorical_vectorizer(feature_name)
    

    def get_numerical_vectorizer(self, feature_name, verbose=False):
        """        
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """

        def numerical_vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            if self.str_input:
                if x.isdigit():
                    transformed_score = (float(x) - self.mean[feature_name]) / self.std[feature_name]
                else:
                    transformed_score = 0 # impute
            else:
                transformed_score = (x - self.mean[feature_name]) / self.std[feature_name]

            return transformed_score
        
        
        return numerical_vectorizer

    def get_histogram_vectorizer(self, feature_name):    

        def histogram_vectorizer(x):
            if self.str_input:
                x = float(x)
            hist, bin_edges = np.histogram(x, bins=self.bin_edges[feature_name])
            transformed_score = hist.argmax()

            return transformed_score

        return histogram_vectorizer

    def get_categorical_vectorizer(self, feature_name):
        """
        :return: function to map categorical x to one-hot feature vector
        """
    
        def categorical_vectorizer(x):
            if self.str_input:
                x = str(x)
            transformed_score = np.where(self.classes[feature_name] == x, 1, 0) 

            return transformed_score

        return categorical_vectorizer