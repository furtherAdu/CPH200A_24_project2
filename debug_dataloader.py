# analyze_weighted_sampler.py

import os
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
import sys
import time

# Ensure custom modules can be imported, adjust path according to actual situation
sys.path.append('/scratch/users/xinyang_han/CPH200_24/CPH200A_24_project2/src')

from dataset import NLST

def get_class_distribution(data):
    """Count the number of samples for each class in the dataset."""
    labels = [int(sample['y']) for sample in data]
    counter = Counter(labels)
    return counter

def print_class_ratios(original, sampled, title_suffix):
    """Print the class distribution ratios before and after sampling."""
    original_total = sum(original.values())
    sampled_total = sum(sampled.values())

    print(f"\nClass Distribution Ratios ({title_suffix}):")
    print(f"Original dataset - Negative (0): {original.get(0,0)} ({original.get(0,0)/original_total:.2%}), "
          f"Positive (1): {original.get(1,0)} ({original.get(1,0)/original_total:.2%})")
    print(f"Sampled dataset  - Negative (0): {sampled.get(0,0)} ({sampled.get(0,0)/sampled_total:.2%}), "
          f"Positive (1): {sampled.get(1,0)} ({sampled.get(1,0)/sampled_total:.2%})")

def main():
    start_time = time.time()
    
    # Global parameters, adjust according to actual situation
    common_params = {
        'num_channels': 3,
        'use_data_augmentation': False,  # Disable data augmentation to simplify analysis
        'batch_size': 64,
        'num_workers': 4,
        'nlst_metadata_path': "/scratch/project2/nlst-metadata/full_nlst_google.json",
        'valid_exam_path': "/scratch/project2/nlst-metadata/valid_exams.p",
        'nlst_dir': "/scratch/project2/compressed",
        'lungrads_path': "/scratch/project2/nlst-metadata/nlst_acc2lungrads.p",
        'group_keys': ['race', 'educat', 'gender', 'age', 'ethnic'],
        'clinical_features': [],  # Add actual clinical features if needed
        'feature_config': []  # Add actual feature configuration if needed
    }

    # Initialize data module without class balance sampler
    datamodule_no_balance = NLST(
        **common_params,
        class_balance=False
    )
    print("Initialized NLST DataModule without class balancing.")

    # Prepare and setup data
    datamodule_no_balance.prepare_data()
    datamodule_no_balance.setup(stage='fit')

    # Get original class distribution (without class balance sampler)
    original_counter_no = get_class_distribution(datamodule_no_balance.train)
    print(f"Original class distribution (No Balancing): {original_counter_no}")
    print_class_ratios(original_counter_no, original_counter_no, "No Class Balancing")

    # Initialize data module with class balance sampler
    datamodule_balance = NLST(
        **common_params,
        class_balance=True
    )
    print("\nInitialized NLST DataModule with class balancing.")

    # Prepare and setup data
    datamodule_balance.prepare_data()
    datamodule_balance.setup(stage='fit')

    # # Get original class distribution (with class balance sampler)
    # original_counter_balance = get_class_distribution(datamodule_balance.train)
    # print(f"Original class distribution (With Balancing): {original_counter_balance}")
    # print_class_ratios(original_counter_balance, original_counter_balance, "With Class Balancing")

    # Setup WeightedRandomSampler
    if datamodule_balance.class_balance:
        weights = datamodule_balance.get_samples_weight(datamodule_balance.train)
        print(f"\nCreated WeightedRandomSampler with {len(weights)} samples.")
        unique_weights = torch.unique(weights)
        print(f"Unique weights: {unique_weights}")

        # Increase num_samples to enhance class balance effect, e.g., set to 2x training set size
        num_samples = len(weights)
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        print(f"Sampler set with num_samples={num_samples} and replacement=True.")
    else:
        sampler = None
        print("Class balancing is disabled. No sampler created.")

    # Create DataLoader, ensure shuffle is disabled
    loader_balance = DataLoader(
        datamodule_balance.train,
        batch_size=datamodule_balance.batch_size,
        num_workers=datamodule_balance.num_workers,
        sampler=sampler,
    )

    # Iterate through DataLoader and count class distribution after sampling
    sampled_labels = []
    max_batches = 100  # Limit iteration count to speed up testing
    print(f"\nStarting to iterate through the DataLoader for {max_batches} batches...")
    import pdb; pdb.set_trace()
    for batch_idx, batch in enumerate(loader_balance):
        labels = batch['y'].numpy()
        sampled_labels.extend(labels)

        if batch_idx + 1 >= max_batches:
            print(f"Reached maximum of {max_batches} batches. Stopping iteration.")
            break

    sampled_counter_balance = Counter(sampled_labels)
    print(f"\nSampled class distribution (With Balancing): {sampled_counter_balance}")
    # print_class_ratios(original_counter_balance, sampled_counter_balance, "With Class Balancing")

    # Validate Sampler weight distribution
    if isinstance(sampler, WeightedRandomSampler):
        print("\nWeightedRandomSampler Details:")
        print(f"Total weights: {len(sampler.weights)}")
        print(f"First 10 weights: {sampler.weights[:10]}")
    else:
        print("\nNo WeightedRandomSampler applied.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal script execution time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()