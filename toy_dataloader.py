# analyze_weighted_sampler_toy_extended.py

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

class ToyDataset(Dataset):
    def __init__(self, total_samples=10000, imbalance_ratio=0.9):
        """
        Initialize ToyDataset.
        
        Args:
            total_samples (int): Total number of samples in the dataset.
            imbalance_ratio (float): Proportion of class 0 (0 < imbalance_ratio < 1).
        """
        self.data = []
        num_class0 = int(total_samples * imbalance_ratio)
        num_class1 = total_samples - num_class0
        for i in range(num_class0):
            sample = {
                'x': torch.randn(3, 256, 256),  # Example image data
                'y': 0  # Class 0
            }
            self.data.append(sample)
        for i in range(num_class1):
            sample = {
                'x': torch.randn(3, 256, 256),  # Example image data
                'y': 1  # Class 1
            }
            self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_class_distribution(data):
    """Count the number of samples for each class in the dataset."""
    labels = [int(sample['y']) for sample in data]
    counter = Counter(labels)
    return counter

def print_class_ratios(original, sampled, title_suffix):
    """Print class distribution ratios for both original and sampled data."""
    original_total = sum(original.values())
    sampled_total = sum(sampled.values())

    print(f"\nClass Distribution Ratios ({title_suffix}):")
    print(f"Original dataset - Negative (0): {original.get(0,0)} ({original.get(0,0)/original_total:.2%}), "
          f"Positive (1): {original.get(1,0)} ({original.get(1,0)/original_total:.2%})")
    print(f"Sampled dataset  - Negative (0): {sampled.get(0,0)} ({sampled.get(0,0)/sampled_total:.2%}), "
          f"Positive (1): {sampled.get(1,0)} ({sampled.get(1,0)/sampled_total:.2%})")
    print(f"Total samples in epoch: {sampled_total}")

def main():
    # Initialize ToyDataset
    total_samples = 10000
    imbalance_ratio = 0.9  # 90% class 0, 10% class 1
    dataset = ToyDataset(total_samples=total_samples, imbalance_ratio=imbalance_ratio)
    print("Initialized ToyDataset.")
    
    # Calculate original class distribution
    original_counter = get_class_distribution(dataset)
    print(f"Original class distribution: {original_counter}")
    print_class_ratios(original_counter, original_counter, "Original Data")
    
    # Calculate sample weights
    targets = [int(sample['y']) for sample in dataset]
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    print(f"\nClass weights: {class_weights}")
    
    # Assign weights to each sample
    samples_weight = torch.tensor([class_weights[y] for y in targets], dtype=torch.double)
    
    # Define different num_samples values for testing
    num_samples_list = [100, 300, 1000, 5000, 10000]  # You can adjust these values as needed
    
    for num_samples in num_samples_list:
        print(f"\n{'-'*50}")
        print(f"Testing with num_samples = {num_samples}")
        
        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=num_samples, replacement=True)
        print(f"Created WeightedRandomSampler with {num_samples} samples.")
        print(f"Unique weights: {torch.unique(samples_weight)}")
        
        # Create DataLoader with shuffle disabled
        loader = DataLoader(
            dataset,
            batch_size=100,  # Set larger batch_size to reduce iterations
            sampler=sampler
        )
        print("Created DataLoader with WeightedRandomSampler.")
        
        # Iterate through DataLoader and count sampled class distribution
        sampled_labels = []
        for batch_idx, batch in enumerate(loader):
            labels = batch['y'].numpy()
            sampled_labels.extend(labels)
        
        sampled_counter = Counter(sampled_labels)
        print(f"Sampled class distribution: {sampled_counter}")
        print_class_ratios(original_counter, sampled_counter, f"With Class Balancing (num_samples={num_samples})")

if __name__ == "__main__":
    main()