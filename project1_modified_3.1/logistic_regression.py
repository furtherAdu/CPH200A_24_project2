import numpy as np
import tqdm
import matplotlib.pyplot as plt

class LogisticRegression():
    """
    A logistic regression model trained with stochastic gradient descent.
    Supports class weighting for imbalanced datasets.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, 
                 regularization_lambda=0, class_weights=None, verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.class_weights = class_weights
        self.train_losses = []
        self.val_losses = []

    def compute_sample_weights(self, Y):
        """
        Compute sample weights based on class weights.
        If class_weights is None, returns array of ones.
        """
        if self.class_weights is None:
            return np.ones(len(Y))
        
        sample_weights = np.zeros(len(Y))
        sample_weights[Y == 0] = self.class_weights.get(0, 1.0)
        sample_weights[Y == 1] = self.class_weights.get(1, 1.0)
        return sample_weights

    def fit(self, X, Y, X_val=None, Y_val=None, patience=5):
        """
        Train the logistic regression model using stochastic gradient descent.
        Incorporates sample weights based on class weights.
        """
        n_samples, n_features = X.shape
        self.theta = np.random.rand(n_features)
        self.bias = 0
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_theta = self.theta.copy()
        best_bias = self.bias
        
        # Initialize loss history lists if they don't exist
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
        if not hasattr(self, 'val_losses'):
            self.val_losses = []
        
        # Compute sample weights once before training
        sample_weights = self.compute_sample_weights(Y)
        
        initial_lr = self.learning_rate
        def get_lr(epoch, initial_lr, decay_rate=0.1):
            return initial_lr / (1 + decay_rate * epoch)
            
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            # Step 1: shuffle data 
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            weights_shuffled = sample_weights[permutation]

            # Step 2: mini-batch training 
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
                weights_batch = weights_shuffled[i:i+self.batch_size]

                # Compute the gradient with weights
                d_theta, d_bias = self.gradient(X_batch, Y_batch, weights_batch)

                # Update parameters
                current_lr = get_lr(epoch, initial_lr)
                self.theta -= current_lr * d_theta
                self.bias -= current_lr * d_bias
            
            # Compute and store training loss
            train_loss = self.compute_loss(X, Y)
            self.train_losses.append(train_loss)
            
            # Validation and early stopping logic
            if X_val is not None and Y_val is not None:
                val_loss = self.compute_loss(X_val, Y_val)
                self.val_losses.append(val_loss)
                
                # Check if we have a new best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_theta = self.theta.copy()
                    best_bias = self.bias
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if patience_counter >= patience:
                    if self.verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    self.theta = best_theta
                    self.bias = best_bias
                    break

            if self.verbose:
                print(f"\nEpoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}", end="")
                if X_val is not None:
                    print(f", Val Loss: {val_loss:.4f}", end="")
                print()

        return self

    def predict_proba(self, X):
        """
        Predict the probability of lung cancer for each sample in X.
        Uses numerically stable sigmoid function.
        """
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        p = []    
        for sample in X:
            z = np.dot(sample, self.theta) + self.bias
            clipped_sig = np.clip(sigmoid(z), 1e-15, 1 - 1e-15)
            p.append(clipped_sig)

        return np.array(p)

    def gradient(self, X, Y, sample_weights):
        """
        Compute the weighted gradient of the loss with respect to theta and bias.
        Includes L2 Regularization.
        """
        n_samples = X.shape[0]
        p = self.predict_proba(X)
        
        # Apply sample weights to the gradient computation
        weighted_diff = sample_weights * (p - Y)
        
        # Compute weighted gradients
        d_theta = (1/n_samples) * (np.dot(X.T, weighted_diff) + self.regularization_lambda * self.theta)
        d_bias = (1/n_samples) * np.sum(weighted_diff)
        
        return d_theta, d_bias

    def predict(self, X, threshold=0.5):
        """
        Predict if patient will develop lung cancer for each sample in X.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def compute_loss(self, X, Y):
        """
        Compute the weighted binary cross-entropy loss with L2 regularization.
        """
        n_samples = X.shape[0]
        p = self.predict_proba(X)
        
        # Apply sample weights to the loss computation
        sample_weights = self.compute_sample_weights(Y)
        
        # Compute weighted binary cross-entropy loss
        weighted_loss = -np.mean(
            sample_weights * (Y * np.log(p) + (1 - Y) * np.log(1 - p))
        )
        
        # Add L2 regularization term
        weighted_loss += (self.regularization_lambda / (2 * n_samples)) * np.sum(self.theta ** 2)
        
        return weighted_loss