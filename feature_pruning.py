import torch

def get_outlier(tensor_data):
    # Flatten the tensor
    flat_tensor = tensor_data.flatten()

    # Compute mean and standard deviation
    mean = flat_tensor.mean()
    std = flat_tensor.std()

    # Compute Z-score
    z_scores = (flat_tensor - mean) / std

    # Define a threshold (e.g., |Z| > 3 is often considered an outlier)
    threshold = 3
    outliers = flat_tensor[torch.abs(z_scores) > threshold]

    print("Outliers:", outliers)
    return outliers