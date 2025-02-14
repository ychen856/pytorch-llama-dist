import torch

def get_outlier(flat_tensor):
    # Flatten the tensor
    #flat_tensor = tensor_data.flatten()

    # Compute mean and standard deviation
    mean = flat_tensor.mean()
    std = flat_tensor.std()

    # Compute Z-score
    z_scores = (flat_tensor - mean) / std

    # Define a threshold (e.g., |Z| > 3 is often considered an outlier)
    threshold = 3
    outliers = flat_tensor[torch.abs(z_scores) > threshold]

    return outliers

#input shape [1, 1024, 4096]
def get_pruning_rate(tensor_data):
    outlier_number_list = []
    pruning_rate_list = []
    for flatten_tensor in tensor_data[0]:
        outliers = get_outlier(flatten_tensor).shape[0]
        outlier_number_list.append(outliers)
        pruning_rate_list.append(1/outliers)

    print('outliers: ', outlier_number_list)
    print('rate: ', pruning_rate_list)


