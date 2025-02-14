import torch
import math
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

    return outliers.shape[0]

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


def prune_feature_vector(tensor_data, rate):
    # Get the mean of the tensor
    mean_val = tensor_data.mean()

    # Flatten the tensor while keeping the first dimension
    flat_tensor = tensor_data.view(1, -1)

    # Compute absolute distances from mean
    distances = torch.abs(flat_tensor - mean_val)

    # Get indices of the top n farthest values
    _, indices = torch.topk(distances, round(1024 * 4096 * rate), dim=1)

    # Create a mask and set selected elements to zero
    flat_tensor[0, indices[0]] = 0

    print('after pruning... ', flat_tensor.view_as(tensor_data))
    # Reshape back to original shape
    return flat_tensor.view_as(tensor_data)

def dense_to_CSR(tensor_data):
    #Convert to sparse tensor
    sparse_tensor = tensor_data.to_sparse()

    #Convert sparse tensor to CSR format
    csr_tensor = sparse_tensor.to_sparse_csr()

    print('CSR size 1: ', csr_tensor.crow_indices().shape)
    print('CSR size 2: ', csr_tensor.col_indices().shape)
    print('CSR size 3: ',  csr_tensor.values().shape)

    print('CSR size 11: ', csr_tensor.crow_indices())
    print('CSR size 22: ', csr_tensor.col_indices())
    print('CSR size 33: ', csr_tensor.values())




