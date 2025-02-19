import torch
import math
def get_outlier(flat_tensor):
    # Flatten the tensor
    #flat_tensor = tensor_data.flatten()

    # Compute mean and standard deviation
    mean = flat_tensor.double().mean()
    std = flat_tensor.double().std()

    #print('mean: ', mean)
    #print('std: ', std)

    # Compute Z-score
    z_scores = (flat_tensor - mean) / std

    # Define a threshold (e.g., |Z| > 3 is often considered an outlier)
    threshold = 3
    outliers = flat_tensor[torch.abs(z_scores) > threshold]

    #print('outliers: ', outliers)

    return mean, outliers.shape[0]

#input shape [1, 1024, 4096]
def get_pruning_rate(tensor_data):
    outlier_number_list = []
    pruning_rate_list = []
    for flatten_tensor in tensor_data[0]:
        outliers = get_outlier(flatten_tensor).shape[0]
        outlier_number_list.append(outliers)
        pruning_rate_list.append(1/outliers)

    #print('outliers: ', outlier_number_list)
    #print('rate: ', pruning_rate_list)


def prune_feature_vector(tensor_data, mean, rate):
    # Flatten the tensor while keeping the first dimension
    flat_tensor = tensor_data.view(1, -1)

    # Compute absolute distances from mean
    distances = torch.abs(flat_tensor - mean)

    # Get indices of the top n farthest values
    _, indices = torch.topk(distances, round(flat_tensor.numel() * rate), dim=1)

    # Create a mask and set selected elements to zero
    flat_tensor[0, indices[0]] = 0

    #print('after pruning... ', flat_tensor.view_as(tensor_data))
    # Reshape back to original shape
    return flat_tensor.view_as(tensor_data)

def dense_to_CSR(tensor_data):
    #Convert to sparse tensor
    sparse_tensor = tensor_data.to_sparse()

    #Convert sparse tensor to CSR format
    csr_tensor = sparse_tensor.to_sparse_csr()

    '''print('CSR size 1: ', csr_tensor.crow_indices().shape)
    print('CSR size 2: ', csr_tensor.col_indices().shape)
    print('CSR size 3: ',  csr_tensor.values().shape)

    print('CSR size 11: ', csr_tensor.crow_indices())
    print('CSR size 22: ', csr_tensor.col_indices())
    print('CSR size 33: ', csr_tensor.values())'''

    #return csr_tensor
    return [csr_tensor.crow_indices(), csr_tensor.col_indices(), csr_tensor.values()]

def csr_to_dense(csr_data):
    csr_tensor = torch.sparse_csr_tensor(
        crow_indices = csr_data[0],  # Row offsets
        col_indices = csr_data[1],  # Column indices (empty)
        values = csr_data[2])

    # Recover the sparse tensor from CSR
    recovered_sparse_tensor = csr_tensor.to_sparse()

    # Convert back to dense tensor to verify correctness
    recovered_dense_tensor = recovered_sparse_tensor.to_dense()

    return recovered_dense_tensor


def dense_to_CSC(tensor_data):
    # Step 2: Extract nonzero values and their indices
    values = tensor_data[tensor_data != 0]  # Nonzero values
    row_indices = tensor_data.nonzero()[:, 0]  # Row indices of nonzero values
    col_indices = tensor_data.nonzero()[:, 1]  # Column indices of nonzero values

    # Step 3: Compute `ccol_indices`
    num_cols = tensor_data.shape[1]
    ccol_indices = torch.zeros(num_cols + 1, dtype=torch.int32)

    for col in range(num_cols):
        ccol_indices[col + 1] = (col_indices == col).sum() + ccol_indices[col]

    # Step 4: Create the CSC tensor
    csc_tensor = [ccol_indices, row_indices, values]
    #csc_tensor = pack_tensors([ccol_indices, row_indices, values])
    #csc_tensor = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=tensor_data.shape)

    # Print results

    return csc_tensor

def csc_to_dense(csc_data):
    csc_tensor = torch.sparse_csc_tensor(
        ccol_indices = csc_data[0],  # Row offsets
        row_indices = csc_data[1],  # Column indices (empty)
        values = csc_data[2])

    # Recover the sparse tensor from CSR
    recovered_sparse_tensor = csc_tensor.to_sparse()

    # Convert back to dense tensor to verify correctness
    recovered_dense_tensor = recovered_sparse_tensor.to_dense()

    return recovered_dense_tensor


def pack_tensors(tensor_list, padding_value = 0.0):
    # Step 1: Get the maximum tensor length
    max_size = max(tensor.size(0) for tensor in tensor_list)

    # Step 2: Pad all tensors to the same length
    padded_tensors = [
        torch.nn.functional.pad(tensor, (0, max_size - tensor.size(0)), value=padding_value)
        for tensor in tensor_list
    ]

    # Step 3: Stack into a single tensor
    packed_tensor = torch.stack(padded_tensors, dim=0)  # Shape: [num_tensors, max_size]
    original_sizes = torch.tensor([tensor.size(0) for tensor in tensor_list], dtype=torch.int32)

    return [packed_tensor, original_sizes]


def unpack_tensors(packed_tensor, original_sizes):
    """
    Unpacks a single packed tensor back into a list of tensors.

    Args:
        packed_tensor (torch.Tensor): The packed tensor (padded).
        original_sizes (torch.Tensor): Tensor storing the original sizes.

    Returns:
        list of torch.Tensor: The unpacked tensors.
    """

    ccol_indices = packed_tensor[0].to(torch.int32)[0: original_sizes[0].item()]
    row_indices = packed_tensor[1].to(torch.int32)[0: original_sizes[1].item()]
    values = packed_tensor[2][0: original_sizes[2].item()]
    #return [packed_tensor[i, :size] for i, size in enumerate(original_sizes)]
    return [ccol_indices, row_indices, values]