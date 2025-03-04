import torch
import math
import msgpack
import lz4.frame
import numpy as np

def get_outlier2(flat_tensor):
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

def get_outlier(flat_tensor, M):
    mean = flat_tensor.double().mean()
    avg_abs_values = flat_tensor.abs().mean(dim=(1, 2), keepdim=True)

    # Count elements where abs(value) > M * avg_abs_value
    outlier_mask = flat_tensor.abs() > (M * avg_abs_values)
    outlier_counts = outlier_mask.sum(dim=(1, 2))  # Sum over height and width

    return outlier_counts, mean

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
    _, indices = torch.topk(distances, round(flat_tensor.numel() * rate), dim=1, largest=False)

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

def serialize_and_compress(start_idx, csr_out, ids, mask, idx, client_comp_time):
    """ Serializes and compresses data using MessagePack + LZ4 """
    # Convert tensors to CPU & byte buffers (for MessagePack compatibility)
    def tensor_to_bytes(tensor):
        return None if tensor is None else tensor.cpu().numpy().tobytes()

    tensor_data = {
        "ccol": tensor_to_bytes(csr_out[0]),
        "crow": tensor_to_bytes(csr_out[1]),
        "value": tensor_to_bytes(csr_out[2]),
        "ids": tensor_to_bytes(ids),
        "mask": tensor_to_bytes(mask)
    }

    # Store metadata for reconstruction
    data_packet = {
        "start_idx": start_idx,
        "tensor": {
            "ccol_shape": csr_out[0].shape if csr_out[0] is not None else None,
            "crow_shape": csr_out[1].shape if csr_out[1] is not None else None,
            "value_shape": csr_out[2].shape if csr_out[2] is not None else None,
            "ids_shape": ids.shape if ids is not None else None,
            "mask_shape": mask.shape if mask is not None else None,
            "index_dtype": str(csr_out[0].dtype) if csr_out[0] is not None else None,
            "tensor_dtype": str(csr_out[2].dtype) if csr_out[2] is not None else None,
            "data": tensor_data
        },
        "idx": idx,
        "client_comp_time": client_comp_time
    }

    #print('original: ', data_packet)
    # Serialize and compress
    packed_data = msgpack.packb(data_packet, use_bin_type=True)
    compressed_data = lz4.frame.compress(packed_data)

    return compressed_data


def decompress_and_deserialize(compressed_data):
    """ Decompresses and deserializes data using LZ4 + MessagePack """
    # Decompress data
    decompressed_data = lz4.frame.decompress(compressed_data)
    # Deserialize from MessagePack
    unpacked_data = msgpack.unpackb(decompressed_data, raw=False)
    #print('original: ', unpacked_data)
    print('start_idx', unpacked_data["start_idx"])

    torch_dtype_map = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.float16": np.float16,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.uint8": np.uint8,
    }


    # Reconstruct tensors
    try:
        index_dtype = torch_dtype_map[unpacked_data["tensor"]["index_dtype"]]
    except Exception as e:
        index_dtype = np.int32

    try:
        tensor_dtype = torch_dtype_map[unpacked_data["tensor"]["tensor_dtype"]]
    except Exception as e:
        tensor_dtype = torch.float32

    def restore_tensor(key, dtype):
        if unpacked_data["tensor"]["data"][key] is None:
            return None
        return torch.from_numpy(
            np.frombuffer(unpacked_data["tensor"]["data"][key], dtype=dtype).reshape(
                unpacked_data["tensor"][f"{key}_shape"]
            )
        ).cuda()

    tensor_ccol = restore_tensor("ccol", index_dtype)
    tensor_crow = restore_tensor("crow", index_dtype)
    tensor_value = restore_tensor("value", tensor_dtype)

    csr_out = [tensor_ccol, tensor_crow, tensor_value]

    ids = restore_tensor("ids", index_dtype)
    mask = restore_tensor("mask", tensor_dtype)

    return [unpacked_data["start_idx"], csr_out, ids, mask, unpacked_data[
        "idx"], unpacked_data["client_comp_time"]]

