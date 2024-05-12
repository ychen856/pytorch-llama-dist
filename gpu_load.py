from numba import cuda
import numpy as np
import time

# Define a CUDA kernel function
@cuda.jit
def kernel(io_array):
    # Calculate the index in the array
    x, y = cuda.grid(2)

    # Do some arbitrary computation
    io_array[x, y] *= x  # Multiply each element by its x-coordinate

# Define the size of the array
array_size = (16384, 16384)

# Allocate memory on the GPU
d_array = cuda.device_array(array_size)

# Define the grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = (array_size[0] + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (array_size[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Launch the kernel
while 1:
    for i in range (0, 10):
        kernel[blockspergrid, threadsperblock](d_array)
    time.sleep(1)

# Copy the result back to the CPU
result = d_array.copy_to_host()

# Print the result
print(result)
