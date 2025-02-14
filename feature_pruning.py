def get_outlier(tensor_data):
    q1 = tensor_data.quantile(0.25, dim=2, keepdim=True)
    q3 = tensor_data.quantile(0.75, dim=2, keepdim=True)
    iqr = q3 - q1

    # Outlier threshold
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (tensor_data < lower_bound) | (tensor_data > upper_bound)

    print(outliers)  # Boolean mask of outliers