def standardize_loudness(loudness_tensor, mean, std, keys):
    '''
    Standardize the loudness tensor using the mean and std values for the specified keys.
    Args:
        loudness_tensor (torch.Tensor): Loudness tensor to be normalized.
        mean (dict): Dictionary containing mean loudness values for each instrument and global.
        std (dict): Dictionary containing std loudness values for each instrument and global.
        keys (list): List of keys (instruments) to standardize against.
    Returns:
        dict: Dictionary containing the standardized loudness tensors for each specified key.
    '''
    return {key: (loudness_tensor - mean[key]) / std[key] for key in keys}


def standardize_loudness_interpolated(loudness_tensor, mean, std, instrument1, instrument2, alpha):
    '''
    Standardize the loudness tensor using interpolated mean and std values between two instruments.
    Args:
        loudness_tensor (torch.Tensor): Loudness tensor to be normalized.
        mean (dict): Dictionary containing mean loudness values for each instrument and global.
        std (dict): Dictionary containing std loudness values for each instrument and global.
        instrument1 (str): Name of the first instrument.
        instrument2 (str): Name of the second instrument.
        alpha (float): Interpolation factor between 0 and 1.
    Returns:
        torch.Tensor: Standardized loudness tensor using interpolated mean and std values.
    '''
    interp_mean = (1 - alpha) * mean[instrument1] + alpha * mean[instrument2]
    interp_std = (1 - alpha) * std[instrument1] + alpha * std[instrument2]
    return (loudness_tensor - interp_mean) / interp_std
