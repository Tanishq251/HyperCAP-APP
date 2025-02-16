
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import random

def load_patches(input_data, gt_data, patch_size):
    """
    Extracts square patches from the input data and ground truth.

    Parameters:
        input_data (np.ndarray): The hyperspectral image data of shape (bands, length, width).
        gt_data (np.ndarray): The ground truth data of shape (length, width).
        patch_size (int): The size of the square patch.

    Returns:
        patches (np.ndarray): The extracted patches of shape (samples, patch_size, patch_size, bands).
        labels (np.ndarray): The corresponding labels of shape (samples,).
    """
    margin = patch_size // 2
    padded_input = np.pad(input_data, ((0, 0), (margin, margin), (margin, margin)), mode='constant', constant_values=0)
    padded_gt = np.pad(gt_data, ((margin, margin), (margin, margin)), mode='constant', constant_values=0)

    patches = []
    labels = []

    for i in range(margin, padded_input.shape[1] - margin):
        for j in range(margin, padded_input.shape[2] - margin):
            if padded_gt[i, j] != 0:  # Exclude background pixels
                patch = padded_input[:, i - margin:i + margin + 1, j - margin:j + margin + 1]
                patches.append(patch)
                labels.append(padded_gt[i, j])

    patches = np.array(patches)
    labels = np.array(labels)

    return patches, labels

def preprocess_patches(patches):
    """
    Preprocesses patches to match the input format of PyTorch models.

    Parameters:
        patches (np.ndarray): The extracted patches of shape (samples, bands, patch_size, patch_size).

    Returns:
        reshaped_patches (np.ndarray): The reshaped patches of shape (samples, bands, patch_size, patch_size).
    """
    # No need to transpose, as the format is already (samples, bands, patch_size, patch_size)
    return patches

def prepare_data(input_data, gt_data, patch_size):
    """
    Prepares data by extracting patches and preprocessing them.

    Parameters:
        input_data (np.ndarray): The hyperspectral image data of shape (bands, length, width).
        gt_data (np.ndarray): The ground truth data of shape (length, width).
        patch_size (int): The size of the square patch.

    Returns:
        prepared_patches (np.ndarray): The preprocessed patches ready for model input.
        labels (np.ndarray): The corresponding labels of the patches.
    """
    patches, labels = load_patches(input_data, gt_data, patch_size)
    prepared_patches = preprocess_patches(patches)
    return prepared_patches, labels

def shuffle_data(patches, labels):
    """
    Shuffles patches and labels to ensure randomness.

    Parameters:
        patches (np.ndarray): The preprocessed patches.
        labels (np.ndarray): The corresponding labels.

    Returns:
        shuffled_patches (np.ndarray): Shuffled patches.
        shuffled_labels (np.ndarray): Shuffled labels.
    """
    return shuffle(patches, labels, random_state=42)

import numpy as np
import math

def data_split(gt, train_fraction=0.10, rem_classes=None, split_method='same_hist'):
    """
    Outputs list of row and column indices for training and test sets with balanced class distribution.

    Parameters:
        gt (np.ndarray): A 2-D numpy array, containing integer values representing class ids.
        train_fraction (float): The ratio of training size to the entire dataset.
        rem_classes (list): Class ids to exclude from analysis.
        split_method (str or dict): Splitting strategy ('same_hist' or dict specifying class sample counts).

    Returns:
        tuple: ((train_rows, train_cols), (test_rows, test_cols))
    """
    if rem_classes is None:
        rem_classes = []

    # Get unique classes and their counts
    catgs, counts = np.unique(gt, return_counts=True)
    mask = np.isin(catgs, rem_classes, invert=True)
    catgs, counts = catgs[mask], counts[mask]
    
    # Calculate initial split
    num_pixels = sum(np.isin(gt, rem_classes, invert=True).ravel())
    catg_ratios = counts / np.sum(counts)
    num_sample_catgs = np.array([math.floor(elm) for elm in (catg_ratios * num_pixels)], dtype='int32')
    all_catg_indices = [np.where(gt == catg) for catg in catgs]

    train_rows, train_cols, test_rows, test_cols = [], [], [], []
    train_counts, test_counts = [], []

    # Initial split of data
    catg_with_indices = zip(num_sample_catgs, all_catg_indices, catgs)
    for num_samples, indices, catg in catg_with_indices:
        all_indices = np.arange(num_samples, dtype='int32')
        
        # Calculate initial training size
        if split_method == 'same_hist':
            train_size = int(num_samples * train_fraction)
        elif isinstance(split_method, dict):
            train_size = split_method.get(catg, int(num_samples * train_fraction))
        else:
            raise ValueError('Please select a valid option')

        # Initial random split
        rand_train_indices = np.random.choice(all_indices, size=min(train_size, num_samples), replace=False)
        rand_test_indices = np.setdiff1d(all_indices, rand_train_indices, assume_unique=True)

        # Store initial split
        train_rows.append(indices[0][rand_train_indices].tolist())
        train_cols.append(indices[1][rand_train_indices].tolist())
        test_rows.append(indices[0][rand_test_indices].tolist())
        test_cols.append(indices[1][rand_test_indices].tolist())
        
        # Store counts
        train_counts.append(len(rand_train_indices))
        test_counts.append(len(rand_test_indices))

    # Balance classes
    train_counts = np.array(train_counts)
    test_counts = np.array(test_counts)
    
    # Find classes with severe imbalance
    median_train_count = np.median(train_counts)
    min_acceptable_samples = median_train_count * 0.3  # Threshold for minimum acceptable samples
    
    # Balance severely imbalanced classes
    for i in range(len(train_counts)):
        if train_counts[i] < min_acceptable_samples:
            # If test set has more samples, swap some samples
            samples_needed = int(min_acceptable_samples - train_counts[i])
            
            if test_counts[i] > samples_needed:
                # Move samples from test to train
                test_indices = range(len(test_rows[i]))
                move_indices = np.random.choice(test_indices, size=samples_needed, replace=False)
                
                # Move selected samples to training set
                new_train_rows = train_rows[i] + [test_rows[i][j] for j in move_indices]
                new_train_cols = train_cols[i] + [test_cols[i][j] for j in move_indices]
                
                # Remove moved samples from test set
                remain_indices = np.setdiff1d(test_indices, move_indices)
                new_test_rows = [test_rows[i][j] for j in remain_indices]
                new_test_cols = [test_cols[i][j] for j in remain_indices]
                
                # Update sets
                train_rows[i], train_cols[i] = new_train_rows, new_train_cols
                test_rows[i], test_cols[i] = new_test_rows, new_test_cols
            else:
                # If test set doesn't have enough samples, swap entirely
                train_rows[i], test_rows[i] = test_rows[i], train_rows[i]
                train_cols[i], test_cols[i] = test_cols[i], train_cols[i]

    # Combine all indices
    train_rows_combined = [item for sublist in train_rows for item in sublist]
    train_cols_combined = [item for sublist in train_cols for item in sublist]
    test_rows_combined = [item for sublist in test_rows for item in sublist]
    test_cols_combined = [item for sublist in test_cols for item in sublist]

    return (train_rows_combined, train_cols_combined), (test_rows_combined, test_cols_combined)

def rescale_data(data_set, method='standard'):
    """
    Rescales image dataset using different methods.

    Parameters:
        data_set (np.ndarray): Contains image data with format: (bands, length, width).
        method (str): Rescaling method ('standard', 'zero_mean', 'min_max_norm', 'mean_norm').

    Returns:
        np.ndarray: Rescaled data.
    """
    if not isinstance(data_set, np.ndarray) or len(data_set.shape) != 3:
        raise ValueError('data_set must be a 3-D numpy array!')

    rescaled_data = np.zeros(data_set.shape)
    if method == 'standard':
        for i in range(data_set.shape[0]):
            channel = data_set[i, :, :]
            rescaled_data[i, :, :] = (channel - np.mean(channel)) / np.std(channel)
    elif method == 'zero_mean':
        for i in range(data_set.shape[0]):
            channel = data_set[i, :, :]
            rescaled_data[i, :, :] = channel - np.mean(channel)
    elif method == 'min_max_norm':
        for i in range(data_set.shape[0]):
            channel = data_set[i, :, :]
            rescaled_data[i, :, :] = (channel - np.amin(channel)) / (np.amax(channel) - np.amin(channel))
    elif method == 'mean_norm':
        for i in range(data_set.shape[0]):
            channel = data_set[i, :, :]
            rescaled_data[i, :, :] = (channel - np.mean(channel)) / (np.amax(channel) - np.amin(channel))
    else:
        raise ValueError(f'{method} is not a valid method.')

    return rescaled_data

def reduce_dim(img_data, n_components=0.95):
    """
    Reduces spectral dimension of image data using PCA.

    Parameters:
        img_data (np.ndarray): Contains image data with shape: (bands, length, width).
        n_components (float or int): Fraction of variance or number of components.

    Returns:
        np.ndarray: Transformed data with reduced dimensions.
    """
    img_shape = img_data.shape
    img_unravel = img_data.reshape(img_shape[0], -1).T

    pca = PCA(n_components=n_components)
    unravel_transformed = pca.fit_transform(img_unravel)

    n_col = unravel_transformed.shape[1]
    img_data_transformed = unravel_transformed.T.reshape(n_col, img_shape[1], img_shape[2])

    return img_data_transformed

def label_2_one_hot(label_list):
    """
    Creates a dictionary mapping class labels to one-hot vectors.

    Parameters:
        label_list (list): Contains class labels.

    Returns:
        dict: A mapping of class labels to one-hot vectors.
    """
    catgs = np.unique(label_list)
    num_catgs = len(catgs)
    one_hot_dict = {elm: np.eye(num_catgs)[idx] for idx, elm in enumerate(catgs)}
    return one_hot_dict

def one_hot_2_label(int_to_vector_dict):
    """
    Converts a one-hot dictionary to an integer label dictionary.

    Parameters:
        int_to_vector_dict (dict): Maps integers to one-hot vectors.

    Returns:
        dict: A mapping of one-hot vectors to integers.
    """
    return {tuple(v): k for k, v in int_to_vector_dict.items()}

def create_patch(data_set, gt, pixel_indices, patch_size=5, label_vect_dict=None):
    """
    Creates input tensors.

    Parameters:
        data_set (np.ndarray): Contains image data with format: (bands, length, width).
        gt (np.ndarray): Contains integers, representing different categories.
        pixel_indices (tuple): Contains lists of integers, representing training pixel rows and columns.
        patch_size (int): Represents patch size.
        label_vect_dict (dict): Associates int labels to a one-hot vector.

    Returns:
        input_tensor (np.ndarray): Input tensor with format: (num_samples, bands, patch_size, patch_size).
        target_tensor (np.ndarray): Target tensor with one_hot format.
    """
    rows = pixel_indices[0]
    cols = pixel_indices[1]

    if len(rows) != len(cols):
        raise ValueError("Unmatched number of rows and columns. The number of rows is {}, but the number of columns is {}".format(len(rows), len(cols)))

    max_row, max_col = (data_set.shape[1]-1), (data_set.shape[2]-1)
    sample_size = len(rows)
    input_tensor = np.zeros(shape=(sample_size, data_set.shape[0], patch_size, patch_size))
    catg_labels = []

    for idx in np.arange(sample_size):
        patch = np.zeros(shape=(data_set.shape[0], patch_size, patch_size))
        patch_center = (rows[idx], cols[idx])
        patch_top_row = patch_center[0] - patch_size // 2
        patch_left_col = patch_center[1] - patch_size // 2
        top_lef_idx = (patch_top_row, patch_left_col)

        catg_labels.append(gt[rows[idx], cols[idx]])
        for i in np.arange(patch_size):
            for j in np.arange(patch_size):
                patch_idx = (top_lef_idx[0] + i, top_lef_idx[1] + j)
                if (patch_idx[0] >= 0) and (patch_idx[0] <= max_row) and (patch_idx[1] >= 0) and (patch_idx[1] <= max_col):
                    patch[:, i, j] = data_set[:, patch_idx[0], patch_idx[1]]
        input_tensor[idx, :, :, :] = patch

    if label_vect_dict is None:
        label_vect_dict = label_2_one_hot(np.unique(gt))

    target_tensor = np.array([label_vect_dict.get(label) for label in catg_labels])
    return input_tensor, target_tensor

def val_split(rows, cols, gt, val_fraction=0.1, rem_classes=None, split_method='same_hist'):
    if rem_classes is None:
        rem_classes = [-1]

    gt_no_test = np.zeros(shape=gt.shape, dtype='int').reshape(gt.shape) - 1
    for elm in zip(rows, cols):
        gt_no_test[rows, cols] = gt[rows, cols]

    (train_rows, train_cols), (val_rows, val_cols) = data_split(gt_no_test, 1 - val_fraction, rem_classes, split_method)

    return (train_rows, train_cols), (val_rows, val_cols)

def calc_metrics(nn_model, test_inputs, y_test, int_to_vector_dict, verbose=True):
    """
    Calculates model performance metrics on test data.

    Parameters:
        nn_model (torch.nn.Module): Trained neural network model containing metrics information.
        test_inputs (torch.Tensor): Input tensor containing test inputs.
        y_test (torch.Tensor): Contains target test data with one_hot format.
        int_to_vector_dict (dict): Associates class int category labels to its corresponding one_hot format.

    Returns:
        model_metrics (dict): A dictionary with int keys representing category labels and list of model error and performance metrics as values.
    """
    vector_2_label = one_hot_2_label(int_to_vector_dict)
    test_catgs, test_catg_counts = np.unique([vector_2_label.get(tuple(elm)) for elm in y_test], return_counts=True)

    from_to_list = []
    res_container = [(elm, []) for elm in test_catgs]

    i = 0
    for elm in test_catg_counts:
        from_idx = i
        to_idx = i + elm
        i += elm
        from_to_list.append((from_idx, to_idx))

    for elm in zip(res_container, from_to_list):
        x = test_inputs[elm[1][0]:elm[1][1], :, :, :]
        y = y_test[elm[1][0]:elm[1][1], :]
        test_metrics = nn_model.evaluate(x=x, y=y, verbose=False)
        elm[0][-1].append(test_metrics)

    model_metrics = dict([(elm[0], elm[-1]) for elm in res_container])
    if verbose:
        for key, val in model_metrics.items():
            print(key, val)
    return model_metrics

def plot_partial_map(nn_model, gt, pixel_indices, input_tensor, targ_tensor, int_to_vector_dict, plo=True):
    """
    Plots prediction map using a trained model and inputs.

    Parameters:
        nn_model (torch.nn.Module): Trained using input data.
        gt (np.ndarray): A 2-D numpy array containing int labels.
        pixel_indices (tuple): Contains arrays of rows and columns of input pixels with format: (row_array, col_array).
        input_tensor (torch.Tensor): Contains input_tensor consistent with the nn_model inputs.
        targ_tensor (torch.Tensor): Target tensor, containing one_hot format of label data.
        int_to_vector_dict (dict): Associates int labels to their corresponding one_hot format.
        plo (bool): If True, plots the map.

    Returns:
        gt_pred_map (np.ndarray): A 2-D numpy.ndarray, representing predicted labels.
    """
    rows, cols = pixel_indices[0], pixel_indices[1]
    vect_2_label_dict = one_hot_2_label(int_to_vector_dict)
    y_pred_vectors = nn_model(input_tensor)
    y_pred = np.zeros(y_pred_vectors.shape[0], dtype=int)
    for elm in enumerate(y_pred_vectors):
        max_idx, *not_used = np.where(elm[1] == np.amax(elm[1]))
        predicted_vec = np.eye(1, y_pred_vectors.shape[1], k=max_idx[0], dtype=int).ravel()
        y_pred[elm[0]] = vect_2_label_dict.get(tuple(predicted_vec))

    map_shape = gt.shape
    gt_pred_map = np.zeros(map_shape, dtype=int)
    for elm in enumerate(zip(rows, cols)):
        gt_pred_map[elm[1]] = y_pred[elm[0]]

    if plo:
        plt.imshow(gt_pred_map)
    return gt_pred_map

def split_text_data_based_on_spatial(text_df, gt, train_rows, train_cols, random_seed=42):
    """
    Split textual data based on the counts obtained from spatial data splitting.
    
    Parameters:
    -----------
    text_df : pandas.DataFrame
        DataFrame containing textual data with 'Class name' column
    gt : numpy.ndarray
        Ground truth data array
    train_rows : numpy.ndarray
        Row indices of training samples from spatial split
    train_cols : numpy.ndarray
        Column indices of training samples from spatial split
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (train_text, val_text) containing split DataFrames
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get training labels and counts from spatial split
    train_labels = gt[train_rows, train_cols]
    train_counts = np.bincount(train_labels)
    
    # Get unique class names and create mapping
    unique_class_names = text_df['Class name'].unique()
    class_name_to_index = {class_name: index for index, class_name in enumerate(unique_class_names, start=1)}
    
    # Split textual data based on spatial counts
    train_indices = []
    val_indices = []
    
    for class_name, class_index in class_name_to_index.items():
        # Skip if class wasn't in spatial training set
        if class_index >= len(train_counts) or train_counts[class_index] == 0:
            continue
            
        # Get samples for current class
        class_samples = text_df[text_df['Class name'] == class_name].index.tolist()
        random.shuffle(class_samples)
        
        # Use same number of samples as in spatial training set
        train_count = train_counts[class_index]
        train_indices.extend(class_samples[:train_count])
        val_indices.extend(class_samples[train_count:])
    
    # Create training and validation sets
    train_text = text_df.loc[train_indices].reset_index(drop=True)
    val_text = text_df.loc[val_indices].reset_index(drop=True)
    
    print("Text data splitted based on spatial split.")
    
    return train_text, val_text

def print_split_info(train_text, val_text, class_column='Class name'):
    """
    Print information about the text data split.
    
    Parameters:
    -----------
    train_text : pandas.DataFrame
        Training text data
    val_text : pandas.DataFrame
        Validation text data
    class_column : str
        Name of the column containing class names
    """
    print(f"Training set shape: {train_text.shape}")
    print(f"Validation set shape: {val_text.shape}")
    
    print("\nClass-wise sample counts for training:")
    train_class_counts = train_text[class_column].value_counts()
    for class_name, count in train_class_counts.items():
        print(f"Class '{class_name}': {count} samples")
    
    print("\nClass-wise sample counts for validation:")
    val_class_counts = val_text[class_column].value_counts()
    for class_name, count in val_class_counts.items():
        print(f"Class '{class_name}': {count} samples")