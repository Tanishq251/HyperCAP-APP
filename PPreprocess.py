import numpy as np
import os
import pandas as pd
from scipy import io as sio
from matplotlib import pyplot as plt
import h5py
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
import img as util  # Ensure this module is correctly implemented and available
from torch.utils.data import Dataset, DataLoader

class DatasetPreprocess:
    def __init__(self, folder_name):
        """
        Initialize the DatasetPreprocess class with the folder name.

        Parameters:
        folder_name (str): The path to the folder containing the dataset files.
        """
        self.folder_name = folder_name
        self.data = None
        self.gt = None
        self.captions = None
        self.train_rows = None
        self.train_cols = None
        self.test_rows = None
        self.test_cols = None
        self.val_rows = None
        self.val_cols = None
        self.train_input_sub = None
        self.val_input = None
        self.test_input = None
        self.y_train_sub = None
        self.y_val = None
        self.y_test = None
        self.int_to_vector_dict = None
        self.text_data = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        self.text_encoder = None
        self.label_encoder = None

    def load_data(self):
        """
        Load the dataset files from the specified folder.

        Returns:
        tuple: (data, gt, captions)
        """
        # Search for the files in the folder
        data_file = None
        gt_file = None
        csv_file = None

        for file in os.listdir(self.folder_name):
            if file.endswith('_data.mat'):
                data_file = file
            elif file.endswith('_gt.mat'):
                gt_file = file
            elif file.endswith('.csv'):
                csv_file = file

        if data_file is None or gt_file is None:
            raise FileNotFoundError("Required .mat files not found in the folder.")

        # Load the .mat files using scipy.io.loadmat or h5py
        try:
            data_set_mat = sio.loadmat(os.path.join(self.folder_name, data_file))
            gt_mat = sio.loadmat(os.path.join(self.folder_name, gt_file))
            self.data = data_set_mat.get('data')
            self.gt = gt_mat.get('gt')
        except Exception as e:
            print(f"Error loading .mat files: {e}")
            try:
                with h5py.File(os.path.join(self.folder_name, data_file), 'r') as f:
                    self.data = f['data'][:]
                with h5py.File(os.path.join(self.folder_name, gt_file), 'r') as f:
                    self.gt = f['gt'][:]
            except Exception as e:
                print(f"Error loading .h5 files: {e}")
                raise

        print("Data loaded.")
        
        # Load the CSV file if provided
        if csv_file:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.captions = pd.read_csv(os.path.join(self.folder_name, csv_file), encoding=encoding)
                    break
                except UnicodeDecodeError as e:
                    continue
            else:
                raise ValueError("Unable to load CSV file with any of the tried encodings.")
            print("CSV file loaded.")
            
        return self.data, self.gt, self.captions

    def preprocess_text_data(self):
        """
        Tokenize and encode the text data.

        Returns:
        tuple: (encoded_text_data, text_encoder)
        """
        if self.text_data is None:
            raise ValueError("Text data not loaded.")

        # Tokenize and encode the text data
        self.text_encoder = LabelEncoder()
        encoded_text_data = self.text_encoder.fit_transform(self.text_data)
        
        print("Text data tokenized and encoded.")

        return encoded_text_data, self.text_encoder

    def determine_patch_size(self, preferred_sizes=[11, 16]):
        """
        Determine the appropriate patch size based on the dataset dimensions.

        Parameters:
        preferred_sizes (list): A list of preferred patch sizes.

        Returns:
        int: The determined patch size.
        """
        for size in preferred_sizes:
            if self.data.shape[1] >= size and self.data.shape[2] >= size:
                return size
        min_dim = min(self.data.shape[1], self.data.shape[2])
        return min_dim // 2 * 2 + 1  # Ensure the patch size is odd

    def split_text_data(self):
        """
        Split the text data into training and validation sets.

        Returns:
        tuple: (train_text, val_text)
        """
        class_column = 'Class name'
        grouped = self.captions.groupby(class_column)
        train_text_sub = []
        val_text_sub = []

        for class_label, group in grouped:
            train_sample = group.sample(frac=0.10, random_state=1)
            val_sample = group.drop(train_sample.index)

            train_text_sub.append(train_sample)
            val_text_sub.append(val_sample)

        train_text = pd.concat(train_text_sub)
        val_text = pd.concat(val_text_sub)

        print("Data split into training and validation sets.")
        
        return train_text, val_text

    def get_patchify_data(self, patch_size=None, train_fraction=0.10, rem_classes=[0]):
        """
        Preprocess the data by rescaling, splitting into training, validation, and test sets,
        and creating patches. Training set is 5% and validation set is the remaining data.

        Parameters:
        patch_size (int, optional): The size of the patches. If not provided, it will be determined automatically.
        train_fraction (float): The ratio of training size to the entire dataset.
        rem_classes (list): Class ids to exclude from analysis.

        Returns:
        tuple: ((train_input_sub, y_train_sub), (val_input, y_val), (test_input, y_test))
        """
        self.data = util.rescale_data(self.data)

        if patch_size is None:
            patch_size = self.determine_patch_size()
        else:
            if self.data.shape[1] < patch_size or self.data.shape[2] < patch_size:
                raise ValueError(f"Patch size {patch_size} is too large for the dataset dimensions.")

        print(f"Using patch size: {patch_size}")

        # First split into training (5%) and test
        (self.train_rows, self.train_cols), (self.test_rows, self.test_cols) = util.data_split(
            self.gt, train_fraction=train_fraction, rem_classes=rem_classes)

        # Now set validation to be all of what was previously test data
        self.val_rows = self.test_rows
        self.val_cols = self.test_cols

        train_pixel_indices_sub = (self.train_rows, self.train_cols)
        val_pixel_indices = (self.val_rows, self.val_cols)
        test_pixel_indices = ([], [])  # Empty test set since validation takes all remaining data

        catg_labels = np.unique([int(self.gt[idx[0], idx[1]]) for idx in zip(self.train_rows, self.train_cols)])
        self.int_to_vector_dict = util.label_2_one_hot(catg_labels)

        self.train_input_sub, self.y_train_sub = util.create_patch(
            data_set=self.data, gt=self.gt, pixel_indices=train_pixel_indices_sub,
            patch_size=patch_size, label_vect_dict=self.int_to_vector_dict)

        self.val_input, self.y_val = util.create_patch(
            data_set=self.data, gt=self.gt, pixel_indices=val_pixel_indices,
            patch_size=patch_size, label_vect_dict=self.int_to_vector_dict)

        # Set test set to empty arrays since we're not using it
        self.test_input = np.array([])
        self.y_test = np.array([])

        # print("Patching is done.")
        # print(f"Train input shape: {self.train_input_sub.shape}")
        # print(f"Validation input shape: {self.val_input.shape}")
        
        print("Patching is done.")

        return (self.train_input_sub, self.y_train_sub), (self.val_input, self.y_val), (self.test_input, self.y_test)

    def recon_patchify_data(self, patch_size, pixel_indices):
        """
        Reconstruct the HSI image from patches.

        Parameters:
        patch_size (int): The size of the patches.
        pixel_indices (tuple): Contains lists of integers, representing training pixel rows and columns.

        Returns:
        numpy.ndarray: The reconstructed HSI image.
        """
        def reconstruct_hsi_image(patches, original_shape, patch_size, pixel_indices):
            reconstructed_image = np.zeros(original_shape)
            patch_half = patch_size // 2
            count_matrix = np.zeros(original_shape[1:])

            for i, patch in enumerate(patches):
                row, col = pixel_indices[0][i], pixel_indices[1][i]
                start_row = max(row - patch_half, 0)
                end_row = min(row + patch_half + 1, original_shape[1])
                start_col = max(col - patch_half, 0)
                end_col = min(col + patch_half + 1, original_shape[2])

                patch_start_row = patch_half - (row - start_row)
                patch_end_row = patch_start_row + (end_row - start_row)
                patch_start_col = patch_half - (col - start_col)
                patch_end_col = patch_start_col + (end_col - start_col)

                reconstructed_image[:, start_row:end_row, start_col:end_col] += patch[:, patch_start_row:patch_end_row, patch_start_col:patch_end_col]
                count_matrix[start_row:end_row, start_col:end_col] += 1

            count_matrix[count_matrix == 0] = 1
            reconstructed_image /= count_matrix[np.newaxis, :, :]

            return reconstructed_image

        reconstructed_hsi = reconstruct_hsi_image(self.train_input_sub, self.data.shape, patch_size, pixel_indices)
        return reconstructed_hsi
    
        print("Reconstruction is done.")

    def plot_patchify(self, samples=5):
        """
        Plot the patches for training, validation, and test sets.

        Parameters:
        samples (int): The number of samples to plot.
        """
        def plot_patches(input_tensor, title, band_index=0, num_patches=samples):
            fig, axes = plt.subplots(1, num_patches, figsize=(15, 3))
            for i in range(num_patches):
                axes[i].imshow(input_tensor[i, band_index, :, :], cmap='jet')
                axes[i].axis('off')
            plt.suptitle(title)
            plt.show()

        plot_patches(self.train_input_sub, "Training Patches (Band 1)", band_index=0, num_patches=samples)
        plot_patches(self.val_input, "Validation Patches (Band 1)", band_index=0, num_patches=samples)

    def plot_raw_data(self, gt, data, samples=None):
        """
        Plot the raw HSI data and the ground truth map.

        Parameters:
        gt (numpy.ndarray): The ground truth data.
        data (numpy.ndarray): The hyperspectral image data.
        samples (int, optional): The number of samples to plot.
        """
        num_bands = data.shape[0]
        num_rows = (num_bands + 4) // 5

        if samples is not None:
            num_bands = min(samples, num_bands)

        plt.figure(figsize=(20, num_rows * 4))
        for band in range(num_bands):
            plt.subplot(num_rows, 5, band + 1)
            plt.imshow(data[band, :, :], cmap='jet')
            plt.title(f'Band {band + 1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(gt, cmap='jet')
        plt.title('Ground Truth Map')
        plt.axis('off')
        plt.show()

    def plot_reconstructed_data(self, reconstructed_hsi, samples=5):
        """
        Plot the reconstructed HSI data.

        Parameters:
        reconstructed_hsi (numpy.ndarray): The reconstructed HSI image.
        samples (int): The number of samples to plot.
        """
        num_bands = reconstructed_hsi.shape[0]
        num_rows = (num_bands + 4) // 5

        plt.figure(figsize=(20, num_rows * 4))
        for band in range(min(samples, num_bands)):
            plt.subplot(num_rows, 5, band + 1)
            plt.imshow(reconstructed_hsi[band, :, :], cmap='jet')
            plt.title(f'Band {band + 1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()