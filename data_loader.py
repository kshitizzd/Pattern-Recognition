import os
import numpy as np
from PIL import Image


class MNISTLoader:
    """Load MNIST images from folders organized by digit (0-9)"""

    def __init__(self, root_dir, train_ratio=0.8):
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.images = []
        self.labels = []

    def load_data(self):
        """Load all images from digit folders"""
        for digit in range(10):
            digit_folder = os.path.join(self.root_dir, str(digit))
            if not os.path.exists(digit_folder):
                continue

            for img_file in os.listdir(digit_folder):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(digit_folder, img_file)
                    img = Image.open(img_path).convert("L")  # Grayscale
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
                    self.images.append(img_array)
                    self.labels.append(digit)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.images)} images")

    def split_data(self, seed=42):
        """Split into train and test sets"""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.images))
        split_idx = int(len(indices) * self.train_ratio)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        X_train = self.images[train_indices]
        y_train = self.labels[train_indices]
        X_test = self.images[test_indices]
        y_test = self.labels[test_indices]

        return X_train, y_train, X_test, y_test


