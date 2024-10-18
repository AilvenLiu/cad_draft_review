import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class OpenImagesDataset(Dataset):
    def __init__(self, images_dir, annotations_file, vocab, transform=None, max_seq_length=50):
        """
        Initializes the dataset with images and annotations.

        Args:
            images_dir (str): Directory containing image tiles.
            annotations_file (str): Path to the annotations JSON file.
            vocab (Vocab): Vocabulary object for encoding labels.
            transform (callable, optional): Transformations to be applied to images.
            max_seq_length (int): Maximum sequence length for captions.
        """
        self.images_dir = images_dir
        self.annotations = self.load_annotations(annotations_file)
        self.transform = transform
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def load_annotations(self, annotations_file):
        with open(annotations_file, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        bboxes = torch.tensor(img_info['bbox'], dtype=torch.float32)
        labels = torch.tensor(self.vocab.numericalize(img_info['category']), dtype=torch.long)
        
        return image, bboxes, labels