from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def load_data(
            script=None,
            lang=None,
            shuffle_seed=None,
            select_range=None,
            split="train",
            output_columns=["im", "text"], 
            **kwargs
            ):

    # load the dataset
    dataset = load_dataset("CATMuS/medieval", split=split)

    # Create an index dataset for quicker filtering
    index_dataset = dataset.select_columns(["script_type", "language"])

    # Generate an index for those columns and map to original dataset
    index_dataset = index_dataset.map(lambda example, idx: {'original_idx': idx}, with_indices=True)

    # Filter by script
    if script:
        print("Filtering by script...")
        index_dataset = index_dataset.filter(lambda example, index: example['script_type'] == script, with_indices=True)

    # Filter by language
    if lang:
        print("Filtering by language...")
        index_dataset = index_dataset.filter(lambda example, index: example['language'] == lang, with_indices=True)
    
    # Optionally randomize the data
    if shuffle_seed:
        index_dataset = index_dataset.shuffle(seed=shuffle_seed)

    # Optionally select the range of the data
    if select_range:
        if select_range < len(index_dataset):
            index_dataset = index_dataset.select(range(select_range))
        else:
            print(f"The total dataset is only {len(dataset)}")
    
    # Grab the relevant indices from the original dataset
    dataset = dataset.select(index_dataset["original_idx"])

    # Select the columns that are necessary
    dataset = dataset.select_columns(output_columns)
    print(dataset)
    return dataset

class HandwrittenTextDataset(Dataset):
    def __init__(self, dataset, processor, max_target_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_data = self.dataset[idx]['im']
        text = self.dataset[idx]['text']

        image = Image.fromarray(np.array(image_data)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length,
                                          truncation=True).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

def split_dataset(dataset, test_size=0.2):
    train_size = int((1 - test_size) * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, processor, batch_size=4):
    train_dataset = HandwrittenTextDataset(train_dataset, processor)
    test_dataset = HandwrittenTextDataset(test_dataset, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader