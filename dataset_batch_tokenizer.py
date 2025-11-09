import torch
from torch.utils.data import Dataset, DataLoader


'''
A way to tokenize the dataset using pytorch Dataset, so as to not save the entire dataset as one large tensor

structure taken from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html "Creating a Custom Dataset for your files '''


#I am creating a named tokenizer A here so that if we decide to tackle task B, we just have to create a tokenizer_B class that does something similar
class BatchTokenizedDataset_A(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self,idx):
        code = str(self.codes[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(code, return_tensors='pt', padding='max_length', truncation=True, max_length = self.max_length)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label,dtype=torch.long)
        }