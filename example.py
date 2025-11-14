#justin
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset_batch_tokenizer import BatchTokenizedDataset_A #this is what I am using to tokenize the dataset, so as to not get memory issues (look at data_tokenizer.py to see how it works)
#basically, batchTokenizedDataset_A creates a Dataset from my dataframes, and then tokenizes them in batches when used with DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #for gpu usage, if possible, kinda really helpful at improving speed of 
                                    #training, so check if your computer can use cuda and learn how to implement it, itll save you a lot of time


# task 13_A,,, example import: im using modernBERT for this, in order to binary classify the code examples as AI(1) or Human (0)  

df_train = pd.read_parquet("data/task_a/task_a_training_set_1.parquet") #datasets are in parquets, we have both task A and task B, so swap these around depending on what we wanna do
df_validation = pd.read_parquet("data/task_a/task_a_validation_set.parquet")
df_test = pd.read_parquet("data/task_a/task_a_test_set_sample.parquet")
df_train["code"] = df_train["code"].astype(str)


print(df_train["code"][0]) #example of printing the first code value in the dataframe,the columns are "code, generator, label, language"

print(df_train["code"])

model_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)   #pulls the tokenizer associated with whatever model we are using 
                                                        #(so pulls modernBERTs pretrained tokenizer for use)
                                                        
                                                        
    #i used commented line of code below at first, but my machine isnt capable of doing all of the encoding all at once,
    # but tokenizer_A does functionally the same thing in batches

#   inputs = tokenizer(df_train["code"].to_list(), return_tensors="pt", padding='max_length',truncation=True, max_length=8192).to(model.device) 
# #uses gpu or cpu with to(model.device), padding truncation just add extra spaces for shorter input, or truncating overflow


train_dataset = BatchTokenizedDataset_A(df_train["code"].to_list(), df_train["label"].to_list(),tokenizer, 8192) #uses 8192, the max length modernBERT is capable of. data_tokenizer py handles truncation etc
train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)

print("Tokenizing Successful")


                                                        
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)           #its possible to call "autoModelforCausalLLM", "AutoModelforSequenceClassification", and AutoModel, but this is just easier to do for now. I think we should experiment with the head/loss function for improvement of performance 
                                                        #which basically takes the base model (which were grabbing here), and then adds a head to it that does something with it, e.g. maybe adds dropout, adds some probability function for binary classification, (tanh, relu)


model.to(device)
model.train()
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5 #this is apparently what was best for DROIDDetect
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, eps=1e-8)


num_training_steps = NUM_EPOCHS * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)


for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss
        logits = outputs.logits
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")




