#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:15:29 2024

@author: maguo
"""
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import torch.nn as nn

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from scipy.stats import mode

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from tabulate import tabulate
from tqdm import tqdm
from tqdm import trange
import random

import spacy
nlp = spacy.load("en_core_web_sm")

MAX_LEN = 510 #token size

# Prepare for training
#train_index = pd.read_csv("training_example_indices.txt",header=None)
#test_indx = pd.read_csv("testing_example_indices.txt",header=None)
label_matrix = pd.read_csv("label_matrix_merge_with_none_new.txt", sep=" ", header=None)
file_path = "removed_meta2_reduced.json"
with open(file_path, 'r') as file:
    data = json.load(file)

text = []
for files in data:
    print("processing ", files)
    paragraphs = ' '.join(data[files]['Client_Text_Replaced_Two'])
    text.append(paragraphs)
    
df_text = pd.DataFrame(text)    
df_dep = label_matrix[2]
df_anx = label_matrix[1]
    
# Tokenize for BERT
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

def sentence_tokenizer(input_text):
    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def divide_doc(input_text, tokenizer):
    sentences = sentence_tokenizer(input_text)
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in tokenized_sentences:
        if current_length + len(sentence) > MAX_LEN:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.extend(sentence)
        current_length += len(sentence)
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def prepare_for_bert(chunks):
    """Prepare the chunks for BERT processing, adding special tokens."""
    prepared_chunks = []
    for chunk in chunks:
        # Add [CLS] at the beginning and [SEP] at the end
        prepared_chunk = ["[CLS]"] + chunk + ["[SEP]"]
        # Convert tokens to their IDs
        input_ids = tokenizer.convert_tokens_to_ids(prepared_chunk)
        prepared_chunks.append(input_ids)
    return prepared_chunks

def preprocessing_chuncks(input_text, tokenizer):
    chunks = divide_doc(input_text, tokenizer)  # Using 510 to account for [CLS] and [SEP]
    
    # Process each chunk separately
    encoding_chunks = [tokenizer.encode_plus(
                            " ".join(chunk),  # Convert chunk back to string
                            add_special_tokens=True,
                            max_length=512,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt',
                            truncation=True
                       ) for chunk in chunks]
    
    # Combine the processed chunks
    """
    combined_encoding = {
        'input_ids': torch.cat([ec['input_ids'] for ec in encoding_chunks], dim=0),
        'attention_mask': torch.cat([ec['attention_mask'] for ec in encoding_chunks], dim=0)
    }
    # Note: token_type_ids are not needed for BERT models in single-sequence tasks
    """
    return encoding_chunks

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512,
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )


def prepare_data(df_text, label):

    input_ids = []
    attention_masks = []
    batch_size = 16
    label_array = label.values
    
    for sample in df_text[0]:
        print("tokenizing ...")
        encoding_dict = preprocessing(sample, tokenizer)
        input_ids.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])


    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    df_label = torch.tensor(label_array, dtype=torch.float)
    
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, df_label, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, df_label, random_state=2018, test_size=0.1)
    
    batch_size = 32
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader

def prepare_chunked_data(df_text, label):
    
    input_ids = []
    attention_masks = []
    document_ids = []
    chunk_labels = []
    label_array = label.values
    
    for doc_id, (sample, label) in enumerate(tqdm(zip(df_text[0], label_array), total=len(label_array))):
        
        encoding_dict = preprocessing_chuncks(sample, tokenizer)
        for chunk_encoding in encoding_dict:
            input_ids.append(chunk_encoding['input_ids'])
            attention_masks.append(chunk_encoding['attention_mask'])
            chunk_labels.append(label)  # Duplicate the label for each chunk
            document_ids.append(doc_id)  # Keep track of the original document ID for each chunk
    
    """
    for sample in df_text[0]:
        print("tokenizing ...")
        encoding_dict = preprocessing_chuncks(sample, tokenizer)
        input_ids.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])
    """

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    #df_label = torch.tensor(label_array, dtype=torch.float)
    chunk_labels = torch.tensor(chunk_labels, dtype=torch.int64)
    
    save_dict = {
    'encoding_dict': encoding_dict,
    'attention_masks': attention_masks,
    'chunk_labels': chunk_labels,
    'document_ids': document_ids,
    'input_ids': input_ids
    }

    # Saving to disk
    torch.save(save_dict, 'data_tensors.pth')
    
def load_and_prepare():
    loaded_dict = torch.load('data_tensors.pth', map_location=torch.device('cpu'))
    device = torch.device("cpu")
    
    input_ids = loaded_dict['input_ids']
    attention_masks = loaded_dict['attention_masks']
    chunk_labels = loaded_dict['chunk_labels']
    document_ids = loaded_dict['document_ids']
    document_ids = torch.tensor(document_ids)
    chunk_labels = torch.tensor(chunk_labels, dtype=torch.int64)
    chunk_labels = chunk_labels[:, [0,2]]
    
    
    train_inputs, validation_inputs, train_labels, validation_labels, train_id, val_id = train_test_split(input_ids, chunk_labels, document_ids, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, chunk_labels, random_state=2018, test_size=0.1)
    
    batch_size = 16
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_id)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, val_id)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader


def b_tp(preds, labels):
    '''Returns True Positives (TP): count of correct predictions of actual class 1'''
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
    '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
    '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
    '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
    '''
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    '''
    preds = np.argmax(preds, axis = 1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity

def majority_vote(all_predictions, all_document_ids, all_label_ids):
    grouped_predictions = defaultdict(list)
    document_labels = {}
    
    for prediction, doc_id, label in zip(all_predictions, all_document_ids, all_label_ids):
        grouped_predictions[doc_id].append(prediction)
        # Assuming each chunk from the same document has the same label
        document_labels[doc_id] = label
    
    # Perform a majority vote within each group
    final_predictions = []
    final_labels = []
    for doc_id, preds in grouped_predictions.items():
        # Use mode to find the most common prediction in the list
        # Mode returns both the value and the count; [0][0] accesses the first mode (most common element)
        most_common_pred = mode(preds)[0][0]
        final_predictions.append(most_common_pred)
        final_labels.append(document_labels[doc_id])
    
    return np.array(final_predictions), np.array(final_labels)

def train_single_class(train_dataloader, validation_dataloader, epochs=2):
    print("starting training...")
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )
    
    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = 5e-5,
                                  eps = 1e-08
                                  )
    
    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    total_steps = len(train_dataloader) * epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
    
        # Progress bar (tqdm) can be wrapped around any iterable
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = tuple(t.to(device) for t in batch)  # Move batch to device
            b_input_ids, b_input_mask, b_labels,_ = batch
            #b_labels = b_labels.to(torch.int64)
            b_labels = b_labels.to(torch.float)
    
            model.zero_grad()  # Clear previously calculated gradients
    
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
    
            loss.backward()  # Perform backpropagation to calculate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
    
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate schedule
    
        # Calculate the average loss over the training data
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.2f}")
        model.save_pretrained(f"bert_model_{epoch}")
        # ========== Validation ==========
    
        # Set model to evaluation mode
        model.eval()
    
        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []
        
        all_predictions = []
        all_label_ids = []
        all_document_ids = []
    
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_document_ids = batch
            with torch.no_grad():
              # Forward pass
              eval_output = model(b_input_ids, 
                                  token_type_ids = None, 
                                  attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            all_predictions.extend(logits)
            all_label_ids.extend(label_ids)
            all_document_ids.extend(b_document_ids.to('cpu').numpy())
            
            final_pred, final_label = majority_vote(all_predictions, all_document_ids, all_label_ids)
            
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)
    
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
    


def train_multi_class(train_dataloader, validation_dataloader, num_class=3, epochs=2):
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = num_class,
        output_attentions = False,
        output_hidden_states = False,
    )
    
    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = 5e-5,
                                  eps = 1e-08
                                  )
    
    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    total_steps = len(train_dataloader) * epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
    
        # Progress bar (tqdm) can be wrapped around any iterable
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = tuple(t.to(device) for t in batch)  # Move batch to device
            b_input_ids, b_input_mask, b_labels = batch
            #b_labels = b_labels.to(torch.int64)
    
            model.zero_grad()  # Clear previously calculated gradients
    
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
            loss = loss_fn(logits, b_labels)
            total_loss += loss
    
            loss.backward()  # Perform backpropagation to calculate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
    
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate schedule
    
        # Calculate the average loss over the training data
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.2f}")
    
        # Validation step can also be added here to monitor model performance
    
    
        # ========== Validation ==========
    
        # Set model to evaluation mode
        model.eval()
    
        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []
    
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
              # Forward pass
              eval_output = model(b_input_ids, 
                                  token_type_ids = None, 
                                  attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)
    
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

#prepare_chunked_data(df_text, label_matrix)
torch.cuda.empty_cache()
train_dataloader, validation_dataloader = load_and_prepare()
train_single_class(train_dataloader, validation_dataloader)




