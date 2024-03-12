import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import sys
# print(sys.executable)

import fitz
import os
import re
import torch
from textblob import TextBlob
from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import string
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset

import logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# Load label
labels_df = pd.read_csv('training_labels.csv')

# Load Transcript PDF File
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

pdf_dir = r'C:\Users\SZ4291\OneDrive - Zebra Technologies\Documents\Shawn\Stanford\CS229\Project\Training Data\Transcripts'
pdf_files = os.listdir(pdf_dir)


# Function to clean and match PDF file names
def match_pdf_to_label(pdf_filename, labels_df):
    # Remove extension and any known formatting from the PDF filename
    clean_pdf_filename = pdf_filename.lower().replace('.pdf', '').replace('_', ' ')
    # Attempt to match the cleaned PDF filename with entries in the labels dataframe
    for _, row in labels_df.iterrows():
        if row['file_name'].lower() in clean_pdf_filename:
            return row['label']
    return None  # If no match is found, you might want to handle this case

# Dictionary to store matched labels with PDF filenames
matched_labels = {}

# Iterate over PDF files and match them to labels
for pdf_file in pdf_files:
    if pdf_file.endswith('.pdf'):
        matched_label = match_pdf_to_label(pdf_file, labels_df)
        matched_labels[pdf_file] = matched_label


# ----------- Start Text Preprocessing ------------------

def preprocess_text(text):
    # Tokenize and convert to lower case, initial tokenization for removing stop words, punctuation, and normalizing the text.
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    stripped_tokens = [w.translate(table) for w in tokens]
    
    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped_tokens if word.isalpha()]
    
    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    return words

def clean_transcript(text):
    # Remove URLs in the footer
    text = re.sub(r'https?://\S+', '', text)
    
    # Normalize text to lower case
    text = text.lower()

    # Remove date and time patterns, e.g., "2/7/24, 8:42 AM"
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APM]{2}', '', text)

    # Remove any text that looks like a website, e.g., "seeking alpha"
    text = re.sub(r'seeking alpha', '', text, flags=re.IGNORECASE)

    # Remove headers that might be repeated, e.g., earnings call transcript titles
    text = re.sub(r'earnings call transcript', '', text)

    # Remove numeric patterns that don't add value, e.g., "Q4 2023"
    text = re.sub(r'q[1-4] \d{4}', '', text)

    # Remove specific phrases or patterns that don't contribute to the analysis
    phrases_to_remove = [
        r'\bfeb\. \d{2}, 2024\b',
        r'\b\d{1,2}:\d{2} [apm]{2} et\b',
        r'\btranscripts\b',
        r'\bfollowers\b',
        # Add more patterns as needed
    ]
    for phrase in phrases_to_remove:
        text = re.sub(phrase, '', text)

    # Remove any leading or trailing whitespace
    text = text.strip()

    return text

# Preprocess texts and store them
preprocessed_texts = {}
for pdf_file in pdf_files:
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        # Preprocess the extracted text
        clean_text = clean_transcript(text)
        processed_text = preprocess_text(clean_text)
        # Store the processed text with its matched label
        matched_label = matched_labels.get(pdf_file)
        preprocessed_texts[pdf_file] = (processed_text, matched_label)
        
# ----------- End Text Preprocessing ------------------

# ----------- Start BERT Tokenization ------------------
        
# Initialize the BERT model for sequence classification with 3 classes
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # The number of output, positive, negative and neutral.
)

# --- Step 1: Tokenize Preprocessed Texts for BERT ---
# BERT requires text to be tokenized in a way that matches its pre-trained vocabulary, including special tokens and padding. 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all preprocessed texts
tokenized_texts = {}
for file, (preprocessed_text, label) in preprocessed_texts.items():
    tokenized_text = tokenizer(' '.join(preprocessed_text), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenized_texts[file] = (tokenized_text, label)
 
# --- End Step 1 ---

# --- Step 2: Prepare Data Loaders for Training ---

class TextDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.items = [(text['input_ids'].squeeze(0), text['attention_mask'].squeeze(0), tensor(label).long()) for _, (text, label) in tokenized_texts.items()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.items[idx]
        return input_ids, attention_mask, label

# Create dataset and data loader
dataset = TextDataset(tokenized_texts)
data_loader = DataLoader(dataset, batch_size=8)  # Adjust batch size as needed        
# --- End Step 2 ---


# --- Step 3: Model Training ---
# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 20
model.train()

logging.info("Starting model training")

for epoch in range(epochs):  # Define epochs
    logging.info(f'Starting epoch {epoch+1}/{epochs}')
    epoch_loss = 0.0

    for step, batch in enumerate(data_loader):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

#         if step % 10 == 0:  # Log every 10 steps.
#             logging.info(f'Epoch: {epoch+1}, Step: {step}, Loss: {loss.item()}')

#     logging.info(f'Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(data_loader)}')

# logging.info("Model training completed")
# --- End Step 3 ---
        
# --- Step 4: Save the Model---

# Save the entire model directly (including configurations)
# This approach saves the entire model, which can then be easily loaded with from_pretrained
model_directory = r"C:\Users\SZ4291\OneDrive - Zebra Technologies\Documents\Shawn\Stanford\CS229\Project"
print(f"Saving model to {model_directory}")
model.save_pretrained(model_directory)
tokenizer.save_pretrained(model_directory)
print("Model saved successfully.")
# --- End Step 4 ---
       
# ----------- End Model Training ------------------

