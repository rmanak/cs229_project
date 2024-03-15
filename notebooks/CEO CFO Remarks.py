import re
import fitz  # PyMuPDF
from fpdf import FPDF
import spacy
import json
import sys
import os


# --- Extract CEO and CFO Names and Remarks ---

def extract_names_from_text(text):
    # Define patterns for CEO and CFO titles more flexibly
    participants_section = re.search(r"Company Participants(.*?)Conference Call Participants", text, re.DOTALL)
    if not participants_section:
        return None, None

    participants_text = participants_section.group(1)
    ceo_name = None
    cfo_name = None

    # Look for names followed by the role
    for line in participants_text.split('\n'):
        if 'Chief Executive Officer' in line or 'CEO' in line:
            ceo_name = line.split('-')[0].strip()
        elif 'Chief Financial Officer' in line or 'CFO' in line:
            cfo_name = line.split('-')[0].strip()

    return ceo_name, cfo_name

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_speaker_remarks(transcript_text, speaker_name, max_words=258):
    lines = transcript_text.split('\n')
    remarks = []
    found_speaker = False

    for i, line in enumerate(lines):
        # Check for the speaker line
        if speaker_name in line and len(line.strip().split()) <= 3:
            found_speaker = True
            continue

        # Start extracting remarks after finding speaker name and encountering an empty line
        if found_speaker:
            # Skip the next empty line and start extracting from the line after
            start_line = i
            for j in range(start_line, len(lines)):
                # Stop extracting if maximum words reached
                if len(remarks) >= max_words:
                    break
                remarks.extend(lines[j].split())

            # Once remarks are collected, break from the loop
            break

    return ' '.join(remarks[:max_words])

def remove_greeting_sentences(text):
    greeting_keywords = ['thanks', 'thank you', 'everyone', 'hello', 'good morning', 'good afternoon', 'good evening']
    pattern = re.compile(r'\b(?:' + '|'.join(greeting_keywords) + r')\b', re.IGNORECASE)

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    filtered_sentences = [sentence for sentence in sentences if not pattern.search(sentence)]
    
    return ' '.join(filtered_sentences)

# Implement

pdf_dir = r'C:\Users\SZ4291\OneDrive - Zebra Technologies\Documents\Shawn\Stanford\CS229\Project\Training Data\Transcripts'
pdf_files = os.listdir(pdf_dir)

for pdf_file in pdf_files:
     if pdf_file.endswith('.pdf'):
        # Construct the full path to the current PDF file
        pdf_path = os.path.join(pdf_dir, pdf_file)

        text = extract_text_from_pdf(pdf_path)
        ceo_name, cfo_name = extract_names_from_text(text)

        # Assuming you have already extracted the CEO and CFO names from the PDF text and they are stored in `ceo_name` and `cfo_name` variables.
        if ceo_name is not None:
            ceo_remarks = extract_speaker_remarks(text, ceo_name, 258)
        else:
            ceo_remarks = "CEO remarks not available."

        if cfo_name is not None:
            cfo_remarks = extract_speaker_remarks(text, cfo_name, 258)
        else:
            cfo_remarks = "CFO remarks not available."

        ceo_remarks = remove_greeting_sentences(ceo_remarks)
        cfo_remarks = remove_greeting_sentences(cfo_remarks)

        # Extract the directory and the PDF file name
        pdf_dir, pdf_file_name = os.path.split(pdf_path)

        # Replace the PDF extension with .txt for the new file name
        txt_file_name = pdf_file_name.replace('.pdf', '.txt')

        # Combine the directory and new file name to get the full path to the new .txt file
        txt_file_path = os.path.join(pdf_dir, txt_file_name)

        # Write the remarks to the text file
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{ceo_remarks}\n")
            f.write(f"\n{cfo_remarks}")