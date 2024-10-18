import os
import re
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np 

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

def calculate_cosine_similarity(new_text, existing_texts):
    preprocessed_new_text = preprocess_text(new_text)
    preprocessed_existing_texts = [preprocess_text(text) for text in existing_texts]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_new_text] + preprocessed_existing_texts)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
    return similarities

def check_similarity(new_text_paths, existing_data_paths, threshold_upper=0.9, threshold_lower=0.3):
    destination_folder = "study_material"
    existing_texts = []
    for path in existing_data_paths:
        with open(path, 'r', encoding='utf-8') as file:
            existing_texts.append(file.read())
    for new_text_path in new_text_paths:
        with open(new_text_path, 'r', encoding='utf-8') as file:
            new_text = file.read()
        similarities = calculate_cosine_similarity(new_text, existing_texts)
        max_similarity = max(similarities)
        if max_similarity >= threshold_upper:
            print(f"No need to add {new_text_path} to the main data folder")
            print(f"Deleted {new_text_path} ")
            os.remove(new_text_path)
        elif max_similarity <= threshold_lower:
            print(f"No need to add {new_text_path} to the main data folder")
            os.remove(new_text_path)
            print(f"Deleted {new_text_path} ")
        else:
            print(f"Adding {new_text_path} to the main data folder")
            shutil.move(new_text_path, destination_folder)

def get_text_files(directory):
    text_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            text_files.append(os.path.join(directory, filename))
    return text_files

def main():
    existing_data_directory = 'study_material'
    existing_data_paths = get_text_files(existing_data_directory)

    new_data_directory = 'new_material_text'
    new_data_paths = get_text_files(new_data_directory)

    check_similarity(new_data_paths, existing_data_paths)

if __name__ == "__main__":
    main()
