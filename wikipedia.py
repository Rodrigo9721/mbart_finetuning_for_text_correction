import requests
from bs4 import BeautifulSoup
import random
import string
from tqdm import tqdm
import re
import pandas as pd

def clean_text(text):
    # Remove newline characters
    text = text.replace("\n", " ")

    # Remove unicode characters like \u200b
    text = re.sub(r"\\u[0-9A-Fa-f]{4}", "", text)

    # Remove non-latin characters
    text = re.sub(r"[^\x00-\x7FÀ-ÖØ-öø-ÿ]+", "", text)

    return text

def get_random_article():
    random_url = "https://es.wikipedia.org/wiki/Especial:Aleatoria"
    response = requests.get(random_url)
    soup = BeautifulSoup(response.text, "html.parser")

    article_text = ""
    for paragraph in soup.find_all("p"):
        article_text += paragraph.text
    article_text = clean_text(article_text)

    return article_text

def insert_error(text, error_rate=0.1):
    def replace_similar_letters(c):
        similar_letters = {
            "b": "v",
            "v": "b",
            "c": "z",
            "z": "c",
            "s": "z",
            "y": "i",
            "q": "k",
            "k": "q",
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "a": "á",
            "e": "é",
            "i": "í",
            "o": "ó",
            "u": "ú"
        }
        return similar_letters.get(c.lower(), c)

    def duplicate_letter(c):
        return c * 2

    def omit_letter(c):
        return ""

    def transpose_letters(s, i):
        if i + 1 < len(s):
            return s[:i] + [s[i+1]] + [s[i]] + s[i+2:]
        return s

    text = list(text)
    num_errors = int(len(text) * error_rate)
    error_operations = [replace_similar_letters, duplicate_letter, omit_letter, transpose_letters]

    for _ in range(num_errors):
        error_pos = random.randint(0, len(text) - 1)
        error_operation = random.choice(error_operations)
        
        if error_operation == transpose_letters:
            text = error_operation(text, error_pos)
        else:
            text[error_pos] = error_operation(text[error_pos])

    return "".join(text)

import time

# Download and process articles
articles = []
num_articles = 30000
for i in tqdm(range(num_articles)):
    try:
        articles.append(get_random_article())
    except:
        time.sleep(10)
        continue



# Generate pairs of incorrect and correct sentences
sentence_pairs = []
for article in articles:
    sentences = article.split(".")
    for sentence in sentences:
        incorrect_sentence = insert_error(sentence)
        if len(sentence) > 50:
            sentence_pairs.append({"incorrect": incorrect_sentence, "correct": sentence})
        else:
            continue



import json

with open('data.json', 'w', encoding="utf-8") as fp:
    json.dump(sentence_pairs, fp, ensure_ascii=False, indent=4)