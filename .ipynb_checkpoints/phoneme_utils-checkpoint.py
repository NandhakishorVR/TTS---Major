import csv
import re
import inflect
import nltk
from nltk.corpus import cmudict
from torchtext.vocab import build_vocab_from_iterator

# Load CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Number converter
num_engine = inflect.engine()

# Load custom phoneme dataset from CSV
def load_phoneme_dataset(file_path="phoneme_dataset.csv"):
    phoneme_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header if exists
            for row in reader:
                if len(row) == 2:
                    word, phoneme = row
                    phoneme_dict[word.lower()] = phoneme
    except FileNotFoundError:
        print("[WARNING] Custom phoneme dataset not found.")
    return phoneme_dict

custom_phoneme_dict = load_phoneme_dataset()

# Expanded abbreviations
ABBREVIATIONS = {
    "dr.": "doctor",
    "mr.": "mister",
    "mrs.": "misses",
    "ph.d.": "doctor of philosophy",
    "u.s.": "united states",
    "it's": "it is",
    "you're": "you are",
    "they're": "they are",
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
}

# Function to clean text
def clean_text(text):
    text = text.lower()
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    text = re.sub(r"\b\d+\b", lambda x: num_engine.number_to_words(x.group()), text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Function to convert text to phonemes
def text_to_phonemes(text):
    words = text.split()
    phoneme_output = []
    for word in words:
        if word in custom_phoneme_dict:
            phoneme_output.append(custom_phoneme_dict[word])
        elif word in cmu_dict:
            phoneme_output.append(" ".join(cmu_dict[word][0]))
        else:
            phoneme_output.append(" ".join(list(word)))
    return " ".join(phoneme_output)

# Create vocabulary for phoneme-to-index mapping
def phoneme_vocab_generator():
    for word in cmu_dict:
        for phoneme_seq in cmu_dict[word]:
            yield phoneme_seq

phoneme_vocab = build_vocab_from_iterator(phoneme_vocab_generator(), specials=["<unk>", "<pad>"])
phoneme_vocab.set_default_index(phoneme_vocab["<unk>"])

# Function to convert phonemes to index sequence
def phonemes_to_index(phoneme_str):
    phonemes = phoneme_str.strip().split()
    return [phoneme_vocab[phoneme] for phoneme in phonemes]

# Full pipeline function
def process_text_to_phonemes(text):
    cleaned_text = clean_text(text)
    phonemes = text_to_phonemes(cleaned_text)
    phoneme_indices = phonemes_to_index(phonemes)
    return phonemes, phoneme_indices

# Example usage
input_text = "hello how are you"
phoneme_result, phoneme_indices = process_text_to_phonemes(input_text)

print("Original Text:", input_text)
print("Cleaned Text:", clean_text(input_text))
print("Phonemes:", phoneme_result)
#print("Phoneme Indices:", phoneme_indices)