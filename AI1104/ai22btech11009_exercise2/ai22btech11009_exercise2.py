import re
import numpy as np
import pandas as pd

# Read the reviews and store them in variables

with open('Doc1.txt', 'r') as file:
    doc1 = file.read()

with open('Doc2.txt', 'r') as file:
    doc2 = file.read()

with open('Doc3.txt', 'r') as file:
    doc3 = file.read()

with open('Doc4.txt', 'r') as file:
    doc4 = file.read()

#Tokenization
#define a function to tokenise
def tokenize(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Split into words
    tokens = text.split()
    return tokens

doc1_tokens = tokenize(doc1)
doc2_tokens = tokenize(doc2)
doc3_tokens = tokenize(doc3)
doc4_tokens = tokenize(doc4)

#Vocabulary building

vocabulary = set(doc1_tokens + doc2_tokens + doc3_tokens + doc4_tokens)
num_words = len(vocabulary)#taking no of words in vocabulary
print("Total number of words in the vocabulary:", num_words) #printing no.of words in vocabulary

#Create frequency tables by defining a function

def create_frequency_table(tokens):
    frequency_table = {} #taking empty initially
    for token in tokens:
        if token in frequency_table:
            frequency_table[token] += 1
        else:
            frequency_table[token] = 1
    return frequency_table

doc1_freq_table = create_frequency_table(doc1_tokens)
doc2_freq_table = create_frequency_table(doc2_tokens)
doc3_freq_table = create_frequency_table(doc3_tokens)
doc4_freq_table = create_frequency_table(doc4_tokens)

#Combine frequency tables into a dataframe

data = {
    'Doc1': doc1_freq_table,
    'Doc2': doc2_freq_table,
    'Doc3': doc3_freq_table,
    'Doc4': doc4_freq_table
}

df = pd.DataFrame(data).fillna(0).astype(int)
print(df)

#Similarity Calculation

# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Create a square matrix to store the cosine similarity between reviews
num_reviews = df.shape[1]
similarity_matrix = np.zeros((num_reviews, num_reviews))

# Calculate cosine similarity between each pair of reviews
for i in range(num_reviews):
    for j in range(i+1, num_reviews):  # Only iterate over upper triangular matrix
        vec1 = df.iloc[:, i].values
        vec2 = df.iloc[:, j].values
        similarity = cosine_similarity(vec1, vec2)
        similarity_matrix[i, j] = similarity

# Set the lower triangular matrix values to negative infinity to exclude them from sorting
similarity_matrix[np.tril_indices(num_reviews)] = float('-inf')

# Find the top 3 review pairs with the highest similarity
top_indices = np.unravel_index(np.argsort(similarity_matrix, axis=None)[-3:], similarity_matrix.shape)
top_scores = similarity_matrix[top_indices]

# Sort the pairs in descending order of similarity scores
sorted_indices = np.argsort(top_scores)[::-1]
sorted_scores = top_scores[sorted_indices]
sorted_pairs = zip(top_indices[0][sorted_indices], top_indices[1][sorted_indices])

# Print the top 3 review pairs with the highest similarity
for i, (doc1_index, doc2_index) in enumerate(sorted_pairs):
    similarity_score = sorted_scores[i]
    review1 = "Doc" + str(doc1_index + 1)
    review2 = "Doc" + str(doc2_index + 1)
    print(f"Top {i+1} Review Pair:")
    print(f"{review1} vs {review2}")
    print("Similarity Score:", similarity_score)
    print()
