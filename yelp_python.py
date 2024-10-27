import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


#this is the set of all business ids we have. The dataset is too big so
#we just store all businesses that we want to tag extract here
business = set()

#read and parse file for certain cities only
def read_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                value = json.loads(line)
                if value['business_id'] in business:
                    data.append(value)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
    return data

#currently set to just philadelphia
def preload(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            try:
                value = json.loads(line)
                if(value['city'] == 'Philadelphia'):
                    business.add(value['business_id'])
            except json.JSONDecodeError as e:
                print(line)
                print(f"Error parsing JSON: {e}")

#load the bert model from tensorflow hub
def load_model():
    # Load the BERT preprocessor and encoder
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    
    preprocess = hub.load(preprocess_url)
    encoder = hub.load(encoder_url)
    
    return preprocess, encoder


#Use NLTK to extact keywords from the review text
def extract_keywords_from_review(text, stop_words):
    # Tokenize the text into sentences and words using nltk
    sentences = sent_tokenize(text.lower())
    words = [word_tokenize(sentence) for sentence in sentences]
    
    # Flatten word list and remove stopwords and punctuation
    all_words = []
    for sentence in words:
        for word in sentence:
            #not a stopword (so it can be a potential keyword)
            if word.isalnum() and word not in stop_words:
                all_words.append(word)

    # Get POS tags
    pos_tags = pos_tag(all_words)
    
    # Extract nouns, adjectives, and verb phrases. I've deemed this to be
    #the best way to get important words for now, this can be changed
    #in the future if needed
    important_words = []
    prev_word = None
    for word, tag in pos_tags:
        # Include nouns and adjectives
        if tag.startswith(('NN', 'JJ')):
            important_words.append(word)
            # This helps create compound phrases (e.g., "great food", "friendly staff")
            if prev_word and prev_word[1].startswith('JJ') and tag.startswith('NN'):
                important_words.append(f"{prev_word[0]} {word}")
        prev_word = (word, tag)
    
    return list(set(important_words))

#here we use the model to get attention weights allowing us to weight tags and
#return most relevant ones if needed. I have very surface level knowledge
#of BERT so some changes may be needed here
def get_bert_attention_scores(text, preprocess, encoder):
    # Preprocess the text
    text_preprocessed = preprocess([text])
    
    # Get BERT outputs including attention weights
    outputs = encoder(text_preprocessed)
    
    # Get attention weights from the last layer
    attention = outputs['encoder_outputs'][-1]  # Shape: [batch_size, seq_length, hidden_size]
    
    # Calculate attention magnitude for each token
    attention_magnitude = tf.reduce_mean(tf.abs(attention), axis=-1)
    
    return attention_magnitude.numpy()[0]  # Remove batch dimension

def extract_tags(review, preprocess, encoder, max_tags=5):
    text = review['text']
    stop_words = set(stopwords.words('english'))
    
    # Extract initial keywords based on POS tagging
    keywords = extract_keywords_from_review(text, stop_words)
    
    # Get BERT attention scores
    attention_scores = get_bert_attention_scores(text, preprocess, encoder)
    
    # Create a mapping of words to their attention scores
    word_scores = defaultdict(float)
    words = word_tokenize(text.lower())
    
    # Normalize attention scores to the word level
    for word, score in zip(words, attention_scores[:len(words)]):
        if word in keywords:
            word_scores[word] += score
    
    # Sort keywords by attention scores
    scored_keywords = [(word, word_scores.get(word, 0)) for word in keywords]
    scored_keywords.sort(key=lambda x: x[1], reverse=True)
    
    # Select top tags
    tags = [word for word, _ in scored_keywords[:max_tags]]
    
    return {
        'review': text,
        'tags': tags
    }

def main():
    preload_path = 'yelp_dataset/yelp_academic_dataset_business.json'
    file_path = 'yelp_dataset/yelp_academic_dataset_review.json'

    print("Loading business data...")
    preload(preload_path)
    
    print("Loading reviews...")
    reviews = read_file(file_path)

    temp = reviews[:10]
    
    print("Loading BERT model...")
    preprocess, encoder = load_model()

    print(f"Processing {len(temp)} reviews...")
    results = []
    for i, review in enumerate(temp):
        if i % 100 == 0:
            print(f"Processing review {i+1}/{len(temp)}")
        result = extract_tags(review, preprocess, encoder)
        results.append(result)

    # Print sample results
    print("\nSample results:")
    for i in range(min(3, len(results))):
        print(f"\nReview {i+1}:")
        print(f"Text: {results[i]['review'][:200]}...")
        print(f"Tags: {results[i]['tags']}")


if __name__ == "__main__":
    main()
