import logging
logging.getLogger('tensorflow').disabled = True
import os
import time
import numpy as np
import streamlit as st
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 

path_to_file = tf.keras.utils.get_file('Les_fleurs_du_mal_Baudelaire', 'https://raw.githubusercontent.com/AmelNozieres/NLP_Generate_poems/master/Les_fleurs_du_mal_Baudelaire')

tokenizer = Tokenizer()
data = open(path_to_file, 'rb').read().decode(encoding='utf-8')
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)



st.title("📜 Baudelaire poem generation")



model = load_model("baudelaire.hdf5", compile = False)



next_words = 50
def generate_text(seed_text):
    generated_text = ""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            break
        if ord(output_word[0]) != 13:
            seed_text = seed_text + " " + output_word
            generated_text += output_word + " "	
    return generated_text.strip()
    
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def main():
    seed_text = st.text_input("Start the poem with a few words ...", "Je suis")
    if seed_text:
        result = generate_text(seed_text)
        st.write(seed_text  + " " + str(result).replace("»", ""))

    st.markdown("""---""")

    intro_markdown = read_markdown_file("text.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

if __name__ == '__main__':
    main()


