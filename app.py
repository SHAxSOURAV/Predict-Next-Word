import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
        
    token_list = pad_sequences([token_list], maxlen = max_sequence_len-1, padding = 'pre')
    
    # Get the probabilities from the model
    predicted_probs = model.predict(token_list, verbose = 0)
    
    #Get the single integer index [0]
    predicted_word_index = np.argmax(predicted_probs, axis = 1)[0] 

    # Loop through the word_index dictionary to find the matching word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title('Next word Prediction with LSTM')
input_text = st.text_input("Enter the sequence of words", 'To be or not to')
if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')