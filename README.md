Next Word Prediction with LSTM (Shakespeare's Hamlet)
This project implements a text generation model using a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) to predict the next word in a sequence based on the full text of Shakespeare's Hamlet.
The model is trained to capture the complex patterns, syntax, and vocabulary of the classic text, allowing it to generate new lines in a similar style.

Key Features:
        LSTM Architecture:
                The model uses a $\text{Sequential}$ structure built with Keras. It includes an $\text{Embedding}$ layer to vectorize words, a single $\text{LSTM}$ layer to capture long-range dependencies, a $\text{Dropout}$ layer for regularization, and a final                            $\text{Dense}$ layer with $\text{softmax}$ activation to output the probability distribution over the entire vocabulary.
        Sequence-to-One Prediction: 
                he architecture is specifically configured for a sequence-to-one task. It takes a sequence of 13 preceding words and predicts the single 14th word.
        Text Preprocessing: 
                The workflow includes robust text preparation, utilizing the Keras $\text{Tokenizer}$ for vocabulary mapping and $\text{pad\_sequences}$ to ensure all input sequences have a consistent length.
        Deployment Ready:
                The project is prepared for emonstration using Streamlit, with an included app.py script for easy deployment and interaction.
        
Getting Started: To run this project and the Streamlit demo locally, follow these steps: Clone the Repository and navigate to the project directory.
Install Dependencies:It is recommended to use a virtual environment before running the following command:
                        
                        pip install -r requirements.txt

Run the App: After installing the dependencies then run this command in same python environment to show the webpage. 
                        
                        streamlit run app.py

Also, you can visit this model prediction in online. just visit this url https://predictnextworde.streamlit.app/.
