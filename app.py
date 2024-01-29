# Import necessary libraries
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import  Image
from rake_nltk import Rake
import spacy
import spacy_streamlit
from collections import Counter
import en_core_web_sm
from nltk.tokenize import sent_tokenize
import nltk
import streamlit as st
from summarizer import Summarizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# check

# Warnings ignore 
warnings.filterwarnings(action='ignore')
# st.set_option('deprecation.showfileUploaderEncoding', False)
showfileUploaderEncoding = False
st.set_option('deprecation.showPyplotGlobalUse', False)

# Import the custom modules 
import spam_filter as sf
import text_analysis as nlp
# import text_summarize as ts

# Describing the Web Application 

# Title of the application 
st.title("Text Summarizer and it's Features\n", )
st.subheader("")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Home",
 "Email Spam Classifier", 
 "Keyword Sentiment Analysis", 
 "Word Cloud", 
 "N-Gram Analysis", 
 "Parts of Speech Analysis", 
 "Named Entity Recognition",
 "Text Summarizer"])

# st.set_option('deprecation.showfileUploaderEncoding', False)
showfileUploaderEncoding = False

if option == 'Home':
	st.write(
			"""
				## Project Description
				 Our project focuses on simplifying and summarizing written content. Using various tools and techniques, we've developed a system capable of condensing lengthy texts into shorter, more understandable versions. To achieve this, we employ different methods like email spam detection, which helps in filtering out unwanted or irrelevant information. Additionally, our system analyzes keywords to understand the sentiment or emotions expressed in the text. We visualize this data using word clouds that highlight the most prominent terms. Moreover, we utilize n-gram analysis, a method for examining sequences of words, enabling us to understand patterns within the text. Furthermore, our system employs Part-of-Speech (POS) tagging to identify and categorize words based on their grammatical roles, and Named Entity Recognition (NER) to spot and classify named entities like names of people, organizations, or locations within the text. Through these tools, we aim to make written information more digestible and accessible for users.
			"""
		)

# Word Cloud Feature
elif option == "Word Cloud":

	st.header("Generate Word Cloud")
	st.subheader("Generate a word cloud from text containing the most popular words in the text.")

	# Ask for text or text file
	st.header('Enter text or upload file')
	text = st.text_area('Type Something', height=400)

	# Upload mask image 
	mask = st.file_uploader('Use Image Mask', type = ['jpg'])

	# Add a button feature
	if st.button("Generate Wordcloud"):

		# Generate word cloud 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()

# N-Gram Analysis Option 
elif option == "N-Gram Analysis":
	
	st.header("N-Gram Analysis")
	st.subheader("This section displays the most commonly occuring N-Grams in your Data")

	# Ask for text or text file
	st.header('Enter text below')
	text = st.text_area('Type Something', height=400)

	# Parameters
	n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
	topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

	# Add a button 
	if st.button("Generate N-Gram Plot"): 
		# Plot the ngrams
		nlp.plot_ngrams(text, n=n, topk=topk)
		st.pyplot()

# Spam Filtering Option
elif option == "Email Spam Classifier":

	st.header("Enter the email you want to send")

	# Add space for Subject 
	subject = st.text_input("Write the subject of the email", ' ')

	# Add space for email text 
	message = st.text_area("Add email Text Here", ' ')

	# Add button to check for spam 
	if st.button("Check"):

		# Create input 
		model_input = subject + ' ' + message
		
		# Process the data 
		model_input = sf.clean_text_spam(model_input)

		# Vectorize the inputs 
		vectorizer = joblib.load('Models/count_vectorizer_spam.sav')
		vec_inputs = vectorizer.transform(model_input)
		
		# Load the model
		spam_model = joblib.load('Models/spam_model.sav')

		# Make the prediction 
		if spam_model.predict(vec_inputs):
			st.write("This message is **Spam**")
		else:
			st.write("This message is **Not Spam**")
		
# POS Tagging Option 
elif option == "Parts of Speech Analysis":
	st.header("Enter the statement that you want to analyze")

	text_input = st.text_input("Enter sentence", '')

	if st.button("Show POS Tags"):
		tags = nlp.pos_tagger(text_input)
		st.markdown("The POS Tags for this sentence are: ")
		st.markdown(tags, unsafe_allow_html=True)


		st.markdown("### Penn-Treebank Tagset")
		st.markdown("The tags can be referenced from here:")

		# Show image
		display_pos = Image.open('images/Penn_Treebank.png')
		display_pos = np.array(display_pos)
		st.image(display_pos)

# Named Entity Recognition 
elif option == "Named Entity Recognition":
	st.header("Enter the statement that you want to analyze")

	st.markdown("**Random Sentence:** A Few Good Men is a 1992 American legal drama film set in Boston directed by Rob Reiner and starring Tom Cruise, Jack Nicholson, and Demi Moore. The film revolves around the court-martial of two U.S. Marines charged with the murder of a fellow Marine and the tribulations of their lawyers as they prepare a case to defend their clients.")
	text_input = st.text_area("Enter sentence")

	ner = en_core_web_sm.load()
	doc = ner(str(text_input))

	# Display 
	spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)



# Keyword Sentiment Analysis
elif option == "Keyword Sentiment Analysis":

	st.header("Sentiment Analysis Tool")
	st.subheader("Enter the statement that you want to analyze")

	text_input = st.text_area("Enter sentence", height=50)

	# Model Selection 
	model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])

	if st.button("Predict"):
		
		# Load the model 
		if model_select == "SVC":
			sentiment_model = joblib.load('Models/SVC_sentiment_model.sav')
		elif model_select == "Logistic Regression":
			sentiment_model = joblib.load('Models/LR_sentiment_model.sav')
		elif model_select == "Naive Bayes":
			sentiment_model = joblib.load('Models/NB_sentiment_model.sav')
		
		# Vectorize the inputs 
		vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')
		vec_inputs = vectorizer.transform([text_input])

		# Keyword extraction 
		r = Rake(language='english')
		r.extract_keywords_from_text(text_input)
		
		# Get the important phrases
		phrases = r.get_ranked_phrases()

		# Make the prediction 
		if sentiment_model.predict(vec_inputs):
			st.success("This statement is **Positve**")
		else:
			st.error("This statement is **Negative**")

		# Display the important phrases
		st.write("These are the **keywords** causing the above sentiment:")
		for i, p in enumerate(phrases):
			st.write(i+1, p)


# Text Summarizer
elif option == "Text Summarizer":
	def main():
		st.title("Text Summarizer App")

		# User input
		text_input = st.text_area("Enter text to summarize:")

		if st.button("Summarize"):
			if not text_input:
				st.warning("Please enter some text to summarize.")
			else:
				# Summarization
				summarizer = Summarizer()
				summary = summarizer(text_input)

				# Display the result
				st.subheader("Original Text:")
				st.write(text_input)

				st.subheader("Summary:")
				st.write(summary)
	if __name__ == "__main__":
		main()



