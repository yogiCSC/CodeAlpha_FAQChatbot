import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Questions and Answers
questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?",
    "What is Streamlit?",
    "Who developed Python?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Python is a programming language.",
    "Machine Learning is a subset of AI.",
    "Streamlit is a Python framework used to build web apps.",
    "Python was developed by Guido van Rossum."
]

# Title
st.title("🤖 FAQ Chatbot")

# User Input
user_input = st.text_input("Ask Your Question")

ask = st.button("Ask")

if ask and user_input:

    # Convert text into vectors
    all_questions = questions + [user_input]

    vectorizer = CountVectorizer().fit_transform(all_questions)

    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])

    index = similarity.argmax()

    st.success("Answer:")
    st.write(answers[index])