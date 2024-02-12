import streamlit as st
import pickle
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')

def preprocessing(msg):
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)

    y=[]
    for i in msg:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                ps = PorterStemmer()
                y.append(ps.stem(i))
    return " ".join(y)

tfidf_vectorize = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input = st.text_area("Enter The Message:")

if st.button("Predict"):

    processed_input = preprocessing(input)

    vector_input = tfidf_vectorize.transform([processed_input])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')



