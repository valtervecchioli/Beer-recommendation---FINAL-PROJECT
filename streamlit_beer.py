import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle

# GB = pickle.load(open("beer_model.pkl","rb"))
# standardizer = pickle.load(open("standardizer.pkl","rb"))

nltk.download("punkt")
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenizer(text):
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens if token.isalpha()]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rye&display=swap');

    .stApp {{
        background: url("https://t4.ftcdn.net/jpg/04/34/38/75/360_F_434387570_IWkGdVNeGyeTneY2ISJjSrvNGSCQm2Dv.jpg") no-repeat center center fixed;
        background-size: cover;
    }}
    .big-font {{
        font-size: 50px !important;
        font-weight: bold;
    }}
    h1, .subtitle, .how-it-works {{
        font-family: 'Rye', cursive;
    }}
    h1 {{
        color: brown;
    }}
    .subtitle {{
        color: black;
        font-size: 30px;
    }}
    .how-it-works {{
        color: black;
        font-size: 35px;
    }}
    </style>
    """,
    unsafe_allow_html=True)

beer = pd.read_csv('beerlo.csv')
features = beer.drop(columns=["rating"])
target = beer["rating"]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

standardizer = StandardScaler()
standardizer.fit(x_train)
x_train_st = standardizer.transform(x_train)
x_test_st = standardizer.transform(x_test)

GB = GradientBoostingRegressor(n_estimators=100, max_depth=20)
GB.fit(x_train_st, y_train)

st.title("BEER RANKING BOT🍻")
st.markdown('<p class="subtitle">Classify your beer by keywords on reviews!</p>', unsafe_allow_html=True)
st.write("---")

if st.checkbox("Show raw data")==True:
    st.text_input("Submit your name for access log")
    st.write(beer)
st.write("---")
st.markdown('<p class="how-it-works">How it works:</p>', unsafe_allow_html=True)
st.markdown("Paste one or more reviews on this site, and find out how a beer is ranked:")


input_text = st.text_area("Paste your review here (A number of at least 20 reviews is recommended for a highly performative result):")


if st.button("Submit"):
    token_rev = tokenizer(input_text)
    lem_rev = lemmatize(token_rev)

    keyword_counts = {col: 0 for col in features.columns}

    for word in lem_rev:
        if word in keyword_counts:
            keyword_counts[word] += 1

    new_row = pd.DataFrame([keyword_counts])
    beer_with_new_row = pd.concat([beer, new_row], ignore_index=True)   

    beer_with_new_row.drop(columns=["rating"], inplace=True)

    features_with_new_st = standardizer.transform(beer_with_new_row)

    new_row_st = features_with_new_st[-1].reshape(1, -1)

    pred = GB.predict(new_row_st)

    st.write("Keyword counts:", keyword_counts)
    st.markdown("**Your beer has a rating of:**")
    st.markdown(f"**{pred}**")
    st.markdown("**and it's classified as:**")
    if pred <= 1:
        st.markdown("**POOR**")
    elif pred > 1 and pred <= 2:
        st.markdown("**MEDIOCRE**")
    elif pred > 2 and pred <= 3:
        st.markdown("**AVERAGE**")
    elif pred > 3 and pred <= 4:
        st.markdown("**GOOD**")
    else:
        st.markdown("**EXCELLENT**")