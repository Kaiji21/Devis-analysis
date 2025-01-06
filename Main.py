import streamlit as st
import pandas as pd
import phonenumbers
from phonenumbers import geocoder
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer

# Title and Project Objective
st.title("Tableau de Bord d'Analyse de Données")
st.write("### Objectif du projet")
st.write("L'objectif de ce projet est d'analyser les données extraites d'un fichier CSV, de nettoyer les messages et d'extraire des mots-clés utiles. Ces mots-clés peuvent ensuite être utilisés dans les processus de publicité sur des plateformes comme Facebook, Google et TikTok.")

# File uploader widget for CSV file
uploaded_file = st.file_uploader("Télécharger le fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Displaying the loaded data
    st.write("### Données Chargées")
    st.dataframe(data)


    st.write("### Vérification de la qualité des données")
    # Step 1: Check missing values and duplicates
    col1, col2, col3 = st.columns([2, 6, 2])
    with col2:
        missing_values = data.isnull().sum()
        duplicates = data.duplicated().sum()
        st.write("Valeurs manquantes par colonne :")
        st.write(missing_values)
        st.write("Nombre de doublons :", duplicates)

    

    # Step 2: Country extraction from phone numbers
    def get_country(phone):
        try:
            parsed_number = phonenumbers.parse(phone, None)
            country = geocoder.region_code_for_number(parsed_number)
            return country
        except phonenumbers.phonenumberutil.NumberParseException:
            return None
    
    st.write("### Pays Extrêmement")
    col1, col2, col3 = st.columns([2, 6, 2])
    with col2:
        data["Téléphone"] = data["Téléphone"].str.strip("'")
        data['Pays'] = data['Téléphone'].apply(get_country)
        st.dataframe(data[['Téléphone', 'Pays']].head(10),width=600)
 

    # Country distribution
    unique_phone_country = data['Pays'].value_counts()
    st.write("### Répartition des Numéros de Téléphone par Pays")
    st.bar_chart(unique_phone_country)

    # Step 3: Extract email domains
    data['Email_Domain'] = data['E-mail'].str.split('@').str[1]
    unique_domains = data['Email_Domain'].value_counts()
    st.write("### Principaux Domaines d'Email")
    st.bar_chart(unique_domains.head(10))

    # Step 4: Clean messages and generate WordCloud
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))
    custom_stopwords = ['bonjour', 'ça', 'va', 'merci', 's\'il', 'vous', 'plaît', 'svp', 'merci', 'salut', 'alors', 'test', 'avoir', 'cordialmement','madam','monsieur','savoire']
    stop_words.update(custom_stopwords)
    stop_words = list(stop_words)

    def clean_message(message):
        if pd.isnull(message):
            return ""
        message = message.lower()
        message = ''.join(char for char in message if char not in string.punctuation)
        message = ' '.join(word for word in message.split() if word not in stop_words and len(word) > 2)
        return message

    data['Cleaned_Message'] = data['Commentaire ou message'].apply(clean_message)
    all_messages = ' '.join(data['Cleaned_Message'])

    st.write("### Nuage de Mots des Messages")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
    st.image(wordcloud.to_array())

    # Step 5: TF-IDF Analysis
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.85, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Cleaned_Message'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    highest_tfidf_words = tfidf_df.max(axis=0).sort_values(ascending=False)
    top_keywords = highest_tfidf_words.head(20)

    # st.write("### Principaux Mots Clés pour les Publicités")
    # st.bar_chart(top_keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_keywords.to_dict())
    st.write("### Nuage de Mots Clés basé sur le TF-IDF")
    st.image(wordcloud.to_array())

    # Step 6: Bigram Analysis
    stemmer = SnowballStemmer('french')

    def advanced_clean_message(message):
        if pd.isnull(message):
            return ""
        message = message.lower()
        message = ''.join(char for char in message if char not in string.punctuation)
        message = ' '.join(word for word in message.split() if word not in stop_words and len(word) > 2)
        return ' '.join(stemmer.stem(word) for word in message.split())

    data['Advanced_Cleaned_Message'] = data['Commentaire ou message'].apply(advanced_clean_message)
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words)
    bigram_matrix = vectorizer.fit_transform(data['Advanced_Cleaned_Message'])
    bigram_df = pd.DataFrame(bigram_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    top_bigrams = bigram_df.sum(axis=0).sort_values(ascending=False).head(20)

    # st.write("### Principaux Bigrams pour les Publicités")
    # st.bar_chart(top_bigrams)

    # Bigram WordCloud
    bigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_bigrams.to_dict())
    st.write("### Nuage de Mots des Bigrams")
    st.image(bigram_wordcloud.to_array())

else:
    st.write("Veuillez télécharger un fichier CSV pour commencer l'analyse.")
