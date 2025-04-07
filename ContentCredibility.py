import pandas as pd
import spacy
import streamlit as st
import wikipedia
from textblob import TextBlob
import re
import os

# Load SpaCy's English model for linguistic analysis
nlp = spacy.load("en_core_web_sm")

# Load the dataset (our predefined "reliable source")
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\samples.csv").rename(columns={"content": "text", "category": "label"})

def check_claims(text):
    """
    Verify claims against dataset and Wikipedia.
    - Dataset match: +50 for real, -20 for fake.
    - Wikipedia match: +20 if found, -10 if not.
    - Returns score and details for UI.
    """
    text = text.lower()
    claim_score = 0
    details = []

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")

    # Check dataset first
    match_found = False
    for index, row in df.iterrows():
        dataset_text = row["text"].lower()
        if text in dataset_text or dataset_text in text:
            match_found = True
            if row["label"] == "real":
                claim_score += 50
                details.append("Matched real news in dataset (+50)")
            else:
                claim_score -= 20
                details.append("Matched fake news in dataset (-20)")
            break

    # If no dataset match, try Wikipedia
    if not match_found:
        try:
            wikipedia.set_lang("en")
            # Use first few words as a search term
            doc = nlp(text)
            search_term = " ".join([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]][:3])
            summary = wikipedia.summary(search_term, sentences=4)
            if any(word in summary.lower() for word in text.split()):
                claim_score += 10
                details.append("Found in Wikipedia (+20)")
            else:
                claim_score -= 10
                details.append("No clear Wikipedia match (-10)")
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            claim_score -= 10
            details.append("Wikipedia check failed (-10)")

    return claim_score, details

def analyze_language(text):
    """
    Analyze language for credibility clues.
    - Exclamation marks: -10 (sensationalism).
    - Short text: -5 (vague).
    - Sentiment: -10 if very positive/negative (bias).
    """
    doc = nlp(text)
    lang_score = 0
    details = []
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")
    
    # Check for sensationalism
    if "!" in text:
        lang_score -= 10
        details.append("Exclamation mark detected (-10)")

    # Check sentence count
    sentences = list(doc.sents)
    if len(sentences) < 2:
        lang_score -= 5
        details.append("Text too short (-5)")

    # Sentiment analysis with TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    if abs(sentiment) > 0.5:  # Very positive or negative
        lang_score -= 10
        details.append("Strong sentiment detected (-10)")

    return lang_score, details

def calculate_credibility(text):
    """
    Compute final credibility score with breakdown.
    - Base: 50 (neutral).
    - Adds claim and language scores.
    - Clamps between 0 and 100.
    """
    txt = text.lower()
    claim_score, claim_details = check_claims(txt)
    lang_score, lang_details = analyze_language(txt)
    total_score = 50 + claim_score + lang_score
    total_score = max(0, min(100, total_score))
    return total_score, claim_details + lang_details

def evaluate_source_credibility(source_url):

    credible_domains = {
        "reuters.com": 95,
        "apnews.com": 95,
        "bbc.com": 90,
        "bbc.co.uk": 90,
        "npr.org": 90,
        "nytimes.com": 85,
        "washingtonpost.com": 85,
        "theguardian.com": 85,
        "economist.com": 85,
        "nature.com": 95,
        "science.org": 95,
        "scientificamerican.com": 90,
        "cdc.gov": 95,
        "nih.gov": 95,
        "who.int": 95,
        "edu": 80,  # Educational domains generally
        "gov": 85,  # Government domains generally
    }
    
    low_credibility_domains = {
        "infowars.com": 10,
        "naturalcures.com": 20,
        "clickbait-site.com": 10,
    }
    
    # Extract domain from URL
    domain = None
    if source_url:
        match = re.search(r'https?://(?:www\.)?([^/]+)', source_url)
        if match:
            domain = match.group(1).lower()
    
    # Check if domain is in our database
    if domain:
        # Check for exact matches
        if domain in credible_domains:
            return credible_domains[domain]
        if domain in low_credibility_domains:
            return low_credibility_domains[domain]
        
        # Check for partial matches (e.g., subdomain.example.com)
        for cred_domain, score in credible_domains.items():
            if domain.endswith(cred_domain):
                return score * 0.9  # Slightly reduce score for subdomains
        
        for low_cred_domain, score in low_credibility_domains.items():
            if domain.endswith(low_cred_domain):
                return score * 1.1  # Slightly increase score for subdomains
                
        # Check for domain type (.edu, .gov, etc.)
        for domain_type, score in credible_domains.items():
            if domain_type in domain and len(domain_type) <= 4:  # Ensure it's a TLD
                return score * 0.8  # Reduced score for general domain type
    else:
        return 0  # No domain found
    
    # Default score for unknown sources
    return 50

# Streamlit UI
st.title("Credibility Checker")
st.markdown("Enter text to evaluate its credibility score!")

# Text input
user_input = st.text_area("Text", height=150, placeholder="Type something like 'The sky is blue'")
user_input1 = st.text_area("Text", height=150, placeholder="Also give the source (URL) of the text(optional)")

# Button and results
if st.button("Check Credibility"):
    score = evaluate_source_credibility(user_input1)
    if score == 0:
        st.write("Please enter a valid URL!\n Credibility Score without the source URL: ")
        if user_input:
            
            score, details = calculate_credibility(user_input)
            # Display score with color coding
            if score > 70:
                st.success(f"**Credibility Score: {score}/100** - Looks credible!")
            elif score > 30:
                st.warning(f"**Credibility Score: {score}/100** - Might be questionable.")
            else:
                st.error(f"**Credibility Score: {score}/100** - Highly doubtful!")
            
            # Show breakdown
            st.subheader("Score Breakdown")
            for detail in details:
                st.write(f"- {detail}")
        else:
            st.warning("Please enter some text!")
    else:
        if score > 70:
            st.success(f"**Credibility Score: {score}/100** - Looks credible!")
        elif score > 30:
            st.warning(f"**Credibility Score: {score}/100** - Might be questionable.")
        else:
            st.error(f"**Credibility Score: {score}/100** - Highly doubtful!")

# Add a little footer for flair
st.markdown("---")
st.write("Built for a 24hr AI Hackathon by Ankith, Chinmayee, Yashvi and Aditya!")