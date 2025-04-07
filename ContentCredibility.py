import pandas as pd
import spacy
import streamlit as st
import wikipedia
from textblob import TextBlob
import re
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# Load SpaCy's English model for linguistic analysis
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading SpaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load the dataset (our predefined "reliable source")
try:
    # Update this path to your actual dataset location
    df = pd.read_csv("samples.csv").rename(columns={"content": "text", "category": "label"})
except FileNotFoundError:
    # Create a minimal dummy dataset if file not found
    df = pd.DataFrame({
        "text": ["The Earth orbits around the Sun.", "The Moon is made of cheese."],
        "label": ["real", "fake"]
    })
    st.warning("Sample dataset not found. Using minimal example dataset.")

class WikipediaBasedClassifier:
    """
    Naive Bayes classifier that uses Wikipedia summaries as a reference point
    for determining the credibility of text.
    """
    def __init__(self):
        # Initialize the classifier pipeline
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.classifier = MultinomialNB(alpha=0.1)
        self.trained = False
        
    def train_with_wikipedia(self, search_terms):
        """Train classifier with Wikipedia articles for the given search terms"""
        summaries = []
        labels = []
        
        # For each search term, get Wikipedia summary and label as credible
        for term in search_terms:
            try:
                wikipedia.set_lang("en")
                summary = wikipedia.summary(term, sentences=4)
                summaries.append(summary)
                labels.append("credible")  # Wikipedia content is considered credible
                
                # Create some "less credible" examples by modifying the summary
                modified_summary = self._create_modified_summary(summary)
                summaries.append(modified_summary)
                labels.append("less_credible")
                
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        
        if not summaries:
            return False
            
        # Fit the vectorizer and transform the summaries
        X = self.vectorizer.fit_transform(summaries)
        
        # Train the Naive Bayes classifier
        self.classifier.fit(X, labels)
        self.trained = True
        return True
    
    def _create_modified_summary(self, summary):
        """Create a modified version of the summary that's less credible"""
        # Add some dubious phrases to the summary
        dubious_phrases = [
            "Some people believe", 
            "It is said that", 
            "Contrary to popular belief",
            "Despite what experts claim",
            "The truth they don't want you to know is"
        ]
        
        # Select a random dubious phrase
        import random
        phrase = random.choice(dubious_phrases)
        
        # Split the summary into sentences
        sentences = summary.split('. ')
        if len(sentences) > 1:
            # Insert the dubious phrase at the beginning of the second sentence
            sentences[1] = phrase + ", " + sentences[1].lower()
            return '. '.join(sentences)
        else:
            # If the summary is just one sentence, add the phrase at the beginning
            return phrase + ", " + summary.lower()
    
    def get_credibility_score(self, text, search_term=None):
        """
        Get a credibility score for the text based on comparison with Wikipedia.
        If search_term is not provided, extract it from the text.
        """
        if not self.trained:
            # Extract search terms from the text if none provided
            if search_term is None:
                search_terms = self._extract_search_terms(text)
            else:
                search_terms = [search_term]
                
            # Train with these search terms
            training_success = self.train_with_wikipedia(search_terms)
            if not training_success:
                return 50, ["Wikipedia training failed, using neutral score (50)"]
        
        # Transform the input text
        X_text = self.vectorizer.transform([text])
        
        # Get probability predictions
        try:
            # Get probability scores for each class
            proba = self.classifier.predict_proba(X_text)[0]
            # Index 1 corresponds to "credible" class if we have 2 classes
            if len(proba) >= 2:
                credible_probability = proba[1]  # Assuming second class is "credible"
            else:
                credible_probability = proba[0]  # Fall back to the only class
                
            # Scale to 0-100
            score = credible_probability * 100
            
            # Add details for explanation
            details = []
            if score > 75:
                details.append(f"Strong similarity to Wikipedia sources (+{round(score-50)})")
            elif score > 50:
                details.append(f"Moderate similarity to Wikipedia sources (+{round(score-50)})")
            elif score > 25:
                details.append(f"Low similarity to Wikipedia sources (-{round(50-score)})")
            else:
                details.append(f"Very low similarity to Wikipedia sources (-{round(50-score)})")
                
            return score, details
            
        except Exception as e:
            # Handle any classification errors
            return 50, [f"Classification error: {str(e)}. Using neutral score (50)"]
    
    def _extract_search_terms(self, text):
        """Extract potential search terms from the text"""
        doc = nlp(text)
        
        # Extract named entities as potential search terms
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]]
        
        # If no entities found, extract noun chunks
        if not entities:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            if noun_chunks:
                # Take the first 2 noun chunks
                entities = noun_chunks[:2]
        
        # If still no good search terms, take first few tokens with NOUN or PROPN POS tags
        if not entities:
            nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            if nouns:
                entities = nouns[:3]
        
        # If all else fails, take the first few words
        if not entities:
            entities = [text.split()[:5]]
        
        return entities

def analyze_language(text):
    """
    Analyze language for credibility clues.
    - Exclamation marks: -10 (sensationalism).
    - Short text: -5 (vague).
    - Sentiment: -10 if very positive/negative (bias).
    - All caps words: -5 (shouting).
    - Hedging language: -5 (uncertainty).
    """
    doc = nlp(text)
    lang_score = 0
    details = []
    
    # Check for sensationalism
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        penalty = min(exclamation_count * 5, 15)  # Cap at -15
        lang_score -= penalty
        details.append(f"Exclamation marks detected (-{penalty})")

    # Check sentence count
    sentences = list(doc.sents)
    if len(sentences) < 2:
        lang_score -= 5
        details.append("Text too short (-5)")

    # Check for all caps words (shouting)
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    if all_caps_words > 0:
        penalty = min(all_caps_words * 2, 10)  # Cap at -10
        lang_score -= penalty
        details.append(f"ALL CAPS words detected (-{penalty})")

    # Check for hedging language
    hedging_words = ["may", "might", "could", "possibly", "perhaps", "allegedly",
                     "reportedly", "some say", "rumored", "claimed"]
    hedging_count = sum(1 for word in hedging_words if word in text.lower())
    if hedging_count > 0:
        penalty = min(hedging_count * 2, 10)  # Cap at -10
        lang_score -= penalty
        details.append(f"Hedging language detected (-{penalty})")

    # Sentiment analysis with TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    if abs(sentiment) > 0.5:  # Very positive or negative
        lang_score -= 10
        details.append("Strong sentiment detected (-10)")

    # Check for balanced perspective (presence of contrasting viewpoints)
    contrast_markers = ["however", "but", "although", "on the other hand", "conversely", "nevertheless"]
    has_contrast = any(marker in text.lower() for marker in contrast_markers)
    if has_contrast:
        lang_score += 5
        details.append("Balanced perspective detected (+5)")

    # Check for citation patterns
    citation_patterns = [r'\(\d{4}\)', r'\[\d+\]', r'according to', r'cited by']
    has_citation = any(re.search(pattern, text.lower()) for pattern in citation_patterns)
    if has_citation:
        lang_score += 10
        details.append("Citations detected (+10)")

    return lang_score, details

def evaluate_source_credibility(source_url):
    """
    Evaluate the credibility of a source based on domain.
    Returns a credibility score from 0 to 100.
    """
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
        "conspiracytheories.com": 15,
        "fake-news.com": 10,
        "hoaxalert.com": 20,
        "yournewswire.com": 5,
        "beforeitsnews.com": 10,
        "theonion.com": 5,
        "worldnewsdailyreport.com": 5,
        "newsbiscuit.com": 5,
        "prntly.com": 15,
        "infowars.net": 10,
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

def extract_main_topic(text):
    """Extract the main topic from the text for Wikipedia search"""
    doc = nlp(text)
    
    # First try to find named entities
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]]
    if entities:
        return entities[0]  # Return the first entity
        
    # If no entities, try noun chunks
    chunks = list(doc.noun_chunks)
    if chunks:
        return chunks[0].text
        
    # Last resort: first few words
    return ' '.join([token.text for token in doc[:3]])

def calculate_credibility(text, use_naive_bayes=True):
    """
    Compute final credibility score with breakdown.
    - Base: 50 (neutral).
    - Adds linguistic analysis score.
    - Adds naive bayes-based Wikipedia comparison score.
    - Clamps between 0 and 100.
    """
    # Extract main topic for Wikipedia search
    main_topic = extract_main_topic(text)
    
    # Language analysis (patterns, sentiment, etc.)
    lang_score, lang_details = analyze_language(text)
    
    # Wikipedia-based classification using Naive Bayes
    if use_naive_bayes:
        wiki_classifier = WikipediaBasedClassifier()
        wiki_score, wiki_details = wiki_classifier.get_credibility_score(text, main_topic)
    else:
        # Fallback to a neutral Wikipedia score if Naive Bayes is disabled
        wiki_score, wiki_details = 50, ["Wikipedia comparison skipped"]
    
    # Calculate weighted final score (language: 40%, wikipedia: 60%)
    total_score = 50 + (lang_score * 0.4) + ((wiki_score - 50) * 0.6)
    
    # Clamp the score between 0 and 100
    total_score = max(0, min(100, total_score))
    
    return total_score, lang_details + wiki_details

# Streamlit UI
def main():
    st.title("📊 Advanced Credibility Checker")
    st.markdown("Enter text to evaluate its credibility using Naive Bayes classification with Wikipedia references!")

    # Text input
    user_input = st.text_area("Text to analyze", height=150, 
                           placeholder="Enter the text you want to check for credibility...")
    
    source_url = st.text_input("Source URL (optional)", 
                             placeholder="https://example.com/article")
    
    # Advanced options
    with st.expander("Advanced Options"):
        use_naive_bayes = st.checkbox("Use Naive Bayes classifier with Wikipedia", value=True)
        
    # Button and results
    if st.button("Check Credibility"):
        if not user_input:
            st.warning("Please enter some text to analyze!")
            return
            
        with st.spinner("Analyzing credibility..."):
            # Calculate scores
            source_score = evaluate_source_credibility(source_url) if source_url else 0
            text_score, details = calculate_credibility(user_input, use_naive_bayes)
            
            # Combine scores if source URL was provided
            if source_score > 0:
                final_score = (text_score * 0.6) + (source_score * 0.4)
                source_detail = f"Source domain credibility: {source_score}/100"
                details.append(source_detail)
            else:
                final_score = text_score
            
            # Round the final score
            final_score = round(final_score)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Score meter (visual representation)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.progress(final_score/100)
                
                # Color-coded score display
                if final_score > 70:
                    st.success(f"Credibility Score: {final_score}/100")
                    st.markdown("**Verdict:** Likely Credible ")
                elif final_score > 40:
                    st.warning(f"Credibility Score: {final_score}/100")
                    st.markdown("**Verdict:** Somewhat Questionable ")
                else:
                    st.error(f"Credibility Score: {final_score}/100")
                    st.markdown("**Verdict:** Highly Doubtful ")
            
            # Show breakdown
            st.subheader("Score Breakdown")
            for detail in details:
                st.write(f"• {detail}")
                
            # Extract main topic
            main_topic = extract_main_topic(user_input)
            st.write(f"**Detected main topic:** {main_topic}")
            
            # Try to fetch some Wikipedia information for reference
            try:
                wikipedia.set_lang("en")
                wiki_summary = wikipedia.summary(main_topic, sentences=2)
                st.info(f"**Wikipedia reference:** {wiki_summary}")
            except:
                st.info("No direct Wikipedia reference found for this topic.")

    st.markdown("---")
    st.write("Built for Checking Credibility Enhanced with Naive Bayes Classification")

if __name__ == "__main__":
    main()



## TO RUN THE PROGRAM , TYPE streamlit run name_of_the_file.py IN THE TERMINAL
