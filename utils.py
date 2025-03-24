import feedparser
from newspaper import Article
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import spacy
from collections import Counter
from gtts import gTTS
from bs4 import BeautifulSoup
import re
import os

# -------------------------
# News Extraction Function
# -------------------------
def get_news_articles(company_name, max_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={company_name}"
    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:max_articles]:
        # Clean summary text
        soup = BeautifulSoup(entry.summary, "html.parser")
        clean_summary = soup.get_text().strip()
        
        article_data = {
            "title": entry.title.strip(),
            "summary": clean_summary,
            "link": entry.link,
            "published": entry.published
        }
        
        try:
            article = Article(entry.link)
            article.download()
            article.parse()
            # Clean full content
            if article.text:
                article_data["full_content"] = ' '.join(article.text.split())
            else:
                article_data["full_content"] = clean_summary
        except Exception as e:
            article_data["full_content"] = clean_summary
            print(f"Error extracting content from {entry.link}: {e}")

        articles.append(article_data)
        time.sleep(1)
    return articles

# -------------------------
# Sentiment Analysis Setup
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(text):
    if not text or text.strip() == "":
        return "Neutral"

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = ' '.join(text.split())
    text = text[:1000]

    try:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits[0].numpy()
        scores = np.exp(scores) / np.sum(np.exp(scores))
        sentiment_index = np.argmax(scores)
        return SENTIMENT_LABELS[sentiment_index]
    except Exception as e:
        print(f"Warning: Error in sentiment analysis: {e}")
        return "Neutral"

# -------------------------
# Load spaCy for Topic Extraction
# -------------------------
nlp = spacy.load("en_core_web_sm")

def extract_topics(text, top_n=3):
    """
    Extracts key topics from text using spaCy, focusing on meaningful nouns.
    """
    if not text or text.strip() == "":
        return []

    # Clean the text
    text = BeautifulSoup(text, "html.parser").get_text()
    text = ' '.join(text.split())
    
    doc = nlp(text)
    
    # Extract only meaningful nouns (excluding stop words and short words)
    nouns = [
        token.text.lower() for token in doc 
        if (token.pos_ == "NOUN" or token.pos_ == "PROPN") 
        and len(token.text) > 2  # Exclude short words
        and not token.is_stop  # Exclude stop words
        and token.text.isalnum()  # Only include alphanumeric words
    ]
    
    # Count frequencies
    freq = Counter(nouns)
    
    # Get top topics, excluding the company name if it appears too frequently
    topics = [word for word, count in freq.most_common(top_n + 3)  # Get extra topics
              if not any(char.isdigit() for char in word)][:top_n]  # Remove topics with numbers
    
    return topics

# -------------------------
# Hindi Text-to-Speech (TTS)
# -------------------------
def text_to_speech_hindi(text, output_path="output_hindi.mp3"):
    if not text or text.strip() == "":
        text = "कोई जानकारी उपलब्ध नहीं है।"
    try:
        tts = gTTS(text=text, lang='hi')
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None

# -------------------------
# Full Report Generator
# -------------------------
def generate_company_report(company_name, max_articles=10):
    raw_articles = get_news_articles(company_name, max_articles)
    processed_articles = []
    all_topics = []
    sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in raw_articles:
        sentiment = analyze_sentiment(article["summary"])
        topics = extract_topics(article["summary"])
        sentiment_distribution[sentiment] += 1
        processed_articles.append({
            "Title": article["title"],
            "Summary": article["summary"],
            "Sentiment": sentiment,
            "Topics": topics
        })
        all_topics.append(set(topics))

    # Comparative Analysis
    coverage_differences = []
    for i in range(len(processed_articles)):
        for j in range(i + 1, len(processed_articles)):
            a1 = processed_articles[i]
            a2 = processed_articles[j]
            comparison = {
                "Comparison": f"Article {i+1} highlights {a1['Title']}, while Article {j+1} discusses {a2['Title']}.",
                "Impact": f"The first article suggests {a1['Sentiment']} sentiment; the second suggests {a2['Sentiment']} sentiment."
            }
            coverage_differences.append(comparison)

    common_topics = set.intersection(*all_topics) if all_topics else set()
    unique_topic_map = {}
    for idx, topics in enumerate(all_topics):
        unique_topic_map[f"Unique Topics in Article {idx+1}"] = list(topics - common_topics)

    if sentiment_distribution["Positive"] > max(sentiment_distribution["Negative"], sentiment_distribution["Neutral"]):
        final_summary = f"{company_name}’s latest news coverage is mostly positive. Potential stock growth expected."
    elif sentiment_distribution["Negative"] > sentiment_distribution["Positive"]:
        final_summary = f"{company_name}’s latest news coverage is mostly negative. Potential risks highlighted."
    else:
        final_summary = f"{company_name}’s news coverage shows mixed sentiments."

    audio_file = text_to_speech_hindi(final_summary)
    audio_output = "[Play Hindi Speech]" if audio_file else "[Audio Generation Failed]"

    report = {
        "Company": company_name,
        "Articles": processed_articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_distribution,
            "Coverage Differences": coverage_differences[:5],
            "Topic Overlap": {
                "Common Topics": list(common_topics),
                **unique_topic_map
            }
        },
        "Final Sentiment Analysis": final_summary,
        "Audio": audio_output
    }
    return report

# -------------------------
# Test Example
# -------------------------
if __name__ == "__main__":
    company = input("Enter company name: ")
    report = generate_company_report(company)
    import json
    print(json.dumps(report, indent=4, ensure_ascii=False))
