import streamlit as st
import requests
import json
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="Company News Analyzer", layout="wide")
st.title("📰 Company News Summarizer & Sentiment Analyzer")

# Input for company name
company_name = st.text_input("Enter Company Name:", "Apple")

if st.button("Generate Report"):
    with st.spinner("Fetching and analyzing news..."):
        try:
            response = requests.get(f"https://<your-api-url>/generate_report?company_name={company_name}")
            data = response.json()

            if data["status"] == "success":
                report = data["data"]

                st.subheader(f"📊 Sentiment Summary for {report['Company']}")
                sentiment_chart_data = {
                    "Positive": report["Comparative Sentiment Score"]["Sentiment Distribution"]["Positive"],
                    "Neutral": report["Comparative Sentiment Score"]["Sentiment Distribution"]["Neutral"],
                    "Negative": report["Comparative Sentiment Score"]["Sentiment Distribution"].get("Negative", 0),
                }
                st.bar_chart(sentiment_chart_data)

                st.markdown("### 📝 Final Sentiment Analysis")
                st.success(report["Final Sentiment Analysis"])

                st.markdown("---")
                st.markdown("### 🗞️ Articles")
                for i, article in enumerate(report["Articles"], start=1):
                    st.markdown(f"**{i}. {article['Title']}**")
                    st.markdown(f"Summary: {article['Summary']}")
                    st.markdown(f"Sentiment: `{article['Sentiment']}`")
                    st.markdown(f"Topics: {', '.join(article['Topics'])}")
                    st.markdown("---")

                st.markdown("### 🎧 Hindi Audio Summary")
                # Generate Hindi TTS locally (optional fallback if no audio URL)
                tts = gTTS(text=report["Final Sentiment Analysis"], lang='hi')
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format='audio/mp3')

            else:
                st.error("API Error: Unable to fetch report.")

        except Exception as e:
            st.error(f"Error: {e}")
