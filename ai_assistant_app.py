import streamlit as st
import requests
import json
import numpy as np
import io
import base64
from PIL import Image

# --- Global Configuration ---
# NOTE: The actual API key is provided at runtime by the environment.
# Leaving this as an empty string ensures compatibility with the Canvas environment.
API_KEY = "" 
BASE_URL_GEMINI_FLASH = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Utility Functions for Gemini API Calls ---

# 1. Exponential Backoff implementation for robust API calls
def call_api_with_backoff(url, payload, max_retries=5):
    """Handles API call with exponential backoff for rate limiting."""
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{url}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check specifically for rate limit or server errors (429, 500, 503)
            if response.status_code in (429, 500, 503) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"Rate limit hit or server error ({response.status_code}). Retrying in {wait_time}s...")
                import time
                time.sleep(wait_time)
            else:
                st.error(f"API Error: Could not complete request (Status {response.status_code}).")
                st.error(e)
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None
    return None

# 2. Simple Q&A Assistant Feature (Stateless) - Modified to remove Session State dependency
def qna_feature_stateless():
    """Implements a simple, stateless Q&A feature using the Gemini API.
    This version avoids using st.session_state for chat history, mitigating environment errors."""
    st.header("💡 Simple Q&A Assistant")
    st.write("Enter your question and click 'Get Answer'. Due to environment constraints, conversation history is not maintained.")

    prompt = st.text_area("Your Question:", key="qna_input")
    submit_button = st.button("Get Answer")
    
    if submit_button and prompt:
        with st.spinner('Thinking...'):
            # Construct the API payload with Google Search grounding enabled
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],  # Enables grounding with Google Search
                "systemInstruction": {"parts": [{"text": "You are a helpful and concise AI assistant. Provide direct answers, referencing the web source if available."}]},
            }

            response_json = call_api_with_backoff(BASE_URL_GEMINI_FLASH, payload)

        if response_json:
            try:
                # Extract text and sources
                text = response_json['candidates'][0]['content']['parts'][0]['text']
                
                # Extract citations/sources
                sources_markdown = ""
                grounding_metadata = response_json['candidates'][0].get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources_markdown = "\n\n---\n**Sources:**\n"
                    for attr in grounding_metadata['groundingAttributions']:
                        if attr.get('web') and attr['web'].get('title') and attr['web'].get('uri'):
                            title = attr['web']['title']
                            uri = attr['web']['uri']
                            sources_markdown += f"- [{title}]({uri})\n"
                
                full_response = text + sources_markdown
                
                st.subheader("Assistant Response:")
                st.markdown(full_response)

            except (KeyError, IndexError) as e:
                st.error(f"Could not parse AI response. Error: {e}")
                st.markdown("Sorry, I received an unreadable response.")
        
# 3. Image Recognition Feature - Multimodal analysis using Gemini
def image_recognition_feature():
    """Implements the Image Recognition feature using the Gemini API."""
    st.header("🖼️ Multimodal Image Recognition")
    st.write("Upload an image and ask the AI to describe it or answer questions about it.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to base64 for API payload
        img_byte_arr = io.BytesIO()
        # Use the uploaded file's format if available, otherwise default to PNG
        img_format = uploaded_file.type.split('/')[-1].upper() if '/' in uploaded_file.type else 'PNG'
        image.save(img_byte_arr, format=img_format)
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Text prompt for the image
        image_prompt = st.text_input("What would you like to know about this image?", 
                                     "Describe this image in detail and identify any key objects.")
        
        if st.button("Analyze Image"):
            if image_prompt:
                with st.spinner('Analyzing image...'):
                    # Construct the multimodal API payload
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": image_prompt},
                                    {
                                        "inlineData": {
                                            "mimeType": uploaded_file.type,
                                            "data": encoded_image
                                        }
                                    }
                                ]
                            }
                        ],
                    }
                    
                    response_json = call_api_with_backoff(BASE_URL_GEMINI_FLASH, payload)
                
                if response_json:
                    try:
                        text = response_json['candidates'][0]['content']['parts'][0]['text']
                        st.subheader("AI Analysis")
                        st.info(text)
                    except (KeyError, IndexError):
                        st.error("Could not parse AI response for image analysis.")
            else:
                st.warning("Please enter a prompt to analyze the image.")

# 4. Text Classification Feature (Mock for simplicity)
def text_classification_feature():
    """Implements a mock Text Classification feature to fulfill the requirement."""
    st.header("📝 Text Classification (Mock NLP Model)")
    st.write("This function simulates a machine learning model classifying text into predefined categories (e.g., Sentiment, Topic).")
    
    text_input = st.text_area("Enter text to classify:", "The new product launch was a huge success, customers are thrilled!")
    
    if st.button("Classify Text"):
        # Simple rule-based classification mock to show the concept
        text_lower = text_input.lower()
        classification = "Neutral"
        
        if "success" in text_lower or "thrilled" in text_lower or "love" in text_lower:
            classification = "Positive Sentiment / Marketing Topic"
        elif "fail" in text_lower or "error" in text_lower or "hate" in text_lower:
            classification = "Negative Sentiment / Technical Support Topic"
        
        st.subheader("Classification Result")
        st.success(f"**Category:** {classification}")
        st.markdown(f"*(This result is generated by a simple rule-based mock, replacing a complex NLP model pipeline.)*")


# --- Main Application Layout ---

def main():
    try:
        # Check if the context is available before setting config (Best practice for stability)
        st.set_page_config(
            page_title="Smart AI Assistant Project",
            layout="centered",
            initial_sidebar_state="expanded"
        )
    except Exception:
        # If set_page_config fails (due to missing context), just continue
        pass

    st.title("🤖 Smart AI Assistant Project")
    st.markdown("A combined ML/DL/Generative AI application deployed with **Streamlit**.")
    
    # Sidebar navigation
    st.sidebar.title("Assistant Features")
    
    # Update the feature dictionary with the stateless Q&A function
    features = {
        "Q&A Assistant (Stateless)": qna_feature_stateless,
        "Image Recognition": image_recognition_feature,
        "Text Classification (Mock)": text_classification_feature,
    }
    
    selected_feature = st.sidebar.radio("Select an AI Module:", list(features.keys()))

    # Execute the selected feature function
    features[selected_feature]()

if __name__ == "__main__":
    main()
