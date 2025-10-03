🤖 Smart AI Assistant Project
This is a multi-feature AI application built with Streamlit and powered by the Google Gemini API (gemini-2.5-flash-preview-05-20). It combines Generative AI capabilities with machine learning concepts.





🌟 Key Features
The application offers three modules:

Q&A Assistant: Provides factual, up-to-date answers using the Google Search tool (grounding) and includes source citations. It runs as a stateless, single-query tool.

Image Recognition: Upload an image and provide a text prompt to receive a detailed analysis from the multimodal Gemini model.

Text Classification (Mock): A mock feature demonstrating a classification task, assigning predefined categories (e.g., Sentiment) to text input.





⚙️ Setup and Run
Prerequisites
Python 3.8+





Install dependencies: pip install streamlit requests pillow





Execution
Save the code as ai_assistant_app.py.

Run from terminal: streamlit run ai_assistant_app.py





🔑 API Key
The application uses API_KEY = "" in ai_assistant_app.py.

Local Setup: Replace "" with your actual Gemini API key.

Integrated Platform: The key is automatically injected by the environment at runtime.





♻️ Robustness
The code uses exponential backoff for all API calls to reliably handle transient issues like rate limiting (HTTP 429).
