# LLM-Q&A-Application
# https://llm-q-a-application.streamlit.app/

The "LLM-Q&A-Application" application is an advanced question-answering (QA) tool crafted using Streamlit, Langchain, and Google's Generative API models. This innovative tool empowers users to upload textual documents and pose inquiries based on the document's content, leveraging RAG for precise and contextually relevant responses.

## Key Features

**Document Upload:** Users can effortlessly upload text documents in .pdf format.
**Query Processing:** Following document upload, users can input queries or questions to extract insights from the document's content.
**Secure API Key Management:** The application mandates a Google Generative API key for processing, ensuring secure and authenticated access to the language models.
**Responsive User Interface:** Developed using Streamlit, the app delivers an interactive and user-friendly interface.

## Usage Guidelines

### To run "Ask the Doc App" on your local machine, follow these steps:

Clone the Repository: Clone this repository to your local machine using git clone.

Install Dependencies:

Ensure you have Python installed. Install required packages using pip install -r requirements.txt. Set Up Google Generative API Key: You must have an Google Generative API key to use this app. If you don't have one, you can obtain it from Googlegenerative's website.

Run the Streamlit App:

- Access the application through your web browser following the execution of the Streamlit command.

- Utilize the file uploader feature to upload your .txt document seamlessly.

- Once the document is uploaded, input your question or query into the text input field.

- Enter your Google Generative API key securely in the designated field.

- Click 'Submit' to process your query. The application will promptly present the answer based on the content of your uploaded document.

The app lets you type in your Google Generative API key directly, so it's never stored or recorded. Remember to keep your API key private and avoid sharing it with anyone. Additionally, it clears your chat history once you upload a new file for added privacy.

## Contributions

Contributions to this project are encouraged. Please fork the repository and submit a pull request containing your proposed modifications.
