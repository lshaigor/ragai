# AI RAG example webscraping Website content from URL using LangChain Chatbot with Streamlit UI

LangChain Chatbot with Streamlit UI. 
Extract content of the URL and additional linked pages. 
In order to chat with this UI using need to enter top level URL and bot will webscrape the content into vector DB to use this info in the chat with AI both 


## Features
- **Website Interaction**: The chatbot uses the latest version of LangChain to interact with and extract information from various websites.
- **Large Language Model Integration**: current implementation is done with OpenAI but can be adopted with other models

- **Streamlit UI**: Simple and nice user interface built with Streamlit, that include top level URL for RAG content to be used in the chat with AI where langchain combine info from webscraped pages and pass it to model to get RAG based answers

## Brief explanation of how RAG works

A RAG bot is short for Retrieval-Augmented Generation. This means that we are going to "augment" the knowledge of our LLM with new information that we are going to pass in our prompt. We first vectorize all the text that we want to use as "augmented knowledge" and then look through the vectorized text to find the most similar text to our prompt. We then pass this text to our LLM as a prefix.

![RAG Diagram](docs/HTML-rag-diagram.jpg)

## Installation
Ensure you have Python installed on your system. Then clone this repository:

```bash
git clone [repository-link]
cd [repository-directory]
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create your own .env file with the following variables:

```bash
OPENAI_API_KEY=[your-openai-api-key]
```

## Usage
To run the Streamlit app:

```bash
streamlit run app.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

To run in venv: 

```bash
python -m venv ragai
```
```bash
source ragai/bin/activate
```
