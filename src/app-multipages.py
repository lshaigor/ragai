# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
#example URL: https://www.metacritic.com/game/elden-ring/
import os
import streamlit as st
import langchain

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
from bs4 import BeautifulSoup
import cloudscraper
from urllib.parse import urljoin

load_dotenv()

# main method to web crawl pages and store in the embeddings vector store
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    #print("get_vectorstore_from_url",document)
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    #print here example of chunks
    print("===========\n")
    print("document chunk 0", document_chunks[0])
    print("===========\n")
    print("document chunk 1",document_chunks[1])
    print("===========\n")
    print("document chunk 2", document_chunks[2])

    vector_store_dir = "/Users/leonshaigorodsky/projects/ragai/chromaDB"
    embeddings = OpenAIEmbeddings()

    # Check if the vector store directory exists
    if os.path.exists(vector_store_dir):
        # Load the existing vector store from the directory
        vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)

        # Append new content to the vector store
        vector_store.add_texts(document_chunks)
        print("IF vector_store:", vector_store)
    else:
        # Create a new vector store in the directory
        vector_store = Chroma.from_documents(
            document_chunks,
            embeddings,
            persist_directory=vector_store_dir
            )
        vector_store.persist()

        print("ELSE vector_store:", vector_store)



    # OLD create a vectorstore from the chunks
    #vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

# method to retrieve relevant document chunks to sent to LLM along with the user prompt
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    
    #look up for document chunks, relevant to the user question
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    print("get_context_retriever_chain",retriever_chain)

    return retriever_chain

# method to combine the retriever chain and user prompt to send to LLM
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    print ("\nget_conversational_rag_chain function prompt:", prompt)

    #combine the retriever chain with the LLM
    #create_retrieval_chain(retriever_chain, llm, prompt)
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# method to get the response from the LLM based on the user input 
def get_response(user_input):
    #pull relevant document chunks from the embeddings vector store that are semantically relevant to the user_input
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    #get the conversational rag chain (combining the retriever chain with the LLM)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    #add the user input to the chat history and get the response from the LLM based on provided doc chunks + LLM common knowledge    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# Web page setup

# Basic page config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("RAG AI based chat with websites")

# sidebar
website_url = 'https://www.metacritic.com/game/elden-ring/'
with st.sidebar:
    st.image('src/wbg-logo.svg', width=140) #, caption='WBA AI Hackathon')

    #create blank space before text box
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")    
    #st.header("Settings")
    website_url = st.text_input("Website URL",value=website_url)


if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a WBA hackaton AI bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    
        
        html_doc = requests.get(website_url)
        scraper = cloudscraper.create_scraper()

        soup = BeautifulSoup(scraper.get("https://www.metacritic.com/game/elden-ring/").text, 'html.parser')
        with st.spinner(':red[Please wait while we fetch data from urls]'):
            secondary_urls = []
            for a in soup.find_all('a', href=True):
                secondary_url = urljoin(website_url,a['href'])
                secondary_urls.append(secondary_url)

            ctr = 0
            for secondary_url in set(secondary_urls):
                try:
                    st.session_state.vector_store = get_vectorstore_from_url(secondary_url) 
                    ctr += 1
                    with st.sidebar:
                        st.write(f':blue[Processed {secondary_url}]')                    

                except Exception as e:
                    print(f'Error scraping {secondary_url}')
                    pass
                if ctr > 9:
                    break
    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
