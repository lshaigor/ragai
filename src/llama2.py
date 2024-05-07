# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
#example URL: https://www.metacritic.com/game/hogwarts-legacy/

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain.chains import LangChain
from os.path import expanduser
from langchain_community.llms import LlamaCpp

from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)


load_dotenv()

def get_vectorstore_from_url(url = "https://gist.githubusercontent.com/Red-Folder/433b929b026be0aaae46bc49be0d2bea/raw/3422f90f1e7494477540cfac1bbb76e83aaf2480/payload.json"):
    '''
        Load vector store. 
        Right now loads a singled URL
    '''
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    print("get_vectorstore_from_url",document)
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):    
    llm = get_llm(st.session_state.current_selected_model)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    print("get_context_retriever_chain",retriever_chain)

    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    # llm = get_llm()
    llm = get_llm(st.session_state.current_selected_model)
        
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    print("get_conversational_rag_chain",stuff_documents_chain)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# Web page setup
# Basic page config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("RAG AI based chat with websites")

if "chains" not in st.session_state:
    st.session_state.chains = {}
if "current_selected_model" not in st.session_state:
    st.session_state.current_selected_model = ""

def get_llm(model_name) -> LlamaCpp:
    if not st.session_state.chains.get(model_name):
        # model_path = expanduser("./models/llama-2-7b-chat.Q4_0.gguf")
        model_path = expanduser(f"./models/{model_name}")

        return LlamaCpp(
            model_path=model_path,
            streaming=False,
            verbose=True, 
            n_ctx=2048
        )


        
def get_response(user_input):
    response = st.session_state.conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']
    
def get_chain(model_name):
    if st.session_state.chains.get(model_name):
        return st.session_state.chains.get(model_name)
    else:
        raise ValueError(f"Chain not set: {model_name} - Initialized Models: {st.session_state.chains}")
        
def update_chat_history(message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=message),
        ]
    else:
        st.session_state.chat_history.append(AIMessage(content=message))

    
# Left sidebar configuration to show logo & URL text box
with st.sidebar:
    st.image('src/wbg-logo.svg', width=140) #, caption='WBA AI Hackathon')

    #create blank space before text box
    for i in range(1,13):
        st.text("")

    # Create a dropdown with some options
    option = st.selectbox(
        'Select model',
        ('llama-2-7b-chat.Q4_0.gguf', 
         'llama-2-13b-chat.Q5_K_M.gguf', 
         'codellama-13b-instruct.Q5_K_M.gguf',
         'codellama-34b-instruct.Q5_K_M.gguf'))

    # Store the previous selection in session state
    prev_option = st.session_state.get("prev_option", None)

        
    if prev_option != option:
        st.write(f"You selected: {option}")
        # Check if the selection has changed
        if not prev_option:
            update_chat_history(f"Hello, I am a WBA hackaton AI bot. How can I help you? I will use model: {option}")        
        else:
            update_chat_history(f"LLM Model Selection Changed from: {prev_option} - to: {option}")        
            
        # Update the session state with the current selection
        st.session_state.prev_option = option    
        # chain = set_chain(option)         
        st.session_state.current_selected_model = option


if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url()    
    st.session_state.retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    st.session_state.conversation_rag_chain = get_conversational_rag_chain(st.session_state.retriever_chain)


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
