# Import libraries
import os
from dotenv import load_dotenv
import uuid
import datetime
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def load_environment_variables():
    """
    Loads environment variables from a .env file and sets them in the OS environment.
    """
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
    os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
    os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
    os.environ['LANGCHAIN_TRACING_V2'] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    return groq_api_key


def create_embeddings():
    """
    Create and return HuggingFace embeddings using the specified model.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def handle_uploaded_files(uploaded_files):
    """
    Handles the uploaded files by saving them temporarily and loading their contents.
    """
    documents = []
    for up_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, 'wb') as file:
            file.write(up_file.getvalue())
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    return documents


def create_vectorstore(documents, embeddings):
    """
    Creates a vector store from the given documents and embeddings.

    This function splits the input documents into smaller chunks using a 
    RecursiveCharacterTextSplitter and then creates a FAISS vector store 
    from the split documents and provided embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def create_retriever(vectorstore, llm):
    """
    Creates a history-aware retriever using a given vector store and language model.

    This function sets up a retriever that can contextualize user questions based on chat history.
    It uses a system prompt to reformulate questions so they can be understood without the chat history.
    """
    contextualize_q_system_prompt = (
        """
        Given a chat history and the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history. DO NOT answer the question,
        just reformulate it if needed, otherwise return as is.
        """
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    retriever = vectorstore.as_retriever()
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def create_question_answer_chain(llm):
    """
    Creates a question-answer chain using a provided language model (LLM).

    This function sets up a system prompt for an AI assistant to answer questions
    in detail based on the provided context. The function then creates a 
    question-answer prompt template using the system prompt and a placeholder for
    chat history and user input.
    """
    system_prompt = (
        """
        You are an AI assistant. Answer the questions in detail based on the provided context only.
        You must generate a response that answers the user's question in detail using the provided context. 
        The context can be any length and may contain references to previous responses.
        \n\n
        {context} 
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)


def initialize_session_state():
    """
    Initialize the session state with default values if they are not already set.
    """
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())  # Generate a unique ID
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if "query_count" not in st.session_state:
        st.session_state['query_count'] = 0
        st.session_state['submit_disabled'] = False


def get_session_history():
    """
    Retrieve the chat message history for the current session.

    This function checks if the current session ID exists in the session state store.
    If it does not exist, it initializes a new ChatMessageHistory for the session.
    It then returns the chat message history associated with the current session ID.
    """
    session_id = st.session_state['session_id']
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def manage_query_count():
    """
    Manages the query count in the session state.

    If the query count reaches or exceeds 5, a warning message is displayed,
    and the submit button is disabled. Otherwise, the query count is incremented
    by 1, and the submit button remains enabled.
    """
    if st.session_state['query_count'] >= 5:
        st.warning("You have reached the limit of 5 queries. Please try again later.")
        st.session_state['submit_disabled'] = True
    else:
        st.session_state['query_count'] += 1
        st.session_state['submit_disabled'] = False


def create_app():
    """
    Creates a Streamlit application for exploring PDFs conversationally.
    """
    st.title("Conversational PDF Explorer")
    st.write("")
    st.markdown("""
                Upload PDFs, ask questions, get answers instantly and review your entire conversation history with this Conversational 
                RAG app (built on Streamlit, LangChain, FAISS, and Groq). It's like having a surprisingly 
                knowledgeable friend who only talks about your documents, and never forgets a conversation.
                """)
    st.write("")


def process_uploaded_files(uploaded_files, embeddings, llm):
    """
    Processes the uploaded files and sets up a retrieval-augmented generation (RAG) 
    chain for question answering.
    """
    documents = handle_uploaded_files(uploaded_files)
    vectorstore = create_vectorstore(documents, embeddings)
    history_aware_retriever = create_retriever(vectorstore, llm)
    question_answer_chain = create_question_answer_chain(llm)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    session_id = st.session_state['session_id']
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    handle_user_input(conversational_rag_chain, session_id)


def handle_user_input(conversational_rag_chain, session_id):
    """
    Handles the user input for a conversational RAG (Retrieval-Augmented Generation) chain.

    This function captures the user's question from a text input field, manages the query count,
    and invokes the conversational RAG chain to get a response. It then displays the response
    and the chat history.
    """
    user_input = st.text_input("Your question:")
    if user_input:
        manage_query_count()
        if not st.session_state['submit_disabled']:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.subheader("Response:")
            st.write(response['answer'])
            st.write("")
            st.write("")
            st.subheader("Chat History")
            display_chat_history()


def display_chat_history():
    """
    Displays the chat history in a Streamlit expander.

    This function retrieves the session history and displays each message
    in the chat history. Human messages are prefixed with the current
    date and time, while AI messages are simply displayed with a separator.

    The chat history is shown within a collapsible expander labeled "Chat History".
    """
    with st.expander(label="Chat History"):
        session_history = get_session_history()
        if session_history.messages:
            for message in session_history.messages:
                if isinstance(message, HumanMessage):
                    now = datetime.datetime.now()
                    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"**You ({formatted_date_time}):** {message.content}")
                elif isinstance(message, AIMessage):
                    st.markdown(f"**AI:** {message.content}")
                    st.write("-----------------------")


def main():
    """
    Main function to initialize and run the RAG QA application.

    This function performs the following steps:
    1. Loads environment variables to retrieve the Groq API key.
    2. Creates embeddings for document processing.
    3. Initializes the application.
    4. Sets up the language model (LLM) using the Groq API key and specified model.
    5. Initializes session state for the application.
    6. Provides a file uploader for users to upload PDF documents.
    7. Processes the uploaded PDF documents using the created embeddings and LLM.

    Returns:
        None
    """
    groq_api_key = load_environment_variables()
    embeddings = create_embeddings()
    create_app()
    llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")
    initialize_session_state()
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        process_uploaded_files(uploaded_files, embeddings, llm)


if __name__ == "__main__":
    main()
