
# @TODO: Add the streaming of models
# @TODO: Add hugging face implementation


# Description: Main application file for the Streamlit app
from dotenv import load_dotenv # Load environment variables
from langchain_core.messages import AIMessage, HumanMessage # AI Message & User Message
from langchain_core.runnables import RunnablePassthrough # Runnable Passthrough
from langchain_core.output_parsers import StrOutputParser # String Output Parser
from langchain_core.prompts import ChatPromptTemplate # Chat Message Prompt Template
from langchain_community.utilities import SQLDatabase # Custom SQLDatabase class for connecting to MySQL
import streamlit as st # Streamlit library for building web applications
from langchain_openai import ChatOpenAI # OpenAI API for generating responses
from langchain_groq import ChatGroq # Groq API for generating responses

# Load environment variables
load_dotenv() # Load environment variables from .env file


stakeholder = "CEO"

# Function to initialize the database
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)


# SQL Chain
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interracting with the CEO who wants to get some insights from the company's database.
    Based on the table schema below, write a SQL query that will help you get the required information. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the query in any text, not even backticks or quotes.

    For example:
    Question: Get the total number of customers in the database.
    SQL Query: SELECT COUNT(*) FROM customers;
    Question: Name Which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
   
    your turn:

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    #llm = ChatGroq(model="mixtral-8x7b-32768")  # Load the Groq model
    llm = ChatOpenAI(model="gpt-4o") # Load the GPT model # model = "gpt-4-0125-preview"

    def get_schema(_):
        return db.get_table_info()
    
    return(
        RunnablePassthrough.assign(schema = get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


# Function to get the response bigger chain
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User Question: {question}
    SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(model="mixtral-8x7b-32768")  # Load the Groq model
    #llm = ChatOpenAI(model="gpt-3.5-turbo") # Load the GPT-3.5 model # model = "gpt-4-0125-preview"
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


# Check if chat history exists in the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello there! I am you SQL Assistant. How can I help you today?"),
    ]
 

st.set_page_config(page_title="Chat with MySQL", page_icon="🚀", layout="wide")
st.title("Chat with MySQL")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application that uses MySQL as the database.")

    st.text_input("Host", value = "localhost", key="Host")
    st.text_input("Port", value = "3306", key="Port")
    st.text_input("User", value = "root", key="User")
    st.text_input("Password", type = "password", value = "azelpi-admin", key="Password")
    st.text_input("Database", value = "Chinook", key="Database")
    if st.button("Connect"):
        with st.spinner("Connecting to the database..."):
            db = init_database(
                user = st.session_state["User"],
                password = st.session_state["Password"],
                host = st.session_state["Host"],
                port = st.session_state["Port"],
                database = st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to the database")

# Messages container
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
         with st.chat_message("Human"):
            st.markdown(message.content)

# Chat input
user_query = st.chat_input("Enter your message here...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(user_query)) # Add user message to chat history

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)


    st.session_state.chat_history.append(AIMessage(content = response)) # Add AI message to chat history

