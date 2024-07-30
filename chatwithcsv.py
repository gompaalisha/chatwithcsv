import os
from dotenv import load_dotenv
import psycopg2
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory.buffer import ConversationBufferMemory
import pandas as pd
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set the API key directly from the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Initialize the language model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
except Exception as e:
    st.error(f"Failed to initialize language model: {e}")
    st.stop()

# Create conversation buffer memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Streamlit app
st.sidebar.title("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Create the agent for CSV interaction
    try:
        agent_executer = create_csv_agent(
            llm,
            uploaded_file,
            verbose=True,
            pandas_kwargs={"encoding": "unicode_escape"},
            allow_dangerous_code=True,
            memory=memory,  # Pass memory to the agent
        )
    except Exception as e:
        st.error(f"Failed to create agent: {e}")
        st.stop()

    # Define function for continuous chat
    def continuous_chat():
        st.title("Chat with CSV")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response from agent
            response = agent_executer.invoke(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # st.markdown(response)
                st.markdown(response["output"])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})

    # Start the continuous chat 
    continuous_chat()

else:
    st.write("PLEASE UPLOAD A CSV FILE TO START THE CHAT.")