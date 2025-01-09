import streamlit as st
import os
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from typing import Optional, List

# Load environment variables
load_dotenv()

# Set the GROQ API key from environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize the knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url)
)

# Load the knowledge base
knowledge_base.load()

# Initialize storage
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

# Function to initialize the assistant
def initialize_assistant(user: str, new_session: bool) -> Assistant:
    run_id: Optional[str] = None

    # Check for existing runs if not starting a new session
    if not new_session:
        existing_run_ids = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    # Initialize Assistant
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    return assistant

# Streamlit interface for chatbot
def main():
    st.title("PDF Assistant Chatbot")
    st.sidebar.title("Session Options")

    # User input for name and session choice
    user_name = st.sidebar.text_input("Enter your name:", value="user")
    new_session = st.sidebar.checkbox("Start a new session", value=False)

    st.sidebar.write("Click below to start:")
    if st.sidebar.button("Start Chat"):
        st.session_state.assistant = initialize_assistant(user_name, new_session)
        st.session_state.chat_history = []
        st.success("Chat session started!")

    if "assistant" not in st.session_state:
        st.info("Start a chat session using the sidebar.")
        return

    # Chatbot interface
    st.header("Chat with the Assistant")
    user_input = st.text_input("Enter your message:", value="")

    if st.button("Send"):
        if user_input.strip():
            assistant = st.session_state.assistant
            response = assistant.respond(user_input)
            st.session_state.chat_history.append((user_input, response))
            st.text_input("Enter your message:", value="", key="reset_input")
        else:
            st.warning("Please enter a message.")

    # Display chat history
    if "chat_history" in st.session_state:
        st.write("### Chat History")
        for user_message, bot_response in st.session_state.chat_history:
            st.markdown(f"**You:** {user_message}")
            st.markdown(f"**Assistant:** {bot_response}")

if __name__ == "__main__":
    main()
