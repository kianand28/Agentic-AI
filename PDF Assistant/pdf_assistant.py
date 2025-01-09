import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve GROQ API Key from the environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define PostgreSQL URL for database connection
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Set up the knowledge base with PDF document and vector database
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url)
)

# Load the knowledge base (downloads and processes the PDF)
knowledge_base.load()

# Set up assistant storage with PostgreSQL
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

# Function for starting the assistant with optional new session
def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # Check if an existing run ID exists
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    # Initialize the assistant
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,  
        read_chat_history=True,
    )

    # If no run ID, start a new session
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Run the assistant's CLI interface
    assistant.cli_app(markdown=True)

# Run the application with Typer CLI
if __name__ == "__main__":
    typer.run(pdf_assistant)
