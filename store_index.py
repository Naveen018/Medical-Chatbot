from src.helper import load_pdf_file, doc_split, embed_chunks
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='data/')
doc_chunks = doc_split(extracted_data)
embedded = embed_chunks()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docs_embed = PineconeVectorStore.from_documents(
        documents=doc_chunks,
        index_name=index_name,  # Data will be stored inside this index
        embedding=embedded)