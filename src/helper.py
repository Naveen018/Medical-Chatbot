from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def doc_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    doc_chunks = text_splitter.split_documents(extracted_data)
    return doc_chunks

def embed_chunks():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
    return embeddings