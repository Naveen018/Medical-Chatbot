import os
from flask import Flask, render_template, jsonify, request
from src.helper import embed_chunks
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from src.prompt import generate_prompt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load existing chunks stored in vector db
embedded = embed_chunks()
index_name = "medicalbot"
existing_docs_embed = PineconeVectorStore.from_existing_index(
        index_name=index_name,  # Data/chunks will be stored inside this index
        embedding=embedded)


# Initialize LangChain Model
llm_model = ChatOpenAI(model="gpt-4o-mini", max_tokens=500)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST','GET'])
def chat():
    query = request.form['user_input']
    
    # Invoke the retriever to retrieve relevant docs
    retriever = existing_docs_embed.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    messages = generate_prompt(query, relevant_docs)
    
    bot_response = llm_model.invoke(messages)  # Generate response using LangChain
    print(bot_response.content)
    return render_template('index.html', user_input=query, bot_response=bot_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8090, debug=True)