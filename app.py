import os
from flask import Flask, render_template, jsonify, request, session
from src.helper import embed_chunks
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.prompt import generate_prompt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

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

def get_or_create_memory():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def create_memory_from_history(chat_history):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    for message in chat_history:
        if message['type'] == 'human':
            memory.chat_memory.add_user_message(message['content'])
        else:
            memory.chat_memory.add_ai_message(message['content'])
    return memory

@app.route('/')
def home():
    # Clear any existing session when starting fresh
    session.clear()
    return render_template('index.html')

@app.route('/chat', methods=['POST','GET'])
def chat():
    query = request.form['user_input']
    
    # Get or create chat history
    chat_history = get_or_create_memory()
    
    # Create memory from history
    memory = create_memory_from_history(chat_history)
    
    # Create the conversation chain with the session's memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=existing_docs_embed.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )
    
    # Get response using the conversation chain
    result = qa_chain({"question": query})
    bot_response = result["answer"]
    
    # Update session chat history
    chat_history.append({"type": "human", "content": query})
    chat_history.append({"type": "ai", "content": bot_response})
    session['chat_history'] = chat_history
    
    return render_template('index.html', 
                         user_input=query, 
                         bot_response=bot_response,
                         chat_history=chat_history)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8090, debug=True)