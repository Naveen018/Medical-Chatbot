from langchain.schema import SystemMessage, HumanMessage

def generate_prompt(query, relevant_docs):
    # Create a prompt to send to LLM along with retrieved docs
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )

    messages = [
        SystemMessage(content="You are a Medical assistant."),
        HumanMessage(content=combined_input),
    ]

    return messages
