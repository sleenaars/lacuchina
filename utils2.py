#file that contains all the functions used (utils.py)
from openai import OpenAI
from pinecone import Pinecone
import PyPDF2

pc = Pinecone() 
client = OpenAI()



def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small")
    return response.data[0].embedding 




# Step 1: Embed user query
def get_query_embedding(query, client):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small" )
    
    return response.data[0].embedding

# Step 2: Search Pinecone
def search_pinecone(query, index, client, top_k=3):
    query_embedding = get_query_embedding(query, client)
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True )
    
    return result.matches


# Global conversation memory
conversation_history = []

# Step 3: Build prompt for OpenAI
def build_prompt(query, results, conversation_history):
    context = "\n\n".join([
        f"{i+1}. {match['metadata'].get('text', '')}"
        for i, match in enumerate(results)
    ])

    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in conversation_history
        ])

    prompt = f"""
You are an AI assistant. Please give a polite answer based on the information you received"

{f'Conversation so far:\n{history_text}' if history_text else ''}

Context:
{context}

Question: {query}
Answer:
"""
    return prompt

# Step 4: Get response from Chat Completion API
def answer_query(query, index, client):
    global conversation_history

    results = search_pinecone(query, index, client)
    prompt = build_prompt(query, results, conversation_history)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on context."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = completion.choices[0].message.content

    # Save conversation
    conversation_history.append({
        "question": query,
        "answer": answer
    })

    return answer

