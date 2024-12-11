#pip install langchain-huggingface langchain_community langchain-pinecone pinecone
import os

from pinecone import Pinecone
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Load variables from .env file
load_dotenv()

# Access and print value
pinecone_api_key = os.getenv("PINECONE_API_KEY")
inference_api_key = os.getenv("INFERENCE_API_KEY")

app = Flask(__name__)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

pinecone = Pinecone(api_key=pinecone_api_key)

index_name = "suggession-index-hugging-face-augmented-100" 
existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

if index_name not in existing_indexes:
    raise Exception("index does not exist")

index = pinecone.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)



results = vector_store.similarity_search(query="want to update name in Hub system",k=2)
retrievedContent=""
for res in results:
    # print(res.page_content)
    print(res.metadata)
    # retrievedContent+=res.page_content
    retrievedContent+=str(res.metadata)

# print(retrievedContent)


#completed data prepration
print("#completed data prepration...ready to serve!")



inferenceClient = InferenceClient(api_key=inference_api_key)

def getFormattedMessage(content, query):
    return f"""
                    You are part of a customer support ticketing system.

                    User Query: "{query}"

                    Here are details of the support ticket data from VectorDB in json:
                    {content}

                    Write response as given below:

                    1) If the data from the vectorDB does not match with User Query give not match found 
                    response with "Not matching response"
                    2) If the data form VectorDB matches to User Query
                    Respond as JSON in the following format without any prefix: [{{
                        "category": "string",
                        "sub-category": "string",
                        "description": "string",
                        "url": "string"
                        }}]
                    3) Given the best match response in the expected format                    
                   """


# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This assumes index.html is in the 'templates' folder

# API endpoint that receives a GET request
@app.route('/api', methods=['GET'])
def api():
    # Get the query parameter from the URL
    query = request.args.get('query', default="")
    print("user question: " + query)
    results = vector_store.similarity_search(query=query,k=5)
    
    retrievedContent=""
    for res in results:
        # print(res.page_content)
        print(res.metadata)
        retrievedContent+=str(res.metadata)

    #start our RAG code
    messages = [
        { "role": "user", "content": query },
        { "role": "system", "content": getFormattedMessage(retrievedContent, query) } 
    ]
    stream = inferenceClient.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct", 
    	messages=messages, 
    	temperature=0.5,
    	max_tokens=1024,
    	top_p=0.7,
    	stream=False
    )
    
    answer=stream.choices[0].message.content

    print("answer: " + answer)
    #complete our RAG code


    response = {
        "ragresponse": f"{retrievedContent}",
        "llmresponse": f"{answer}"
    }

    return jsonify(response)


if __name__ == '__main__':
   app.run(debug=True)
