# pip install python-dotenv pinecone langchain-core langchain-pinecone langchain-community

import os
import csv
import uuid
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Load variables from .env file
load_dotenv()

# Access and print value
pinecone_api_key = os.getenv("PINECONE_API_KEY")
inference_api_key = os.getenv("INFERENCE_API_KEY")


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# creating index
pinecone = Pinecone(api_key=pinecone_api_key)

index_name = "suggession-index-hugging-face-augmented-100" 
existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

if index_name not in existing_indexes:
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

docs = []
uuids = []
with open("output100.csv", 'r') as f:
    reader = csv.reader(f)    
    for row in reader:
        metadata = {
        "id": str(uuid.uuid4()),        
        "category": row[0],
        "sub_category": row[1],
        "description": row[2],        
        "url": row[3]
        }        
        content = str({"category": row[0], "sub_category": row[1], "description": row[2], "detail_description": row[4]})
        docs.append(Document(page_content=content, metadata=metadata, id=metadata['id']))
        uuids.append(metadata['id'])        

# adding the data to vector store
vector_store.add_documents(documents=docs, ids=uuids)

# Testing vector store
results = vector_store.similarity_search(query="want to update name in Hub system",k=2)
retrievedContent=""
for res in results:
    # print(res.page_content)
    print(res.metadata)
    # retrievedContent+=res.page_content
    retrievedContent+=str(res.metadata)

print(retrievedContent)


#completed data prepration
print("#completed data prepration...ready to serve!")