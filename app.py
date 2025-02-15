from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, download_huggingfaceembedding
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate  
from langchain.chains import RetrievalQA
from src.prompt import *
import os
import faiss
from langchain.vectorstores import FAISS

app = Flask(__name__)

extracted_pdf = load_pdf("Data/")

text_chunks = text_split(extracted_pdf)
embedding = download_huggingfaceembedding()


def store_embeddings_faiss(embeddings, text_chunks):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS database saved successfully!")

store_embeddings_faiss(embedding, text_chunks)
vector_store = FAISS.load_local("faiss_index", embedding)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"max_new_tokens":512,
                            'temperature':0.8})
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
chain_type_kwargs = {"verbose": True}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True)