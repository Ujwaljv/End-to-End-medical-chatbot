from src.helper import load_pdf, text_split, download_huggingfaceembedding
import faiss
from langchain.vectorstores import FAISS



extracted_pdf = load_pdf("Data/")

text_chunks = text_split(extracted_pdf)
embedding = download_huggingfaceembedding()



def store_embeddings_faiss(embeddings, text_chunks):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS database saved successfully!")

store_embeddings_faiss(embedding, text_chunks)
vector_store = FAISS.load_local("faiss_index", embedding)

