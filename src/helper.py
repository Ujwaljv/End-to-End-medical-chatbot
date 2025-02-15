
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def text_split(extracted_pdf):
   text_spliter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
   text_chunks=text_spliter.split_documents(extracted_pdf)
   
   return text_chunks

def download_huggingfaceembedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding