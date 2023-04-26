import os
import subprocess
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import pinecone
import shutil





def clone_repository(repo_url, local_path):
    if(os.path.isdir(local_path)):
        print("Removing exsisting code repository !")
        shutil.rmtree(local_path)
    subprocess.run(["git", "clone", repo_url, local_path])


def load_docs(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(
                    dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def create_pinecone_index_ifnot_exsists(pinecone_obj, index_name):
    if index_name not in pinecone_obj.list_indexes():
        pinecone_obj.create_index(
            name=index_name,
            dimension=EMBEDDING_SIZE,
            metric='cosine'
        )


def main(repo_url, root_dir,index_name, pinecone_obj ):
    clone_repository(repo_url, root_dir)
    docs = load_docs(root_dir)
    texts = split_docs(docs)
    embeddings = OpenAIEmbeddings()
    create_pinecone_index_ifnot_exsists(pinecone_obj, index_name)
    _ = Pinecone.from_texts( texts=[t.page_content for t in texts],embedding=embeddings, index_name=index_name)


if __name__ == "__main__":
    load_dotenv()
    #Langchain's defaul openai model(text-embedding-ada-002) and default embedding size
    EMBEDDING_SIZE = 1536
    index_name = os.environ.get('INDEX_NAME')
    

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    pinecone.init(
            api_key=os.environ.get('PINECONE_API_KEY'), 
            environment=os.environ.get('PINECONE_REGION')  
        )

    repo_url = os.environ.get('REPO_URL')
    root_dir = "./gumroad"



    main(repo_url, root_dir, index_name, pinecone)
