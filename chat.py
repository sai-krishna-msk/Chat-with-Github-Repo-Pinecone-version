import os
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai
import pinecone
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st


load_dotenv()

#Langchain's defaul openai model(text-embedding-ada-002) and default embedding size
EMBEDDING_SIZE = 1536
index_name = os.environ.get('INDEX_NAME')
    

openai.api_key = os.environ.get('OPENAI_API_KEY')
pinecone.init(
            api_key=os.environ.get('PINECONE_API_KEY'), 
            environment=os.environ.get('PINECONE_REGION')  
        )

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(model='gpt-3.5-turbo')

db = Pinecone.from_existing_index(index_name=index_name ,embedding=embeddings)


def generate_response(prompt):
    # Generate a response using OpenAI's ChatCompletion API and the specified prompt
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])
    response = completion.choices[0].message.content
    return response


def get_text():
    # Create a Streamlit input field and return the user's input
    input_text = st.text_input("", key="input")
    return input_text


def search_db(query):
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model='gpt-3.5-turbo')
    # Create a RetrievalQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)


# Initialize the session state for generated responses and past inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = ['i am ready to help you ser']

if 'past' not in st.session_state:
    st.session_state['past'] = ['hello']

# Get the user's input from the text input field
user_input = get_text()

# If there is user input, search for a response using the search_chroma function
if user_input:
    output = search_db(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# If there are generated responses, display the conversation using Streamlit messages
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
