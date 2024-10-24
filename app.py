'''__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')'''

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.document_loaders import PyPDFLoader
import os

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    clean_chunks = [{'content': chunk} if isinstance(chunk, str) else chunk for chunk in chunks]
    return chunks
    

def create_embeddings(chunks):
    contents = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', dimensions=1536)
    vector_store = Chroma.from_documents(contents, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    #from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.4, convert_system_message_to_human=True)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    setup = RunnableParallel(context=retriever, question=RunnablePassthrough())
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    chain = setup| prompt | llm
    answer = chain.invoke(q)
    return answer.content 

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    

if __name__ == "__main__":

    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image("img.png")
    st.subheader('LLM Question and Answer Application 🦜🔗')

    with st.sidebar:
        api_key = st.text_input('Google API Key :key::', type='password')
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a File :spiral_note_pad:', type=['pdf', 'txt', 'docx'])
        chunk_size = st.number_input('Chunk Size', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input("K", min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Readding, Chinking and embedding file...:rocket:'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./',uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'chunk_size :{chunk_size},chunks: {len(chunks)}' )

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploded and chunked')

    q = st.text_input('Ask a question about the content of your file :question:')
    button = st.button('Enter:star2:')
    if button:
        with st.spinner('In Progress...:rocket:'):
            if q:
                if "vs" in st.session_state:
                    vector_store = st.session_state.vs
                    st.write(f'k {k}')

                    answer = ask_and_get_answer(vector_store, q,k)

                    st.text_area('LLM Answer:', value=answer)
        


                    st.divider()
                    if 'history' not in st.session_state:
                        st.session_state.history = ''
                    value = f'Q: {q} \n A: {answer}'
                    st.session_state.history = f'{value}, \n {"-"*100}, \n {st.session_state.history}'
                    h = st.session_state.history
                    st.text_area(label='chat History', value=h, key='history', height=400)

