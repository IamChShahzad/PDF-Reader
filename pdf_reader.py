import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from concurrent.futures import ProcessPoolExecutor
import os

#write chatgpt APi key: 
os.environ['OPENAI_API_KEY'] = '##################'      


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_pdf_text_parallel(pdf_docs):
    with ProcessPoolExecutor() as executor:
        text_chunks = list(executor.map(get_pdf_text, pdf_docs))
        return "".join(text_chunks)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": k})
    )
    return chain

    # memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory= memory,
    # )
    # return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
  
  
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
   
    if "chat_history" not in st.session_state:
        st.session_state.chat_history =[]
   
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about pdf documents: ")
    if user_question:
        k_value = st.session_state.k_value if "k_value" in st.session_state else 3
        conversation_chain = st.session_state.conversation or get_conversation_chain(user_question, st.session_state.vectorstore, k = k_value)
        handle_userinput(user_question, conversation_chain, st.session_state.chat_history)



    st.write(user_template.replace("{{MSG}}", "Hello rebot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your pdf and press and click process", accept_multiple_files=True)
        if pdf_docs:
            k_value = len(pdf_docs)
            st.session_state.k_value = k_value
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(user_question,vectorstore, k = k_value)  



if __name__ == '__main__':
    main()