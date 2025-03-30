import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import tempfile
from PyPDF2 import PdfReader


# Initialize Streamlit app
st.title("PDF Question Answering App")

# OpenAI API key
OPENAI_key='Enter your openAPI key'
PINECONE_API_KEY='Enter PineconeAPI key'
os.environ["OPENAI_API_KEY"] = OPENAI_key
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = "Update with your Pinecone environment"

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text = text_splitter.split_documents(pages)
    st.write(f"Split into {len(text)} text chunks.")
    st.write("Initializing Pinecone...")
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), environment=PINECONE_ENV)
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {e}")
        st.stop()
    index_name = 'app'
    if index_name not in pc.list_indexes():
        #index = pc.create_index(index_name, dimension=768, metric='cosine')
        st.write(f"Index '{index_name}' created successfully.")
    else:
        index = pc.Index(index_name)  # Connect to the existing index
        st.write(f"Index '{index_name}' already exists.")
    index = pc.Index(index_name)
    st.write(index.describe_index_stats())
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_key)
    from langchain.vectorstores import Pinecone
    st.write("Creating document search index...")
    docsearch = Pinecone.from_documents(text, embeddings, index_name=index_name)
    doc_prompt = PromptTemplate(
        template="Content: {page_content}\nSource: {source} \n Page:{page}", # look at the prompt does have page#
        input_variables=["page_content", "source","page"],)
    
    retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')
    def parse_response(response):
        st.write(response['result'])
        st.write('\n\nSources:')
        for source_name in response["source_documents"]:
            st.write(source_name.metadata['source'], "page #:", source_name.metadata['page']+1)
            
    llm_src=OpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    qa_chain = create_qa_with_sources_chain(llm_src)
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    prompt_template = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    llm_chain = LLMChain(llm=llm_src, prompt=PROMPT)
    qa_chain = RetrievalQA.from_chain_type(llm=llm_src,
                                      chain_type="stuff",
                                      retriever=retriever,
                                      return_source_documents=True)
    final_qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain, # Pass the LLMChain, not the RetrievalQA object
        document_variable_name='context',
        document_prompt=doc_prompt,
    )
    retrieval_qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=final_qa_chain
    )
    
    def chatbot_response(query):
        if query:
            st.write("Processing query...")
            response = qa_chain(query)
            parse_response(response)
        #return parse_response

    #user_input = st.text_input("You:", "")

    def main():
        st.title("Chatbot")
        st.write("Type 'exit' to end the chat.")
    
        # Initialize session state for tracking conversation
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'exit_flag' not in st.session_state:
            st.session_state.exit_flag = False
    
        # Input widget for new query
        query = st.text_input("Enter your question:", key="new_query")
        if st.button("Send"):
            if query.lower() == "exit":
                st.write("Chatbot: Thank you!")
                st.session_state.exit_flag = True
            else:
                st.session_state.conversation.append(("You", query))
                chatbot_response(query)

    
    if __name__ == "__main__":
        main()
