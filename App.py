from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import streamlit as st
# import os

# load_dotenv()

repo_id = "facebook/bart-large-cnn"
llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature":0.5, "max_length":1000},
    huggingfacehub_api_token=st.secrets["api_token"]
)
text_splitter = CharacterTextSplitter()


def preprocessing_text(text_splitter, input_text):
    texts = text_splitter.split_text(input_text)
    docs = [Document(page_content=t) for t in texts]
    return docs


def summarize_text(llm, docs):
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    summary = chain.invoke(docs)
    return summary['output_text']


def main():
    st.title("Text Summarization 🤗")

    with st.form(key="summarize_form"):
        input_text = st.text_area('Enter the text to summarize (minimum 600 characters):')
        preprocessed_text = preprocessing_text(text_splitter, input_text)

        submitted = st.form_submit_button("Summarize")

        if submitted and len(input_text) >= 600:
            with st.spinner('Summarizing sentences...'):
                summary = summarize_text(llm, preprocessed_text)
            st.subheader("Summary:")
            st.write(summary)
        elif submitted and len(input_text) < 600:
            st.error("Please enter at least 600 characters of text.")
        elif not submitted and len(input_text) < 600:
            st.info("Minimum 600 characters of text required to summarize.")

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Powered by <a href='https://python.langchain.com/' target='_blank'>LangChain</a> and <a href='https://huggingface.co/facebook/bart-large-cnn' target='_blank'>BART Large CNN</a> </div>", unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()

