import streamlit as st
from PIL import Image
from rag import NawalicRAG

# Title for the app
st.title("Nawalic's Politics App")

# User input
user_input = st.text_input("Enter some data")

# Submit button
if st.button('Submit'):
    # Display the PDF file
    rag = NawalicRAG(
        ocr_data_dir="ocr_data/",
        document_metadata_dir="metadata/",
        vector_data_dir="vecotr_store/",
        top_k=3,
    )
    response = rag.query(user_input)
    st.markdown('<p style="color:blue; font-weight: bold;">Summary</p>', unsafe_allow_html=True)
    st.write(response.response)
    st.markdown('<p style="font-weight: bold;">Metadata</p>', unsafe_allow_html=True)
    for node in response.source_nodes:
        st.write(node.metadata)
        st.write(f"score : {node.score:.3f}")