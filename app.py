import streamlit as st
from PIL import Image
from main import basic_retirever
# Title for the app
st.title("Nawalic's Politics App")

# User input
user_input = st.text_input("Enter some data")

# Submit button
if st.button('Submit'):
    # Display the PDF file
    retrived_nodes = basic_retirever(user_input)
    for key,retirieved_node in retrived_nodes.items():
        st.write(f"{key} results:")
        for node in retirieved_node:
            try:
                page = node.metadata.get("page_number",None)
                if page:
                    st.write(f"page: {page} score: {node.score}")
                else:
                    st.write(node.metadata["kg_rel_texts"])
            except Exception as e:
                breakpoint()
                print("error")