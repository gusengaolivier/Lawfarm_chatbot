import streamlit as st 



# Title of the App

st.title("Lawyer's Assistant")


#Text Area input
st.subheader("Enter the text below")
query = st.text_area("Enter Text","Type Here ...")

# Button
btn = st.button("Generate ")

# Condition
if btn and query:
    st.write("Generating ...")
    st.write("Generated Successfully")
    


