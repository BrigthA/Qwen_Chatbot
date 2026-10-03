import streamlit as st
import ollama

# Page configuration
st.set_page_config(
    page_title="Qwen Chat - Ollama Streamlit",
    page_icon="🤖",
    layout="centered"
)

# App header
st.title("💬 Qwen Local Chat UI")
st.caption("Powered by Streamlit and Ollama")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Qwen Model",
        ["qwen2.5:0.5b", "qwen2.5", "qwen3", "qwen2.5-coder"],
        index=0,
    )
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input from chat box
if prompt := st.chat_input("Type your message here..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = ollama.chat(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,
            )
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error connecting to Ollama: {e}"
            message_placeholder.markdown(full_response)
            
        # Append assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": full_response})