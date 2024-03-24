import streamlit as st
from time import sleep
import random
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import torch
import intel_extension_for_pytorch as ipex

# Import Intel-based model and tokenizer
model_name = "Intel/neural-chat-7b-v3-1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True, level="O1", auto_kernel_selection=True)

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login 🔓"):
        # Placeholder for actual authentication logic
        if username == "demo" and password == "demo":
            st.success("Logged in as {}".format(username))
            with st.spinner("Loading Chatbot"):
                sleep(random.randint(2, 6))
            st.session_state["logged_in"] = True
            return True
        else:
            st.error("Invalid username or password")
    return False

def signup():
    st.subheader("Create New Account")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")

    if st.button("Sign Up 📝"):
        # Placeholder for actual signup logic
        if new_username and new_password:
            st.success("Account created successfully")
            st.info("Please log in.")
        else:
            st.warning("Username and password are required")

def chatbot():
    st.title("AI powered Large-Language Model Chatbot 🤖💬")
    st.markdown('''
         - This may produce inaccurate information or recommendations
         - Knowledge on the parts may be limited and not up-to-date
         💡 Note: No Sign-in or API key required to use the bot!
    ''')

    if st.button("Logout 🔒"):
        st.session_state["logged_in"] = False
        st.info("Logged out successfully. Please log in.")

    if st.session_state.get("logged_in"):
        st.title("Chat")
        
        # Accept user input using st.text_input
        user_input = st.text_input("You:", "")
        
        if user_input:  # Check if user_input is not empty
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Append the dialogue history to the user's prompt
            dialogue_history = "\n".join([message["content"] for message in st.session_state.messages])
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_input)
            # Display assistant response in chat message container
            with st.spinner('Generating response....'):
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    try:
                        for response in generate_response(dialogue_history):
                            full_response += response
                            message_placeholder.markdown(full_response + "▌")
                            sleep(0.01)
                        message_placeholder.markdown(full_response)

                        # Check if there are follow-up questions
                        if "?" in user_input:
                            # Update the chat history with the assistant's response
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            # Clear the chat input box
                            user_input = ""
                            # Set the chat input box value to the assistant's response
                            st.text_input("Follow-up question", value=full_response)

                        # Update the chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"An error occurred during response generation: {str(e)}")
                        # Update the chat history with the error message
                        st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

def generate_response(user_input):
    system_input = "You are a computer parts recommendation assistant. Your mission is to help users find suitable computer parts for their needs."
    inputs = f"{system_input}\n{user_input}"
    inputs = tokenizer(inputs, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("PC Parts Recommender Bot🤖")
    st.sidebar.title("Navigation 🧭")

    pages = {
        "Login": login,
        "Signup": signup
    }

    page = st.sidebar.radio("Pages", tuple(pages.keys()))

    if page in pages:
        pages[page]()

    if st.session_state.get("logged_in"):
        # Initialize chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
        chatbot()

if __name__ == "__main__":
    main()
