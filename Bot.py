import streamlit as st
import sqlite3
from time import sleep
import random
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import torch
import intel_extension_for_pytorch as ipex

# Create a connection to the database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT, password TEXT)''')
conn.commit()

# Import Intel-based model and tokenizer
model_name = "Intel/neural-chat-7b-v3-1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True, level="O1", auto_kernel_selection=True)

def create_user_table():
    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT, password TEXT)''')
    conn.commit()

def add_user(username, password):
    # Insert a new user into the table
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
              (username, password))
    conn.commit()

def authenticate_user(username, password):
    # Check if the user exists in the table
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))
    return c.fetchone()

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login 🔓"):
        user = authenticate_user(username, password)
        if user:
            st.success("Logged in as {}".format(user[0]))
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
        if new_username and new_password:
            add_user(new_username, new_password)
            sleep(random.randint(2, 6))
            st.success("Account created successfully")
            st.info("Please log in.")
        else:
            st.warning("Username and password are required")

def chatbot():
    st.title("AI powered Large-Language Model Chatbot 🤖💬")
    st.markdown('''
         - This may produce inaccurate information or recommedations
         - Knowledge on the parts may be limited and not upto date

         💡 Note: No Sign-in or API key required of using the bot!
    ''')

    if st.button("Logout 🔒"):
        st.session_state["logged_in"] = False
        st.info("Logged out successfully. Please log in.")

    if st.session_state.get("logged_in"):
        st.title("Chat")
        user_input = st.text_input("You:", "")

        if st.button("Send"):
            response = generate_response(user_input)
            st.text_area("Bot:", response, height=200)

def generate_response(user_input):
    system_input = "You are a computer parts recommendation assistant. Your mission is to help users find suitable computer parts for their needs."
    inputs = f"{system_input}\n{user_input}"
    inputs = tokenizer(inputs, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    create_user_table()

    st.title("PC Parts Recommender Bot🤖")
    st.title("Access and Registration")
    st.sidebar.title("Navigation 🧭")

    pages = {
        "Login": login,
        "Signup": signup
    }

    page = st.sidebar.radio("Pages", tuple(pages.keys()))

    if page in pages:
        pages[page]()

    if st.session_state.get("logged_in"):
        chatbot()

if __name__ == "__main__":
    main()
