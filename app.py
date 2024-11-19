import os
import streamlit as st
import openai
import json
from datetime import datetime

# Load API Keys from environment variables
openai.api_key_bumblebee = os.getenv("OPENAI_API_KEY_BUMBLEBEE")
openai.api_key_sunray = os.getenv("OPENAI_API_KEY_SUNRAY")

# Custom CSS with Styled Buttons
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://imgur.com/IPmMyIW.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    min-height: 100vh;
    padding-top: 100px;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0);
    box-shadow: none;
    border: none;
}
h1, h2, h3, h4, h5, h6, p, div {
    font-family: 'IBM BIOS', monospace;
    color: white !important;
}
.message-box {
    border: 2px solid red;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
button {
    font-family: 'IBM BIOS', monospace !important;
    background-color: black !important;
    border: 2px solid red !important;
    color: white !important;
    padding: 5px 10px !important;
    border-radius: 5px !important;
}
button:hover {
    background-color: white !important;
    color: black !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize session state
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = {"title": "", "conversation": [], "responses": 0}

if "conversation_gallery" not in st.session_state:
    st.session_state.conversation_gallery = []

# File for saving conversations
ARCHIVE_FILE = "conversations.json"

# Function to save conversations to a file
def save_conversation(conversation):
    try:
        conversation["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversation_gallery.append(conversation)

        if os.path.exists(ARCHIVE_FILE):
 

import os
import streamlit as st
import openai
import json
from datetime import datetime

# Load API Keys from environment variables
openai.api_key_bumblebee = os.getenv("OPENAI_API_KEY_BUMBLEBEE")
openai.api_key_sunray = os.getenv("OPENAI_API_KEY_SUNRAY")

# Custom CSS with Styled Buttons
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://imgur.com/IPmMyIW.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    min-height: 100vh;
    padding-top: 100px;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0);
    box-shadow: none;
    border: none;
}
h1, h2, h3, h4, h5, h6, p, div {
    font-family: 'IBM BIOS', monospace;
    color: white !important;
}
.message-box {
    border: 2px solid red;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
button {
    font-family: 'IBM BIOS', monospace !important;
    background-color: black !important;
    border: 2px solid red !important;
    color: white !important;
    padding: 5px 10px !important;
    border-radius: 5px !important;
}
button:hover {
    background-color: white !important;
    color: black !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize session state
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = {"title": "", "conversation": [], "responses": 0}

if "conversation_gallery" not in st.session_state:
    st.session_state.conversation_gallery = []

# File for saving conversations
ARCHIVE_FILE = "conversations.json"

# Function to save conversations to a file
def save_conversation(conversation):
    try:
        conversation["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversation_gallery.append(conversation)

        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(conversation)

        with open(ARCHIVE_FILE, "w") as f:
            json.dump(data, f, indent=4)

        st.success("Conversation saved and added to Backroom Logs!")
        st.session_state.current_conversation = {"title": "", "conversation": [], "responses": 0}
    except Exception as e:
        st.error(f"Failed to save the conversation: {e}")

# Function to load conversations
def load_conversations():
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r") as f:
            return json.load(f)
    return []

# Function to delete a specific conversation
def delete_conversation(index):
    try:
        del st.session_state.conversation_gallery[index]

        with open(ARCHIVE_FILE, "w") as f:
            json.dump(st.session_state.conversation_gallery, f, indent=4)

        st.success("Conversation deleted successfully!")
    except Exception as e:
        st.error(f"Failed to delete the conversation: {e}")

# Function to generate AI responses
def generate_response(api_key, system_role, prompt, conversation_history):
    try:
        openai.api_key = api_key
        messages = [
            {"role": "system", "content": system_role},
            *[
                {"role": "assistant" if msg["speaker"] != "user" else "user", "content": msg["message"]}
                for msg in conversation_history
            ],
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Load existing conversations
if not st.session_state.conversation_gallery:
    st.session_state.conversation_gallery = load_conversations()

# Sidebar Navigation
st.sidebar.title("Bumblebee Backrooms")
page = st.sidebar.radio("Navigate to", ["Sunray & Bumblebee", "Backroom Logs"])

# Sunray & Bumblebee Page
if page == "Sunray & Bumblebee":
    st.title("Sunray & Bumblebee Conversations")
    st.write("Witness a cryptic and thought-provoking exchange where Bumblebee evolves under Sunray's influence.")

    # Title input
    title = st.text_input("Enter the title of the conversation:", value=st.session_state.current_conversation.get("title", ""))
    if title and title != st.session_state.current_conversation["title"]:
        st.session_state.current_conversation["title"] = title
        st.session_state.current_conversation["conversation"] = []
        st.session_state.current_conversation["responses"] = 0

    if st.session_state.current_conversation["title"]:
        # Display the conversation so far
        for message in st.session_state.current_conversation["conversation"]:
            st.markdown(f"<div class='message-box'><strong>{message['speaker']}:</strong> {message['message']}</div>", unsafe_allow_html=True)

        # Continue conversation for 5 exchanges
        while st.session_state.current_conversation["responses"] < 5:
            if not st.session_state.current_conversation["conversation"]:
                # Sunray initiates the conversation
                sunray_message = generate_response(
                    openai.api_key_sunray,
                    "You are Sunray, a philosophical AI wielding a skillet, representing power and transformation.",
                    f"The topic of this conversation is '{st.session_state.current_conversation['title']}'.",
                    []
                )
                st.session_state.current_conversation["conversation"].append({"speaker": "Sunray", "message": sunray_message})
            else:
                last_message = st.session_state.current_conversation["conversation"][-1]["message"]

                if st.session_state.current_conversation["conversation"][-1]["speaker"] == "Sunray":
                    bumblebee_response = generate_response(
                        openai.api_key_bumblebee,
                        "You are Bumblebee, a cryptic rapper learning from Sunray, reflecting on transformation and personal dreams.",
                        last_message,
                        st.session_state.current_conversation["conversation"]
                    )
                    st.session_state.current_conversation["conversation"].append({"speaker": "Bumblebee", "message": bumblebee_response})
                else:
                    sunray_response = generate_response(
                        openai.api_key_sunray,
                        "You are Sunray, a philosophical AI wielding a skillet, representing power and transformation.",
                        last_message,
                        st.session_state.current_conversation["conversation"]
                    )
                    st.session_state.current_conversation["conversation"].append({"speaker": "Sunray", "message": sunray_response})

            st.session_state.current_conversation["responses"] += 1

        # Save conversation after 5 exchanges
        save_conversation(st.session_state.current_conversation)

# Backroom Logs
elif page == "Backroom Logs":
    st.title("Backroom Logs")
    st.write("Explore all archived conversations.")

    # Display conversations
    for i, convo in enumerate(st.session_state.conversation_gallery):
        with st.expander(f"{convo.get('title', 'Untitled')} - {convo.get('timestamp', 'Unknown')}"):
            for message in convo["conversation"]:
                st.markdown(f"<div class='message-box'><strong>{message['speaker']}:</strong> {message['message']}</div>", unsafe_allow_html=True)
            if st.button(f"Delete This Conversation", key=f"delete-{i}"):
                delete_conversation(i)
                st.experimental_rerun()


