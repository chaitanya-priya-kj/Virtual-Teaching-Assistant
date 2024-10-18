from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from htmlTemplates import css, user_template
from vector_store import vectorstore
from vector_store import get_vectorstore_from_folder
from IPython.display import YouTubeVideo
import openai
from streamlit_player import st_player
from packaging import version
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from scipy import spatial
from scipy.spatial import distance
import regex as re
import numpy as np
import base64
import os
import csv
import time
from piazza_api import Piazza
from bs4 import BeautifulSoup
from py2 import * 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open("assets/img_logo.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <style>
            @keyframes rotate-y {{
                0% {{ transform: rotateY(0deg); }}
                100% {{ transform: rotateY(360deg); }}
            }}
            .rotating-logo {{
                animation: rotate-y 8s linear infinite; /* Changed duration to 8s */
                transform-style: preserve-3d;
            }}
        </style>
        <img src="data:image/png;base64,{base64_image}" class="rotating-logo" width="60" height="60">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def embeddings_ranked_by_relatedness(query_embedding, df, top_n=2):
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1-D array.")

    embeddings = df['embeddings'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    
    cosines = []
    for row_embedding in embeddings:
        if row_embedding.ndim != 1:
            raise ValueError("Each row_embedding must be a 1-D array.")
        cosines.append(1 - distance.cosine(query_embedding, row_embedding))

    sorted_indices = sorted(range(len(cosines)), key=cosines.__getitem__, reverse=True)
    
    top_indices = sorted_indices[:top_n]
    top_embeddings = [embeddings[i] for i in top_indices]
    top_cosines = [cosines[i] for i in top_indices]

    return top_indices, top_embeddings, top_cosines

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Separate questions and answers
    questions = []
    answers = []
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            questions.append(message.content)
        else:
            answers.append(message.content)

    # Reverse the order of questions and answers
    reversed_questions = list(reversed(questions))
    reversed_answers = list(reversed(answers))

    # Display questions and answers in reverse order
    for question, answer in zip(reversed_questions, reversed_answers):
        st.write(user_template.replace("{MSG}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{MSG}", answer), unsafe_allow_html=True)
        #print(answers)


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_questions" not in st.session_state:
        st.session_state.chat_questions = []
    if "selected_chat_history_index" not in st.session_state:
        st.session_state.selected_chat_history_index = -1
    if "selected_chat_history" not in st.session_state:
        st.session_state.selected_chat_history = []

    st.set_page_config(page_title="Chat with Vector Store", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)
    directory = '../data/transcripts_new/'
    
    rotating_logo_html = """
        <style>
            @keyframes rotate-y {{
                0% {{ transform: rotateY(0deg); }}
                100% {{ transform: rotateY(360deg); }}
            }}
            .rotating-logo {{
                animation: rotate-y 8s linear infinite; /* Changed duration to 8s */
                transform-style: preserve-3d;
            }}
        </style>
        <img src="data:image/png;base64,{}" class="rotating-logo" width="80" height="80">
    """

    with open("assets/img_logo.png", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    st.sidebar.markdown(rotating_logo_html.format(base64_image), unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='margin-bottom:-30px; color: #CC9933;'>Data Science GPT</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h5 style='margin-bottom:-60px; color: #D3D3D3;'>Where Exploration Begins</h5>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Dropdown menu
    selected_option = st.sidebar.selectbox("Select an option", ["Piazza", "Ask any question"])

    if selected_option == "Piazza":
        piazza = login_to_piazza()
        dsci560 = piazza.network("lr7e73kounllq")
        posts_data = []
        process_posts(posts_data, dsci560)
        posts_df = pd.DataFrame(posts_data)
        posts_df.drop(posts_df[posts_df['Content'].str.len() == 0].index, inplace=True)
        posts_df.reset_index(drop=True, inplace=True)

        posts_df['GPT_reply'] = ""
        folder_path = 'study_material'
        vectorstore = get_vectorstore_from_folder(folder_path)
        conversation_chain = get_conversation_chain(vectorstore)
        process_similarity(posts_df, folder_path, "/Users/shubhamgujar/Desktop/dsci560/project/data/transcripts_new/")
        posts_df.to_csv("posts_with_paths_gpt_reply.csv", index=False)
        #posts_df = pd.read_csv("posts_with_paths_gpt_reply.csv")
        #st.dataframe(posts_df, width=1000)

        posts_df = pd.read_csv("posts_with_paths_gpt_reply.csv")

        # Display each column vertically for each row
        # for _, row in posts_df.iterrows():
        #     for col in posts_df.columns:
        #         if col == "img_path":
        #             img_path = row[col]
        #             if img_path:
        #                 st.image(img_path, caption='Image', use_column_width=True)
        #         # else:
        #         #     #st.write(f"**{col}:** {row[col]}")
                
        #         st.write(bot_template.replace("{MSG}", f"{col}: {row[col]}"), unsafe_allow_html=True)
        #     st.write("")

        for _, row in posts_df.iterrows():
            message = ""
            for col in posts_df.columns:
                if col == "img_path":
                    img_path = row[col]
                    if img_path:
                        st.image(img_path, caption='Image', use_column_width=True)
                else:
                    message += f"{col}: {row[col]}<br>"
            st.markdown(bot_template.replace("{MSG}", message), unsafe_allow_html=True)


    elif selected_option == "Ask any question":
        #st.write("You selected 'Ask any question'. Implement your functionality here.")
        try:
            folder_path = 'study_material'
            flag = 0
            vectorstore = get_vectorstore_from_folder(folder_path)
            
            if "conversation" not in st.session_state or st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(vectorstore)
            col1, col2 = st.columns([6.5, 3.5])
            with st.sidebar:
                for i, question in enumerate(st.session_state.chat_questions):
                    if st.button(f"Question {i+1}: {question}", key=i):
                        st.session_state.selected_chat_history_index = i
                        #st.session_state.selected_chat_history = st.session_state.chat_history[i]
            
            # Display the chat history in the main area
            style = """
                <style>
                .stTextInput {
                    width: 100%;
                    left: 25%;
                    right: 5%;
                }
                </style>
                """           
                    #st.session_state.chat_questions.append(user_question)
            with col1:
                #st.header("Data Buddies :robot_face:")
                # style = """
                #     <style>
                #     .stTextInput {
                #         position: fixed;
                #         bottom: 3rem;
                #         width: 70%;
                #         left: 25%;
                #         right: 5%;
                #     }
                #     </style>
                #     """
                style = """
                    <style>
                    .stTextInput {
                        width: 100%;
                        left: 25%;
                        right: 5%;
                    }
                    </style>
                    """
                if st.session_state.selected_chat_history_index != -1:
                    selected_history = st.session_state.chat_history[st.session_state.selected_chat_history_index]
                    try:
                        for message in selected_history:
                            if message['sender'] == "user":
                                st.markdown(user_template.replace("{MSG}", message['content']), unsafe_allow_html=True)
                            else:
                                st.markdown(bot_template.replace("{MSG}", message['content']), unsafe_allow_html=True)
                    except TypeError as e:
                        st.error(f"Failed to process message due to an incorrect data type: {e}")
                        st.write(f"Debug Info: {message}")  # Display the problematic message
                        for message in selected_history:
                            if message['sender'] == "user":
                                st.markdown(user_template.replace("{MSG}", message[1]), unsafe_allow_html=True)
                            else:
                                st.markdown(bot_template.replace("{MSG}", message[1]), unsafe_allow_html=True)


                else:
                    st.markdown("<h3 style='margin-bottom:-50px; color: #c69734;'>Data  Buddies</h3>", unsafe_allow_html=True)
                    user_question = st.text_input("", placeholder="Ask your queries  :)")
                    if user_question:
                        if user_question == "Change design" or user_question == "change design":
                            col1, col2 = st.columns([9, 1])
                            flag = 1
                        else:
                            handle_user_input(user_question)
                            st.session_state.chat_questions.append(user_question)

                if flag == 1:
                    folder_path = "../data/images_playlist/"
                    df = pd.read_csv(directory + "/final_embeddings.csv")
                    if user_question:
                        text_to_convert = user_question
                        question_embedding = get_embedding(text_to_convert)
                        top_indices, top_embeddings, similarities = embeddings_ranked_by_relatedness(question_embedding, df)
                        if max(similarities) > 0.3:
                            for i, index in enumerate(top_indices):
                                img_path = folder_path + df['video_id'].iloc[index] + "/" + df['Image Name'].iloc[index]
                                video_id = df['video_id'].iloc[index]
                                video_link = 'https://www.youtube.com/watch?v=' + video_id
                                if i==0: expanded = True
                                else: expanded=False
                                with st.expander(f"Reference Media {i+1}", expanded=expanded):
                                    st.image(img_path, caption=f'Lecture Frame: Stanford ML', use_column_width=True)
                                    #st.write(f"Youtube URL: {video_link}")
                                    pattern = r'(\d+)_(\d+\.?\d*)'  # Updated the regex pattern
                                    matches = re.search(pattern, df['Image Name'].iloc[index])
                                    if matches:
                                        start_time = float(matches.group(2)) / 1000  # Convert milliseconds to seconds
                                        #st.write(f"Watch video from {start_time} seconds")
                                        st.write("Video Player:")
                                        video_url = f"https://www.youtube.com/watch?v={video_id}&start={start_time}"
                                        st_player(video_url, key=f"video-{i+1}", height=150)
                                    else:
                                        st.write("Video watch duration not found!!")
                        else:
                            st.write("Related video not found.")
            if flag != 1:
                # Display the relevant video information in the second column
                with col2:
                    folder_path = "../data/images_playlist/"
                    df = pd.read_csv(directory + "/final_embeddings.csv")
                    if user_question:
                        text_to_convert = user_question
                        question_embedding = get_embedding(text_to_convert)
                        top_indices, top_embeddings, similarities = embeddings_ranked_by_relatedness(question_embedding, df)
                        if max(similarities) > 0.3:
                            for i, index in enumerate(top_indices):
                                img_path = folder_path + df['video_id'].iloc[index] + "/" + df['Image Name'].iloc[index]
                                video_id = df['video_id'].iloc[index]
                                video_link = 'https://www.youtube.com/watch?v=' + video_id
                                if i==0: expanded = True
                                else: expanded=False
                                with st.expander(f"Reference Media {i+1}", expanded=expanded):
                                    st.image(img_path, caption=f'Lecture Frame', use_column_width=True)
                                    #st.write(f"Youtube URL: {video_link}")
                                    pattern = r'(\d+)_(\d+\.?\d*)'  # Updated the regex pattern
                                    matches = re.search(pattern, df['Image Name'].iloc[index])
                                    if matches:
                                        start_time = float(matches.group(2)) / 1000  # Convert milliseconds to seconds
                                        #st.write(f"Watch video from {start_time} seconds")
                                        st.write("Video Player:")
                                        video_url = f"https://www.youtube.com/watch?v={video_id}&start={start_time}"
                                        st_player(video_url, key=f"video-{i+1}", height=150)
                                    else:
                                        st.write("Video watch duration not found!!")
                        else:
                            st.write("Related video not found.")
        except Exception as e:
            st.write(f"Error: {e}")

if __name__ == '__main__':
    client = OpenAI()
    main()