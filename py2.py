import pandas as pd
import csv
import time
from piazza_api import Piazza
from bs4 import BeautifulSoup
import time
from openai import OpenAI
from scipy.spatial import distance
import regex as re
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from vector_store import vectorstore
from vector_store import get_vectorstore_from_folder
import pandas as pd
import gc
import streamlit as st

api_key = "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"  # Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

os.environ['KMP_DUPLICATE_LIB_OK']='True'

client = OpenAI(api_key='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')

# Function to clean HTML syntax from columns in a dictionary
def clean_html_columns(data_dict):
    cleaned_dict = {}
    for key, value in data_dict.items():
        soup = BeautifulSoup(value, 'html.parser')
        cleaned_dict[key] = soup.get_text()
    return cleaned_dict

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def embeddings_ranked_by_relatedness(query_embedding, df, top_n=2):
    """Returns top n embeddings ranked by cosine similarity."""
    # Convert query_embedding to a NumPy array if it's not already one
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    
    # Ensure query_embedding is 1-D
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1-D array.")

    embeddings = df['embeddings'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    print(embeddings.shape)
    cosines = []
    for row_embedding in embeddings:
        # Ensure row_embedding is 1-D
        if row_embedding.ndim != 1:
            raise ValueError("Each row_embedding must be a 1-D array.")
        cosines.append(1 - distance.cosine(query_embedding, row_embedding))

    sorted_indices = sorted(range(len(cosines)), key=cosines.__getitem__, reverse=True)
    
    top_indices = sorted_indices[:top_n]
    top_embeddings = [embeddings[i] for i in top_indices]
    top_cosines = [cosines[i] for i in top_indices]

    return top_indices, top_embeddings, top_cosines

def clear_cache():
    gc.collect()
    
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def handle_piazza_questions(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    bot_responses = ""
    for i, message in enumerate(chat_history):
        if i % 2 != 0:
            bot_responses = f"{message.content}\n"
    
    return bot_responses

def login_to_piazza():
    p = Piazza()
    p.user_login(email='sgujar@usc.edu', password='Shubham10102023#')
    return p

def process_posts(posts_data, dsci560):
    for post in dsci560.iter_all_posts(limit =20):
        try:
            if post['type'] == 'question' and \
                    not post['children'] and \
                    post['status'] == 'active' and \
                    'unanswered' in post['tags']:
                cleaned_data = clean_html_columns({
                    'Subject': post['history'][0]['subject'],
                    'Content': post['history'][0]['content']
                })
                posts_data.append({
                    'Post ID': post['id'],
                    'NR': post['nr'],
                    'Subject': cleaned_data['Subject'],
                    'Content': cleaned_data['Content']
                })
                append_to_csv(post, cleaned_data)
        except Exception as e:
            print("Error occurred:", e)
            print("Waiting for a second before retrying...")
            time.sleep(1)
            continue
        time.sleep(1)

def append_to_csv(post, cleaned_data):
    with open('posts_data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Post ID', 'NR', 'Subject', 'Content'])
        writer.writerow({
            'Post ID': post['id'],
            'NR': post['nr'],
            'Subject': cleaned_data['Subject'],
            'Content': cleaned_data['Content']
        })

def process_similarity(posts_df, folder_path, directory):
    for i in range(len(posts_df['Content'])):
        df = pd.read_csv(directory + "/final_embeddings.csv")
        text_to_convert = posts_df['Content'][i]
        question_embedding = get_embedding(text_to_convert)
        top_indices, top_embeddings, similarities = embeddings_ranked_by_relatedness(question_embedding, df)
        
        posts_df['GPT_reply'][i] = handle_piazza_questions(posts_df['Content'][i], conversation_chain)

        if max(similarities) > 0.2:
            for index in top_indices:
                img_path = folder_path + df['video_id'].iloc[index] + "/" + df['Image Name'].iloc[index]
                video_link = 'https://www.youtube.com/watch?v='+ df['video_id'].iloc[index]
                pattern = r'(\d+)_([\d.]+)'
                matches = re.search(pattern, df['Image Name'].iloc[index])
                try:
                    if matches:
                        start_time = matches.group(1)
                        end_time = matches.group(2)
                        posts_df.loc[i, 'image_path'] = img_path
                        posts_df.loc[i, 'video_path'] = video_link
                        posts_df.loc[i, 'watch_from'] = f"Watch video from {start_time} to {end_time}"
                    else:
                        posts_df.loc[i, 'watch_from'] = "Video watch duration not found!!"
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue
        else:
            posts_df.loc[i, 'video_path'] = "Related video not found."

piazza = login_to_piazza()
dsci560 = piazza.network("lr7e73kounllq")
posts_data = []
process_posts(posts_data, dsci560)
posts_df = pd.DataFrame(posts_data)
posts_df.drop(posts_df[posts_df['Content'].str.len() == 0].index, inplace=True)
posts_df.reset_index(drop=True, inplace=True)
print(len(posts_df))
posts_df['GPT_reply'] = ""
folder_path = 'study_material'
vectorstore = get_vectorstore_from_folder(folder_path)
conversation_chain = get_conversation_chain(vectorstore)
process_similarity(posts_df, folder_path, "/Users/shubhamgujar/Desktop/dsci560/project/data/transcripts_new/")
posts_df.to_csv("posts_with_paths_gpt_reply.csv", index=False)