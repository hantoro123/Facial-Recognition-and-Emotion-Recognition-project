from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import List, Union
import openai
from config import *
import streamlit as st

import json
import os
import pickle
from termcolor import colored
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

qna_dict = {}
def read_qna_dict_from_file():
    global qna_dict
    global qna_dict_file_path
    # Try to load the pickle file if it exists
    try:
        if os.path.exists(qna_dict_file_path):
            with open(qna_dict_file_path, 'rb') as qna_dict_file:
                qna_dict = pickle.load(qna_dict_file)
                print("qna_dict file loaded successfully.")
        else:
            print("qna_dict file not found. Initializing empty dictionary.")
            qna_dict = {}

    except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error occurred while reading the qna_dict file: {e}")
        qna_dict = {}    

def write_qna_dict_to_file():
    global qna_dict
    global qna_dict_file_path
    try:
        with open(qna_dict_file_path, 'wb') as qna_dict_file:
            pickle.dump(qna_dict, qna_dict_file)    
            print(f"QnA data successfully saved to {qna_dict_file_path}")
            #print(f"{qna_dict}")

    except FileNotFoundError:
        print(f"File not found: {qna_dict_file_path}")
    except IOError:
        print(f"IOError: Failed to save data to {qna_dict_file_path}")
    except pickle.PicklingError:
        print("PicklingError: Failed to pickle data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")        

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

def generate_chatgpt_answer(question, rough_answer, user_dict={}):
    chatgpt_answer_prepped = CHATGPT_ANSWER_PROMPT.format(
        QUESTION_HERE=question, ROUGH_ANSWER_HERE=rough_answer
    )    
# def generate_chatgpt_answer(question, rough_answer, user_dict={}):
    # chatgpt_answer_prepped = CHATGPT_ANSWER_PROMPT.format(
    #     QUESTION_HERE=question
    # )    

    if(user_dict != {}):
        user_defined_prompt = USER_DEFINED_PROMPT.format(
                NUM_EXPERIENCE=user_dict["years"], JOB_TITLE=user_dict["title"], COMPANY=user_dict["company"], 
        )
    else:
        user_defined_prompt = USER_DEFINED_PROMPT.format(
            NUM_EXPERIENCE=DEFAULT_EXPERIENCE, JOB_TITLE=DEFAULT_JOB_TITLE, COMPANY=DEFAULT_COMPANY 
        )    
    #print("user_defined_prompt", user_defined_prompt)

    final_answer = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_ANSWER_PROMPT+user_defined_prompt},
                  {"role": "user", "content": chatgpt_answer_prepped}],
        # messages=[{"role": "system", "content": SYSTEM_ANSWER_PROMPT+user_defined_prompt}],
        max_tokens=1000,
    )    
    return final_answer["choices"][0]["message"]["content"]

def generate_all_answers(directory):
    global qna_dict
    question = rough_answer = chatgpt_answer = ""
    for root, dirs, files in os.walk(directory):
        if "sample" in dirs:
            dirs.remove("sample")
        for dir in dirs:
            file_path =  os.path.join(root, dir) + '/'
            with open( file_path + ALL_FILE_NAMES[0], 'r') as file:
                question = file.read()                
                
            with open( file_path + ALL_FILE_NAMES[1], 'r') as file:
                rough_answer = file.read()

            with open( file_path + ALL_FILE_NAMES[2], 'r') as file:
                chatgpt_answer = file.read()    
                        
            if(dir in qna_dict):
                question_data = json.loads(qna_dict[dir])
                if(question != question_data["question"] or rough_answer != question_data["rough_answer"]):
                    #print("Call ChatGPT API only if there is change in question or rough answer", dir)
                    
                    print("Getting ChatGPT answer for", dir)
                    chatgpt_answer = generate_chatgpt_answer(question, rough_answer, )
                    # print(chatgpt_answer)

                    with open( file_path + ALL_FILE_NAMES[2], 'w') as file:
                        file.write(chatgpt_answer)
                    print(file_path + ALL_FILE_NAMES[2] + " updated")

            # update the dictionary
            json_data = {
                "question": question,
                "rough_answer": rough_answer,
                "chatgpt_answer": chatgpt_answer
            }
            qna_dict[dir] = json.dumps(json_data)     
            print("qna_dict updated for ", qna_dict[dir])            
                            
# Main application logic
def main():
    read_qna_dict_from_file()
    generate_all_answers("qna")
    write_qna_dict_to_file()

if __name__ == '__main__':
    main()

