import streamlit as st
import random, os, json
import time

from PIL import Image
import cv2
import math 

from transcription import run_transcription_app, do_transcribe
from create_question_folders import create_folders_for_questions
from config import *
from evaluate_answer import evaluation_result
from generate_answers import generate_chatgpt_answer, qna_dict
# from auth import authenticate_user
from ultralytics import YOLO

import speech_recognition as sr
import os

st.set_page_config(
    page_title="Interview GPT",
    page_icon="📝",
)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    # else:
    #     print(f"Folder '{folder_path}' already exists.")

def create_folders_for_user_data():
    global g_recordings_folder, g_transcripts_folder
    #create_folder(qna_dict_folder)    
    create_folder(g_recordings_folder)    
    create_folder(g_transcripts_folder)
    create_folder(g_qna_folder)

def read_qna_data(single_folder=""):    
    global qna_dict
    question = rough_answer = chatgpt_answer = ""
    for root, dirs, files in os.walk(g_qna_folder):
        if dirs == []:
            continue        
        if single_folder != "":
            # Keep only the passed folder name and remove all other elements
            dirs = [element for element in dirs if element == single_folder]
            #print("dirs updated to ", dirs)

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

            with open( file_path + ALL_FILE_NAMES[3], 'r') as file:
                final_answer = file.read()    

            # update the dictionary
            json_data = {
                "question": question,
                "rough_answer": rough_answer,
                "chatgpt_answer": chatgpt_answer,
                "final_answer": final_answer
            }
            qna_dict[dir] = json.dumps(json_data)     
            #print("qna_dict updated for ", qna_dict[dir])       
    #print("qna_dict generated.")        

def reset_fields():    
    st.session_state.rough_answer_text = "" 
    st.session_state.chatgpt_answer_text = ""
    st.session_state.final_answer_text = ""


def get_user_confirmation():
    st.write('Are you sure you want to save this answer? This will overwrite the existing saved answer.')
    col1, col2 = st.columns([.2, 1])

    with col1:
        yes_btn = st.button('Yes', key='yes')
    with col2:
        no_btn = st.button('No', key='no')

    # Confirmation buttons
    if yes_btn:
        st.success('Confirmed!')
        print("Clicked Yes")
        return True
    if no_btn:
        st.error('Cancelled!')
        print("Clicked No")
        return False
    return False

def save_answer_to_file(directory, folder, filename, content=""):    
    file_path =  directory + '/' + folder + '/' + filename 
    print("Saving to file ", file_path)
    with open(file_path, 'w') as file:
        #print("content, ", content)
        file.write(content)
    # update dictionary as well for the saved folder
    read_qna_data(folder) 

def display_qna_widgets(user_dict):

    selected_option = '문항 생성'

    # =========================================================================================== #
    rough_answer = st.text_area("자기소개서를 입력해 주세요:", value=st.session_state.rough_answer_text, height=200)
    col1, col2 = st.columns([.2, 1])

    with col1:
        rough_answer_submit = st.button('Submit', key='rough_answer')
    with col2:
        rough_answer_save = st.button('Save', key='rough_answer_save')
    
    if rough_answer_submit:
        chatgpt_answer = generate_chatgpt_answer(st.session_state.selected_question, rough_answer, user_dict)
        st.session_state.rough_answer_text = rough_answer
        st.session_state.chatgpt_answer_text = chatgpt_answer

    if rough_answer_save:        
        st.session_state.rough_answer_text = rough_answer 
        save_answer_to_file(g_qna_folder, selected_option, ALL_FILE_NAMES[1], rough_answer)
    # =========================================================================================== #
    chatgpt_answer = st.text_area("ChatGPT가 생성한 문항:", value=st.session_state.chatgpt_answer_text, height=200)    
    
    col1, col2 = st.columns([.2, 1])

    with col1:
        chatgpt_answer_copy = st.button('Copy', key='chatgpt_answer')
    with col2:
        chatgpt_answer_save = st.button('Save', key='chatgpt_answer_save')        

    if chatgpt_answer_copy:
        st.session_state.chatgpt_answer_text = chatgpt_answer
        st.session_state.final_answer_text = chatgpt_answer
    
    # if chatgpt_answer_save:
    #     st.session_state.chatgpt_answer_text = chatgpt_answer 
    #     save_answer_to_file(g_qna_folder, selected_option, ALL_FILE_NAMES[2], chatgpt_answer)


# questions_sample = ['오늘 뭐해요?', '집이 어디인가요?', '점심 뭐 드셨나요?']
def display_main_content():
    """Render the main page of the application."""

    user_dict = {}
    # Show user input fields
    col1, col2, col3 = st.columns([1, .5, 1])
    with col1:
        user_dict["title"] = st.text_input('직무', DEFAULT_JOB_TITLE)
    with col2:
        user_dict["years"] = st.text_input('경력', DEFAULT_EXPERIENCE)
    with col3:
        user_dict["company"] = st.text_input('지원 회사', DEFAULT_COMPANY)    
    st.markdown("---")

    if st.session_state.selected_question:
        st.header(st.session_state.selected_question)
        
        display_qna_widgets(user_dict)       
        run_transcription_app(g_recordings_folder)        

        if st.button('Analyze', key='analyze', disabled=st.session_state.get("analyze_button_disable", True)):
            transcription = do_transcribe(g_recordings_folder, g_transcripts_folder)
            selected_option = '문항 생성'
            start_time = time.time()
            eval_result = evaluation_result(g_qna_folder, transcription, selected_option, st.session_state.final_answer_text, user_dict)
            end_time = time.time()
            time_taken = int(end_time - start_time)
            st.write(eval_result)
            st.write(f"*Time taken: {time_taken} seconds*")
            st.session_state.analyze_button_disable = True            

def init_session_state():
    """Initialize the session state variables."""
    if 'initialization_done' not in st.session_state:
        st.session_state.initialization_done = False

    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = '자기소개서를 입력해 주세요'

    if 'prev_selected_question' not in st.session_state:
        st.session_state.prev_selected_question = '문항 생성'

    if 'random_button_pressed' not in st.session_state:
        st.session_state.random_button_pressed = False

    if 'analyze_button_disable' not in st.session_state:
        st.session_state.analyze_button_disable = True        

    if 'rough_answer_text' not in st.session_state:
        st.session_state.rough_answer_text = ""    

    if 'chatgpt_answer_text' not in st.session_state:
        st.session_state.chatgpt_answer_text = ""

    if 'final_answer_text' not in st.session_state:
        st.session_state.final_answer_text = ""    



def emotion_score(score:50.0,emotion):
    score = score
    good, bad = 0, 0
    threshold = 0.1

    good += (threshold * (emotion.count('happy')) + threshold * (emotion.count('normal')))
    bad -=  (threshold * (emotion.count('anger')) + threshold * (emotion.count('embarrass')) + threshold * (emotion.count('anxiety')) + threshold * (emotion.count('pain')) + threshold * (emotion.count('sad')))
    print(emotion)
    print(score+good+bad)
    return score+good+bad


def main():
    global g_qna_folder, g_recordings_folder, g_transcripts_folder

    classes = {
    0:'anger',
    1:'normal',
    2:'happy',
    3:'embarrass',
    4:'anxiety',
    5:'pain',
    6:'sad'
                }

    init_session_state()    
    unique_user_id = 'ssm'
    g_qna_folder = "qna/" + unique_user_id
    g_recordings_folder = "recordings/" + unique_user_id
    g_transcripts_folder = "transcripts/" + unique_user_id

    create_folders_for_questions(g_qna_folder)
    create_folders_for_user_data()
    read_qna_data()
    
    st.session_state.initialization_done = True
    st.title("모의 면접 프로그램")       

    display_main_content()

    st.title("모의 면접")
    run = st.checkbox('Run')
    confidence = float(40)
    if run:
        interview_score = 0
        emotion = []


        # FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([]) 
        video_stream = st.image([], channels='RGB')

        # YOLO 모델 초기화
        model = YOLO('./best.pt')
        st.write("실시간 감정 분석")
        while True:
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            if not ret:
                st.error('웹캠에서 프레임을 읽을 수 없습니다.')
                camera.release()
                break

            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classes[cls])
                    if interview_score:
                        interview_score = emotion_score(score=interview_score,emotion=classes[cls])
                    else:
                        interview_score = emotion_score(score=50,emotion=classes[cls])
                    print(interview_score)
                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(frame, classes[cls], org, font, fontScale, color, thickness)
                    video_stream.image(frame)
            
            # # 웹캠 캡처 해제
            # camera.release()
            # cv2.destroyAllWindows()
            # print(emotion)
            # interview_score = emotion_score(emotion)
            # st.write(interview_score)
    else:
        st.write('Stopped')

if __name__ == '__main__':
    main()