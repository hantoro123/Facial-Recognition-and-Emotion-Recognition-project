import streamlit as st
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



st.markdown(
        """
        #### Interview GPT는 최고의 기업에서의 행동 면접에 대비하는 데 도움을 주는 생성형 AI 어플리케이션입니다.


    """
    )