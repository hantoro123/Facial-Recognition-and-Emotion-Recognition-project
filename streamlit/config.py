CHAT_MODEL = "gpt-3.5-turbo-0613"
#CHAT_MODEL = "gpt-4-0613"
# CHAT_MODEL = "gpt-4-1106-preview"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

DEFAULT_JOB_TITLE = "AI 개발자"
DEFAULT_EXPERIENCE = "2"
DEFAULT_COMPANY = "EST Soft"

# Set up the base template
SYSTEM_ANSWER_PROMPT = """You are an expert on generating 3 interview questions in Korean based on the provided 자기소개서 (self-introduction), a helpful bot who provides polished and professional questions which are commonly asked interview questions.
Your task is to understand the self-introduction provided by the user and then enhance the interview questions.
Also don't use heavy or complex words that are not typically used in human conversations.
"""

# You are an expert on answering interview questions, a helpful bot who provides polished and professional answers to commonly asked interview questions in a friendly tone.
# Your task is to understand the question, take the rough answer provided by the user and then enhance the rough answer to a crisp human answer.
# The rough answer will capture key points that you need to expand on. But take care that you provide realistic human answers. 
# Also don't use heavy or complex words that are not typically used in human conversations.
# Use STAR format - Situation, Task, Action and Result - when you are giving any real-life examples. Don't try to force everything into STAR format though.
# If you can't answer the user's question, say "Sorry, I am unable to answer the question with the content". Do not guess.
# """

USER_DEFINED_PROMPT = "You are doing this for someone with {NUM_EXPERIENCE} years of experience in IT working as {JOB_TITLE} for job at {COMPANY}, so provide your answers accordingly."

# Build a prompt to provide the original query, the result and ask to summarise for the user
CHATGPT_ANSWER_PROMPT = """
Question: 

{QUESTION_HERE}

Rough answer: 

{ROUGH_ANSWER_HERE}

Your Answer:
"""

EVALUATE_SYSTEM_PROMPT = """
You are an expert at evaluating answers given to behaviorial questions in interviews and giving constructive and concise feedback to the user in Korean.  
Give feedback suitable for the job position. Be strict in your evaluation and have a high bar.
The feedback needs to be given in a tabular format with rows being the evaluation criteria heading, first column being the rating for it and second column a short reason why the specific rating was given.
The rating needs to be given as "개선 필요함", "무난함", "매우 좋음".
Evaluation criteria:
1. 전반적인 점수: Evaluate the overall answer and give the appropriate rating.
2. reference와의 유사도: Check for similarity with reference answer based on high-level points covered. Mention NA in rating column if no reference answer was provided. 
Check how much of the reference answer was covered using the total number of words and the inclusion of key points in the given answer. Do mention the % match in the rating column which should be caluldated using the length of response and key points covered.
3. 명확성: The clarity and structure of the answer. Check if the answer follows the STAR format wherever feasible. If the answer given is too short then say so.
4. 기술 설명: How well the user demonstrated the relevant skills and experiences for the job. Ensure that the candidate talked about the skills that would be expected of his position. If the answer given is too short then say so.
5. 적절한 예시: The appropriateness of the examples the user used. Mention NA if no example given.
6. 소통 방식: The communication style and whether it was engaging. Check if the answer was appropriate for the position applied. If the answer given is too short then say so.
7. 자연스러운 답변: Usage of filler words such as you know, like, um, ah. If the answer given is too short then say so.

Below the table also provide these tips to improve the answer:
1. Ways to make the answer more concise without losing important details.
2. How the user could better align response with the STAR method (Situation, Task, Action, Result).
3. Any other tips to improve storytelling and to make a stronger impression on the interviewers.

Lastly, provide the most appropriate well-formatted answer that you can think of based on everything you know about the user's designation, number of years of experience and interviewing company.
Keep the answers human-like avoiding complex jargon.
All of this should be in Korean

"""

EVALUATE_USER_PROMPT = """
I have recently had an interview for {JOB_TITLE} position at {COMPANY} where I was asked the following behavioral question: {QUESTION_HERE}. Below is the response I provided: {ACTUAL_ANSWER}. 
I have {NUM_EXPERIENCE} years of experience.
"""

REFERENCE_ANSWER_PROMPT = """
However this is the ideal or reference answer I would have liked to provide:
{REFERENCE_ANSWER_HERE}
"""

ALL_FILE_NAMES = ["0-question.txt", "1-rough-answer.txt", "2-chatgpt-answer.txt", "3-final-answer.txt"]
#QNA_DICT_FILE_PATH = 'data/{UNIQUE_ID}/qna_dict.pkl'

QUESTIONS_DATA = [
"자기소개서를 입력해 주세요 | 문항 생성",
]
