import os
import logging
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import json



# Change what type of job needs to be searched
JOB_TITLES = [
    'Data Engineer',
    'Python Developer',
    'Python Engineer',
    'Software Engineer',
    'Software Developer',
    'Data Scientist',
]
JOB_LOCATIONS = 'India'
LINKEDIN_MAX_PAGES_TO_LOAD_PER_JOB_TITLES= 50
LINKEDIN_POST_TO_PROCESS= 1000
LINKEDIN_GET_POST_LINK= True
LINKEDIN_GET_SUBMIT_JOB_APPLICATION= False
LINKEDIN_APPLY_EASY_OPTION= True
LINKEDIN_APPLY_24_HOURS_FILTER= False

# prediction Model
QUESTION_TYPES= [   
    "current_ctc",
    "expected_ctc",
    "personal_information",
    "education",
    "working_experince",
    "skills",
    "availability",
    "others",
    ]
with open("Model_dataset/cv.json", "r") as file:
    CV_DATA= json.load(file)


#Models
q_type_model= 'model/fine_tuned_question_classifier_model_lite-AdamW'
# MOCEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUESTION_CLASSIFIER = DistilBertForSequenceClassification.from_pretrained(q_type_model)
QUESTION_CLASSIFIER.eval()
# QUESTION_CLASSIFER_MODEL.to(MOCEL_DEVICE)  # Move model to GPU if available

qa_type_model= "t5-large"
QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_type_model)
QUESTION_ANSWER_MODEL.eval()
QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_type_model, legacy= False)

QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_model)
ID2LABEL = QUESTION_CLASSIFIER.config.id2label


# Path variables
LOGGER_FOLDER = os.path.join(os.getcwd(), "Logger")
LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log.log")
# LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log_" + datetime.now().strftime('%Y-%m-%d') + ".log")
BROWSER_CAHCHE_FOLDER = os.path.join(os.getcwd(), "Browser_cahche")
TEMP_FOLDER = os.path.join(os.getcwd(), ".temp")
JOB_TABLE_FOLDER = os.path.join(os.getcwd(), "DataBase")

# File variables
LINKEDIN_DB_FILE = os.path.join(JOB_TABLE_FOLDER, 'Linkedin.db')
LINKEDIN_JOB_DETAILS_TABLE = 'raw_job_details'
LINKEDIN_FORM_QA_TABLE = 'raw_question_ans'
LINKEDIN_POSTS_DETAILS_TABLE= 'raw_post_details'
# String formats
# LOGGER_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s'
LOGGER_FORMAT = '%(asctime)s - %(name)s - %(funcName)s - line %(lineno)d - %(levelname)s - %(message)s'

DB_FILE_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'  # datetime.now()

# Creating folders if not exists
os.makedirs(LOGGER_FOLDER, exist_ok=True)
os.makedirs(BROWSER_CAHCHE_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(JOB_TABLE_FOLDER, exist_ok=True)



# Suppress urllib3 logs
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)
# Disable WebDriver Manager logs by setting the level to WARNING or higher
logging.getLogger('webdriver_manager').setLevel(logging.WARNING)
logging.getLogger('selenium').setLevel(logging.WARNING)

# Configure application logging
logging.basicConfig(
    filename= LOGGER_FILE,
    level= logging.DEBUG,  # Adjust this to your desired logging level for application-specific logs
    format= LOGGER_FORMAT
)
# DEBUG < INFO < WARNING < ERROR < CRITICAL




