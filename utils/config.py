import os
import json
import logging



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
USE_HEADLESS_BROWSER= True
LINKEDIN_MAX_PAGES_TO_LOAD_PER_JOB_TITLES= 20
LINKEDIN_SUBMIT_JOB_APPLICATION= False
LINKEDIN_APPLY_EASY_OPTION= True
LINKEDIN_APPLY_24_HOURS_FILTER= True
LINKEDIN_JD_VS_CV_THRESHOLD= 0
LINKEDIN_POST_TO_PROCESS= 9999
LINKEDIN_GET_POST_LINK= False

# models
q_type_models= [
        'model/fine_tuned_question_classifier_model_lite-default',
        "model/fine_tuned_question_classifier_model_lite-SGD_v2",
        ]
qa_type_models= [
    "model/fine_tuned_question_answer_model-base",
    "model/fine_tuned_question_answer_model-base-filtered",
    "model/fine_tuned_question_answer_model-small",
    ]
similarity_check_model= "all-MiniLM-L6-v2"
QUESTION_TYPES= [   
    "current_ctc",
    "expected_ctc",
    "personal_information",
    "education",
    "working_experience",
    "skills",
    "availability",
    "others",
    ]
JD_SIMILARITY_CHECK_WEIGHTS= {  #out of total 100
    "current_ctc": 0,
    "expected_ctc": 0,
    "personal_information": 0,
    "education": 15,
    "working_experience": 30,
    "skills": 40,
    "availability": 5,
    "others": 10,
    }


# Path variables
LOGGER_FOLDER = os.path.join(os.getcwd(), "Logger")
LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log.log")
# LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log_" + datetime.now().strftime('%Y-%m-%d') + ".log")
BROWSER_CAHCHE_FOLDER = os.path.join(os.getcwd(), "Browser_cahche")
TEMP_FOLDER = os.path.join(os.getcwd(), ".temp")
JOB_TABLE_FOLDER = os.path.join(os.getcwd(), "DataBase")
HTML_TABLE_FOLDER= os.path.join(os.getcwd(), "file_out")

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
os.makedirs(HTML_TABLE_FOLDER, exist_ok=True)



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



# rising errors for model data missmatch
for model_path in q_type_models:
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        id2label_values = list(config.get("id2label", {}).values())
        if sorted(id2label_values) != sorted(QUESTION_TYPES):
            print(f"Directory: {model_path}")
            raise ValueError(f"Mismatch in values for id2label and QUESTION_TYPES at config_path")
    else:
        raise FileNotFoundError(f"Config file not found for a model. {config_path} is missing.")
JD_SIMILARITY_CHECK_WEIGHTS_total= sum(JD_SIMILARITY_CHECK_WEIGHTS.values())
if JD_SIMILARITY_CHECK_WEIGHTS_total != 100:
    raise ValueError(f"Total weight in JD_SIMILARITY_CHECK_WEIGHTS is {JD_SIMILARITY_CHECK_WEIGHTS_total}, but it should be 100.")
if sorted(list(JD_SIMILARITY_CHECK_WEIGHTS.keys()))!= sorted(QUESTION_TYPES):
    raise ValueError(f"Some of the weights are missing at JD_SIMILARITY_CHECK_WEIGHTS.")

try:
    CV_DATA = {
        "availability": (lambda: open('Model_dataset/cv_data/availability.txt', 'r').read())(),
        "current_ctc": (lambda: open('Model_dataset/cv_data/current_ctc.txt', 'r').read())(),
        "education": (lambda: open('Model_dataset/cv_data/education.txt', 'r').read())(),
        "expected_ctc": (lambda: open('Model_dataset/cv_data/expected_ctc.txt', 'r').read())(),
        "others": (lambda: open('Model_dataset/cv_data/others.txt', 'r').read())(),
        "personal_information": (lambda: open('Model_dataset/cv_data/personal_information.txt', 'r').read())(),
        "skills": (lambda: open('Model_dataset/cv_data/skills.txt', 'r').read())(),
        "working_experience": (lambda: open('Model_dataset/cv_data/working_experience.txt', 'r').read())(),
        }
except Exception as e:
    raise e
