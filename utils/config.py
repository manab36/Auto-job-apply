import os
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
LINKEDIN_MAX_PAGES_TO_LOAD_PER_JOB_TITLES= 5
LINKEDIN_POST_TO_PROCESS= 10000
CURRENT_CTC= 517000
EXEPCTED_CTC= 850000
EMAIL_ID= "mboro497@gmail.com"
PH_NO= "9101925089"



# Path variables
LOGGER_FOLDER = os.path.join(os.getcwd(), "Logger")
LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log.log")
# LOGGER_FILE = os.path.join(LOGGER_FOLDER, "basic_log_" + datetime.now().strftime('%Y-%m-%d') + ".log")
BROWSER_CAHCHE_FOLDER = os.path.join(os.getcwd(), "Browser_cahche")
TEMP_FOLDER = os.path.join(os.getcwd(), "temp")
JOB_TABLE_FOLDER = os.path.join(os.getcwd(), "DataBase")

# File variables
LINKEDIN_DB_FILE = os.path.join(JOB_TABLE_FOLDER, 'Linkedin.db')
LINKEDIN_JOB_DETAILS_TABLE = 'raw_job_details'
LINKEDIN_FORM_QA_TABLE = 'raw_question_ans'
LINKEDIN_POSTS_DETAILS_TABLE= 'raw_post_details'
LINKEDIN_FORM_QA_GT_values= os.path.join(JOB_TABLE_FOLDER, 'form_gt_values.csv')
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




