from utils.config import *
import random
import pandas as pd
import ast
logger = logging.getLogger(__name__)


pre_submitted_data= pd.DataFrame([])
try:
    pre_submitted_data= pd.read_csv(LINKEDIN_FORM_QA_GT_values)
except FileNotFoundError:
    logger.warning("Pre answered file missing")
except Exception as e:
    logger.error("Error accessing pre answered file")



def __predict_ans(question_dict):
    print(question_dict)
    print(type(question_dict))
    input_type= question_dict["input_type"]
    if input_type == 'select':
        return 1
    elif input_type == 'radio':
        return random.choice(question_dict["available_options"])
    elif input_type in ['text']:
        return "2"
    elif input_type== 'email':
        return "mboro497@gmail.com"
    elif input_type== 'tel':
        return "2"



def predict_ans(question_dict):
    input_type= question_dict["input_type"]
    question= question_dict["question"]
    # available_options= ast.literal_eval(question_dict["available_options"])
    available_options= question_dict["available_options"]
    pre_ans= ''
    if not pre_submitted_data.empty:
        pre_ans= pre_submitted_data[pre_submitted_data["question"] == question]

    if input_type == 'select':
        if not pre_submitted_data.empty:
            if len(pre_ans)>0:
                pre_ans= pre_ans.iloc[0]["pre_ans"]
                if pre_ans in available_options:
                    logger.warning(f"select index: {available_options.index(pre_ans)}")
                    return available_options.index(pre_ans)
        return predict_select_type_ans(question, available_options)
            
    elif input_type == 'radio':
        if not pre_submitted_data.empty:
            if len(pre_ans)> 0:
                pre_ans= pre_ans.iloc[0]["pre_ans"]
                if pre_ans in available_options:
                    logger.warning(f"select index: {pre_ans}")
                    return pre_ans
        return predict_radio_type_ans(question, available_options)
    
    elif input_type in 'text':
        if not pre_submitted_data.empty:
            if len(pre_ans)> 0:
                logger.warning(f"select index: {pre_ans.iloc[0]['pre_ans']}")
                return pre_ans.iloc[0]["pre_ans"]
        return predict_text_type_ans(question, available_options)

    elif input_type== 'email':
        return "EMAIL_ID"
    

def predict_text_type_ans(question, available_options):
    return '2'

def predict_select_type_ans(question, available_options):
    return 2

def predict_radio_type_ans(question, available_options):
    return random.choice(available_options)


