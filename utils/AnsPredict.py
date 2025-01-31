from utils.config import *
import random
import pandas as pd
import ast
import re
logger = logging.getLogger(__name__)



def get_question_type_prediction(text):
    inputs = QUESTION_CLASSIFIER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = QUESTION_CLASSIFIER(**inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1)
    id2label = QUESTION_CLASSIFIER.config.id2label
    return id2label[predicted_classes.item()]


def get_model_out_raw(question):
    predicted_question_type= get_question_type_prediction(question)
    context= CV_DATA[predicted_question_type]
    input_text = f"question: {question} context: {context}"
    inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt")
    outputs = QUESTION_ANSWER_MODEL.generate(input_ids=inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)

    return predicted_question_type, QUESTION_ANSWER_TOKENIZER.decode(outputs[0], skip_special_tokens=True)

def get_most_similar_option(raw_predicted_ans, available_options):
    """TODO"""
    return available_options[0]

def predict_ans(question_dict):
    question= question_dict["question"]
    # pre_ans= question_dict["pre_ans"]
    input_type= question_dict["input_type"]
    available_options= question_dict["available_options"]
    predicted_ans= None
    predicted_question_type, raw_predicted_ans= get_model_out_raw(question)
    if input_type in ("select", "radio"):
        predicted_ans= get_most_similar_option(raw_predicted_ans, available_options)
        predicted_ans= int(predicted_ans[0]) if re.findall(r'\d+', predicted_ans) else predicted_ans
        return predicted_question_type, predicted_ans
    raw_predicted_ans= int(raw_predicted_ans[0]) if re.findall(r'\d+', raw_predicted_ans) else raw_predicted_ans
    return predicted_question_type, raw_predicted_ans

 