from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from utils.config import logging
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import json
import gc
logger = logging.getLogger(__name__)
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
q_type_models= [
        'model/fine_tuned_question_classifier_model_lite-default',
        'model/fine_tuned_question_classifier_model_lite-Adam',
        'model/fine_tuned_question_classifier_model_lite-AdamW',
        'model/fine_tuned_question_classifier_model_lite-SGD']
qa_type_model= "t5-large"
similarity_check_model= "all-MiniLM-L6-v2"
# Load CV data
try: 
    with open("Model_dataset/cv.json", "r") as file:
        CV_DATA= json.load(file)
except FileNotFoundError:
    logger.error("CV file not found.")
except Exception as e:
    logger.error("Unable to load the CV file ")




def get_question_type_prediction(text):
    # Load model:
    try: 
        QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
        QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
        ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label
        if not all(item in ID2LABEL.values() for item in QUESTION_TYPES) and all(ID2LABEL[key] in QUESTION_TYPES for key in ID2LABEL):
            logger.critical("Question classifier model doesn't match the class provided")
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            QUESTION_CLASSIFIER_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                pass
        QUESTION_CLASSIFIER_MODEL.eval()
    except FileNotFoundError:
        logger.error("Question classifier model folder not found.")
    except Exception as e:
        logger.error(f"Unable to load the Question classifier model, error: {e}")
    
    # Prediction
    inputs = QUESTION_CLASSIFIER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt").to(QUESTION_CLASSIFIER_MODEL.device)
    # print(f"\n\n\nQUESTION_CLASSIFIER_MODEL_DEVICE: {QUESTION_CLASSIFIER_MODEL.device}\n\n\n")
    with torch.no_grad():
        outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1)
    id2label = QUESTION_CLASSIFIER_MODEL.config.id2label
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    return id2label[predicted_classes.item()]


def get_model_out_raw(question):
    # Load model
    try: 
        QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_type_model, use_cache=True)
        QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_type_model, legacy= False)
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            QUESTION_ANSWER_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                pass
        QUESTION_ANSWER_MODEL.eval()
    except FileNotFoundError:
        logger.error("Question answer model folder not found.")
    except Exception as e:
        logger.error(f"Unable to load the Question answer model, error: {e}")

    # Make predictions
    predicted_question_type= get_question_type_prediction(question)
    context= CV_DATA[predicted_question_type]
    input_text = f"question: {question} context: {context}"
    inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt").to(QUESTION_ANSWER_MODEL.device)
    outputs = QUESTION_ANSWER_MODEL.generate(input_ids=inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    # print(f"\n\n\nQUESTION_ANSWER_MODEL_DEVICE: {QUESTION_ANSWER_MODEL.device}\n\n\n")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    return predicted_question_type, QUESTION_ANSWER_TOKENIZER.decode(outputs[0], skip_special_tokens=True)


def get_most_similar_option(text, available_options):
    # Load model
    try: 
        similarity_check_model= "all-MiniLM-L6-v2"
        SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_model)
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            SIMILARITY_CHECK_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                pass
    except FileNotFoundError:
        logger.error("Similarity check model folder not found.")
    except Exception as e:
        logger.error(f"Unable to load the Similarity check model, error: {e}")
    
    # Make predictions
    device = SIMILARITY_CHECK_MODEL.device
    text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
    options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
    cosine_similarities = cosine_similarity(text_embedding, options_embeddings)
    most_similar_index = cosine_similarities.argmax()
    most_similar_option = available_options[most_similar_index]
    # similarity_score = cosine_similarities[0][most_similar_index]
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    return most_similar_option


def predict_ans(question_dict):
    question= question_dict["question"]
    # pre_ans= question_dict["pre_ans"]
    input_type= question_dict["input_type"]
    available_options= question_dict["available_options"]
    predicted_ans= None
    predicted_question_type, raw_predicted_ans= get_model_out_raw(question)
    if input_type in ("select", "radio"):
        predicted_ans= get_most_similar_option(raw_predicted_ans, available_options)
        return predicted_question_type, predicted_ans
    
    # make sure to remove once the model is predicting as intended
    raw_predicted_ans= int(''.join(re.findall(r'\d+', raw_predicted_ans))) if re.findall(r'\d+', raw_predicted_ans) else raw_predicted_ans
    return predicted_question_type, raw_predicted_ans

