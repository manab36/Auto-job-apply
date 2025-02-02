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
qa_type_models= [
    ".temp/model_results/fine_tuned_question_answer_model-base/checkpoint-2960",
    ".temp/model_results/fine_tuned_question_answer_model-base/checkpoint-4440",
    ".temp/model_results/fine_tuned_question_answer_model-base-temp/checkpoint-740",
    "model/fine_tuned_question_answer_model-base",
    "model/fine_tuned_question_answer_model-base-filtered",
    ]
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
    # Load model & tokenizer inside function (so it's released later)
    result= None
    QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
    ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        QUESTION_CLASSIFIER_MODEL.to(device)
        QUESTION_CLASSIFIER_MODEL.eval()
        inputs = QUESTION_CLASSIFIER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1)
        result = ID2LABEL[predicted_classes.item()]
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory! Consider using a smaller batch size or switching to CPU.")
    finally:
        # Cleanup: Release memory after execution
        del QUESTION_CLASSIFIER_MODEL, QUESTION_CLASSIFIER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
    return result


def get_qa_model_out_raw(question):
    # Load model & tokenizer inside function (so they are released after execution)
    QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_type_models[-1])
    QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_type_models[-1], legacy=False)
    predicted_question_type= None
    result= None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        QUESTION_ANSWER_MODEL.to(device)
        QUESTION_ANSWER_MODEL.eval()
        # Get predicted question type
        predicted_question_type = get_question_type_prediction(question)
        context = CV_DATA.get(predicted_question_type, "")
        input_text = f"question: {question} context: {context}"
        inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = QUESTION_ANSWER_MODEL.generate(
                input_ids=inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True
            )
        result = QUESTION_ANSWER_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory! Consider reducing input size or using CPU.")
        result = None
    finally:
        # Cleanup: Release memory after execution
        del QUESTION_ANSWER_MODEL, QUESTION_ANSWER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
    return predicted_question_type, result


def __get_most_similar_option(text, available_options):
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


def get_most_similar_option(text, available_options):
    # Load model & tokenizer inside function (so they are released after execution)
    SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_model)
    most_similar_option= None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        SIMILARITY_CHECK_MODEL.to(device)
        SIMILARITY_CHECK_MODEL.eval()
        # Get predicted option
        device = SIMILARITY_CHECK_MODEL.device
        text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
        options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
        cosine_similarities = cosine_similarity(text_embedding, options_embeddings)
        most_similar_index = cosine_similarities.argmax()
        most_similar_option = available_options[most_similar_index]
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory! Consider reducing input size or using CPU.")
        result = None
    finally:
        # Cleanup: Release memory after execution
        del SIMILARITY_CHECK_MODEL
        torch.cuda.empty_cache()
        gc.collect()
    return most_similar_option


def predict_ans(question_dict):
    question= question_dict["question"]
    # pre_ans= question_dict["pre_ans"]
    input_type= question_dict["input_type"]
    available_options= question_dict["available_options"]
    predicted_question_type, raw_predicted_ans= get_qa_model_out_raw(question)
    filtered_predicted_ans= raw_predicted_ans
    if input_type in ("select", "radio"):
        filtered_predicted_ans= get_most_similar_option(raw_predicted_ans, available_options)
    return predicted_question_type, raw_predicted_ans, filtered_predicted_ans

