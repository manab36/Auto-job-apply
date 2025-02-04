from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from utils.config import logging, q_type_models, qa_type_models, similarity_check_model, JD_SIMILARITY_CHECK_WEIGHTS
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import json
import gc
import pandas as pd
logger = logging.getLogger(__name__)
# Load CV data
try: 
    with open("Model_dataset/cv.json", "r") as file:
        CV_DATA= json.load(file)
except FileNotFoundError:
    logger.error("CV file not found.")
except Exception as e:
    logger.error("Unable to load the CV file ")




def get_question_type_predictions(texts, batch_size=64):
    # Load model & tokenizer inside function (so it's released later)
    QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
    ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label
    results = []
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        QUESTION_CLASSIFIER_MODEL.to(device)
        QUESTION_CLASSIFIER_MODEL.eval()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = QUESTION_CLASSIFIER_TOKENIZER(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim=1)
            batch_results = [ID2LABEL[idx.item()] for idx in predicted_classes]
            results.extend(batch_results)
            # Free memory
            del inputs, outputs, logits
            torch.cuda.empty_cache()
            gc.collect()
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            results= []
            device = "cpu"
            QUESTION_CLASSIFIER_MODEL.to(device)
            QUESTION_CLASSIFIER_MODEL.eval()
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = QUESTION_CLASSIFIER_TOKENIZER(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=1)
                batch_results = [ID2LABEL[idx.item()] for idx in predicted_classes]
                results.extend(batch_results)
                # Free memory
                del inputs, outputs, logits
                torch.cuda.empty_cache()
                gc.collect()
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        del QUESTION_CLASSIFIER_MODEL, QUESTION_CLASSIFIER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
    # Convert results to a DataFrame
    df = pd.DataFrame({"text": texts, "predicted_type": results})
    return df


def get_jd_vs_cv_similarity_score(job_description):
    # split sentace to line for '\n' and '.' --> remove special characters excluding ',' and '.' --> exclude the empty lines from the list
    clean_lines = [line for line in [re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' to ', re.sub(r'[^a-zA-Z0-9.,-]+', ' ', s.strip())).strip() for s in re.split(r'\n|\.', job_description) if s.strip()] if line]
    df= get_question_type_predictions(clean_lines)
    if df.empty:
        logger.error("Unable to fetch categorise description")
        return
    df['content'] = df['predicted_type'].map(CV_DATA)
    SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_model)
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        SIMILARITY_CHECK_MODEL.to(device)
        SIMILARITY_CHECK_MODEL.eval()
        # Compute similarity scores
        similarity_scores = []
        for idx, row in df.iterrows():
            text = row['text']
            content = row['content']
            # Compute embeddings
            text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device=device, show_progress_bar=False)
            content_embedding = SIMILARITY_CHECK_MODEL.encode([content], device=device, show_progress_bar=False)
            # Compute cosine similarity
            similarity = cosine_similarity(text_embedding, content_embedding)[0][0]
            similarity_scores.append(similarity)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            device = "cpu"
            SIMILARITY_CHECK_MODEL.to(device)
            SIMILARITY_CHECK_MODEL.eval()
            # Compute similarity scores
            similarity_scores = []
            for idx, row in df.iterrows():
                text = row['text']
                content = row['content']
                # Compute embeddings
                text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device=device, show_progress_bar=False)
                content_embedding = SIMILARITY_CHECK_MODEL.encode([content], device=device, show_progress_bar=False)
                # Compute cosine similarity
                similarity = cosine_similarity(text_embedding, content_embedding)[0][0]
                similarity_scores.append(similarity)
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    finally:
        # Cleanup: Release memory after execution
        del SIMILARITY_CHECK_MODEL
        torch.cuda.empty_cache()
        gc.collect()

    # Add similarity scores to the DataFrame
    df['similarity_score'] = similarity_scores
    df= df.groupby('predicted_type')['similarity_score'].mean().reset_index().rename(columns={'similarity_score': 'avg_similarity_score'})

    total_score= 0
    for index, row in df.iterrows():
        category = row['predicted_type']
        similarity_score = row['avg_similarity_score']
        weight = JD_SIMILARITY_CHECK_WEIGHTS.get(category, 0)
        total_score += similarity_score * weight
    del df
    return round(total_score, 2)


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
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            torch.cuda.empty_cache()
            gc.collect()
            device = "cpu"
            QUESTION_CLASSIFIER_MODEL.to(device)
            QUESTION_CLASSIFIER_MODEL.eval()
            inputs = QUESTION_CLASSIFIER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim=1)
            result = ID2LABEL[predicted_classes.item()]
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
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
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            device = "cpu"
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
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    finally:
        # Cleanup: Release memory after execution
        del QUESTION_ANSWER_MODEL, QUESTION_ANSWER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
    return predicted_question_type, result


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
        text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
        options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
        cosine_similarities = cosine_similarity(text_embedding, options_embeddings)
        most_similar_index = cosine_similarities.argmax()
        most_similar_option = available_options[most_similar_index]
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            device = "cpu"
            SIMILARITY_CHECK_MODEL.to(device)
            SIMILARITY_CHECK_MODEL.eval()
            # Get predicted option
            text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
            options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
            cosine_similarities = cosine_similarity(text_embedding, options_embeddings)
            most_similar_index = cosine_similarities.argmax()
            most_similar_option = available_options[most_similar_index]
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
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
    if input_type in ("select", "radio") and not raw_predicted_ans in available_options:
        filtered_predicted_ans= get_most_similar_option(raw_predicted_ans, available_options)
    return predicted_question_type, raw_predicted_ans, filtered_predicted_ans


