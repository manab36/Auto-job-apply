from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from utils.config import logging, q_type_models, qa_models, similarity_check_models, skill_extract_models, JD_SIMILARITY_CHECK_WEIGHTS, CV_DATA, QA_MODEL_MAX_TOKEN_SIZE
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import json
import gc
import pandas as pd
logger = logging.getLogger(__name__)
"""
get_jd_vs_cv_similarity_score: not processing all input
"""


def __old_extract_skills_from_text(text):
    """Extracts skills from long text using NER with chunking support."""
    text= re.sub(r"([a-zA-Z]+)", lambda m: m.group(1).capitalize(), text)
    from transformers import pipeline, AutoTokenizer, logging as transformers_logging
    transformers_logging.set_verbosity_error()
    skill_extract_model= skill_extract_models[-1]
    tokenizer = AutoTokenizer.from_pretrained(skill_extract_model)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ner_pipeline = pipeline("ner", model=skill_extract_model, aggregation_strategy="max", device=device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        device = "cpu"
        ner_pipeline = pipeline("ner", model=skill_extract_model, aggregation_strategy="max", device=device)

    def chunk_text(text, max_tokens=256, overlap=50):
        """Splits long text into chunks with overlap to prevent loss of context."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk = tokens[i : i + max_tokens]
            chunks.append(tokenizer.decode(chunk))
        return chunks
    # Process text in chunks
    chunks = chunk_text(text)
    merged_entities = set()
    for chunk in chunks:
        ner_results = ner_pipeline(chunk)
        current_word = ""
        for entity in ner_results:
            word = entity["word"]
            if word.startswith("##"):
                current_word += word[2:]
            else:
                if current_word:
                    merged_entities.add(current_word.lower())  # Normalize to lowercase
                current_word = word
        if current_word:
            merged_entities.add(current_word.lower())
    del ner_pipeline, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return list(merged_entities)


def extract_skills_from_text(text):
    """Extracts skills from long text using NER with chunking support."""
    from flair.data import Sentence
    from flair.models import SequenceTagger
    text= re.sub(r"([a-zA-Z]+)", lambda m: m.group(1).capitalize(), text)
    all_entities= []

    def split_text_into_chunks(text, chunk_size=512):
        tokens = text.split()  # Split text into words
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return [" ".join(chunk) for chunk in chunks]
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tagger = SequenceTagger.load(skill_extract_models[-1])
        tagger.eval()
        tagger.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        device = "cpu"
        tagger = SequenceTagger.load(skill_extract_models[-1])
        tagger.eval()
        tagger.to(device)
    except Exception as e:
        logger.exception(f"Error occurred: {e}")

    try:
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            sentence = Sentence(chunk)
            tagger.predict(sentence)
            all_entities.extend(sentence.get_spans('ner'))
        all_entities= list(set([entity.text for entity in all_entities if entity.get_label('ner').value in ("ORG", "MISC")]))
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
    finally:
        del tagger
        torch.cuda.empty_cache()
        gc.collect()
    return all_entities


def get_question_type_predictions(texts, batch_size=64):
    # Load model & tokenizer inside function (so it's released later)
    QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
    ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label
    results = []

    def change_device(device):
        results = []
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
        return results

    try:
        torch.cuda.empty_cache()
        gc.collect()
        results= change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            results= change_device("cpu")
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
    finally:
        del QUESTION_CLASSIFIER_MODEL, QUESTION_CLASSIFIER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
    # Convert results to a DataFrame
    df = pd.DataFrame({"text": texts, "predicted_type": results})
    return df


def __old_get_jd_vs_cv_similarity_score(job_description):
    # split sentace to line for '\n' and '.' --> remove special characters excluding ',' and '.' --> exclude the empty lines from the list
    clean_lines = [line for line in [re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' to ', re.sub(r'[^a-zA-Z0-9.,-]+', ' ', s.strip())).strip() for s in re.split(r'\n|\.', job_description) if s.strip()] if line]
    df= get_question_type_predictions(clean_lines)
    if df.empty:
        logger.error("Unable to fetch categorise description")
        return
    df['content'] = df['predicted_type'].map(CV_DATA)
    SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_models[-1])
    similarity_scores = []
    total_score= 0
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
    except Exception as e:
        logger.exception(f"While trying to fetch similarity score, error: {e}")
    finally:
        # Cleanup: Release memory after execution
        del SIMILARITY_CHECK_MODEL
        torch.cuda.empty_cache()
        gc.collect()

    if len(similarity_scores) != len(df):
        logger.error("Unexpected error while geeting the similarity score")
        return round(total_score, 2)
    
    # Add similarity scores to the DataFrame
    df['similarity_score'] = similarity_scores
    df= df.groupby('predicted_type')['similarity_score'].mean().reset_index().rename(columns={'similarity_score': 'avg_similarity_score'})
    for index, row in df.iterrows():
        category = row['predicted_type']
        similarity_score = row['avg_similarity_score']
        weight = JD_SIMILARITY_CHECK_WEIGHTS.get(category, 0)
        total_score += similarity_score * weight
    del df
    return round(total_score, 2)


def get_jd_vs_cv_similarity_score(job_description):
    total_score= 0

    clean_lines = [line for line in [re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' to ', re.sub(r'[^a-zA-Z0-9.,-]+', ' ', s.strip())).strip() for s in re.split(r'\n|\.', job_description) if s.strip()] if line]
    df= get_question_type_predictions(clean_lines)
    df= df[df["predicted_type"] != "skills"]
    df['content'] = df['predicted_type'].map(CV_DATA)
    df = df.reset_index(drop=True)
    
    jd_text= re.sub(r'(?<!\d)\.(?!\d)|[^A-Za-z0-9\s.]', ' ', job_description.replace("\n", " "))
    jd_text= " ".join([i for i in jd_text.split(" ") if  i])

    cv_text= re.sub(r'(?<!\d)\.(?!\d)|[^A-Za-z0-9\s.]', ' ', CV_DATA["skills"].replace("\n", " "))
    cv_text= " ".join([i for i in cv_text.split(" ") if  i])

    skills_similarity_score= 0
    similarity_scores= []

    SIMILARITY_CHECK_MODEL= SentenceTransformer(similarity_check_models[-1])

    def change_device(device):
        SIMILARITY_CHECK_MODEL.to(device)
        SIMILARITY_CHECK_MODEL.eval()
        # Compute similarity scores
        for idx, row in df.iterrows():
            text = row['text']
            content = row['content']
            # Compute embeddings
            text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device=device, show_progress_bar=False)
            content_embedding = SIMILARITY_CHECK_MODEL.encode([content], device=device, show_progress_bar=False)
            # Compute cosine similarity
            similarity = cosine_similarity(text_embedding, content_embedding)[0][0]
            similarity_scores.append(similarity)
        
        jd_skill_list= extract_skills_from_text(jd_text)
        cv_skill_list= extract_skills_from_text(cv_text)
        
        jd_embeddings = SIMILARITY_CHECK_MODEL.encode(" ".join(jd_skill_list), convert_to_tensor=True)
        cv_embeddings = SIMILARITY_CHECK_MODEL.encode(" ".join(cv_skill_list), convert_to_tensor=True)
        skills_similarity_score = util.pytorch_cos_sim(jd_embeddings, cv_embeddings).item()
        return similarity_scores, skills_similarity_score
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        similarity_scores, skills_similarity_score= change_device(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            similarity_scores, skills_similarity_score= change_device("cpu")
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    except Exception as e:
        logger.exception(f"While trying to fetch similarity score, error: {e}")
    finally:
        del SIMILARITY_CHECK_MODEL

    if len(similarity_scores) != len(df):
        # logger.error("Unexpected error while geeting the similarity score")
        return round(skills_similarity_score, 2)
    
    df['similarity_score'] = similarity_scores
    df= pd.concat([df, pd.DataFrame({
        "text": [""],
        "predicted_type": ["skills"],
        "content" : [""],
        "similarity_score": [skills_similarity_score]
    })], ignore_index=True)
    df= df.groupby('predicted_type')['similarity_score'].mean().reset_index().rename(columns={'similarity_score': 'avg_similarity_score'})
    for index, row in df.iterrows():
        category = row['predicted_type']
        similarity_score = row['avg_similarity_score']
        weight = JD_SIMILARITY_CHECK_WEIGHTS.get(category, 0)
        total_score += similarity_score * weight
    del df
    return round(skills_similarity_score, 2)


def get_question_type_prediction(text):
    # Load model & tokenizer inside function (so it's released later)
    result= None
    QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
    ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label

    def change_device(device):
        QUESTION_CLASSIFIER_MODEL.to(device)
        QUESTION_CLASSIFIER_MODEL.eval()
        inputs = QUESTION_CLASSIFIER_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = QUESTION_CLASSIFIER_MODEL(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1)
        return ID2LABEL[predicted_classes.item()]

    try:
        torch.cuda.empty_cache()
        gc.collect()
        result= change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            result= change_device("cpu")
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
    QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_models[-1])
    QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_models[-1], legacy=False)
    predicted_question_type= None
    result= None

    def change_device(device):
        QUESTION_ANSWER_MODEL.to(device)
        QUESTION_ANSWER_MODEL.eval()
        # Get predicted question type
        predicted_question_type = get_question_type_prediction(question)

        skill_list= extract_skills_from_text(question) if  predicted_question_type== "skills" else []
        context= '\n'.join([line for line in CV_DATA["skills"].split("\n") if any(re.search(rf'\b{skill}\b', line, re.IGNORECASE) for skill in skill_list)]) if skill_list else CV_DATA.get(predicted_question_type, "")
        input_text = f"You are a candidate filling job application form answer the question based on the given information.\nQuestion: {question}\ncontext: {context}"

        inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt").to(device)
        inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt", padding=True, truncation=True, max_length= QA_MODEL_MAX_TOKEN_SIZE).to(device)

        output_ids = QUESTION_ANSWER_MODEL.generate(inputs["input_ids"], max_length= 512)
        return predicted_question_type, QUESTION_ANSWER_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)

    try:
        torch.cuda.empty_cache()
        gc.collect()
        predicted_question_type, result= change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            predicted_question_type, result= change_device("cpu")
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
    SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_models[-1])
    most_similar_option= None

    def change_device(device):
        SIMILARITY_CHECK_MODEL.to(device)
        SIMILARITY_CHECK_MODEL.eval()
        # Get predicted option
        text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
        options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
        cosine_similarities = cosine_similarity(text_embedding, options_embeddings)
        most_similar_index = cosine_similarities.argmax()
        return available_options[most_similar_index]

    try:
        torch.cuda.empty_cache()
        gc.collect()
        most_similar_option = change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            most_similar_option = change_device("cpu")
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
    predicted_question_type, raw_predicted_ans, filtered_predicted_ans= None, None,None
    try:
        question= question_dict["question"]
        # pre_ans= question_dict["pre_ans"]
        input_type= question_dict["input_type"]
        available_options= question_dict["available_options"]
        predicted_question_type, raw_predicted_ans= get_qa_model_out_raw(question)
        filtered_predicted_ans= raw_predicted_ans

        if 'year' in filtered_predicted_ans or 'day' in filtered_predicted_ans:
            match = re.search(r'(\d+(\.\d+)?)\s*(year?|day?)', filtered_predicted_ans.lower())
            if match:
                number = match.group(1)
                # Remove leading zero if there's no decimal point
                if '.' not in number:
                    number = str(int(float(number)))  # Converts the number to an integer and removes leading zero
                filtered_predicted_ans= number

        available_options= [item for item in available_options if re.sub(r'[^a-zA-Z]', '', item.lower()) != "selectanoption"]
        if input_type in ("select", "radio") and not raw_predicted_ans in available_options:
            filtered_predicted_ans= get_most_similar_option(raw_predicted_ans, available_options)
        return predicted_question_type, raw_predicted_ans, filtered_predicted_ans
    except Exception as e:
            logger.exception(f"While trying to predict, error: {e}")
    finally:
        return predicted_question_type, raw_predicted_ans, filtered_predicted_ans

