from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from utils.config import logging, q_type_models, qa_models, similarity_check_models, skill_extract_models,\
    JD_SIMILARITY_CHECK_WEIGHTS, CV_DATA, QA_MODEL_MAX_TOKEN_SIZE, JD_VS_CV_SKILLS_SIMILARITY_THRESHOLD
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import gc
import pandas as pd
logger = logging.getLogger(__name__)
"""
get_jd_vs_cv_similarity_score: not processing all input
"""



def extract_skills_from_text(text):
    """Extracts skills from long text using NER with chunking support."""
    from flair.data import Sentence
    from flair.models import SequenceTagger
    model = SequenceTagger.load(skill_extract_models[-1])
    text= re.sub(r"([a-zA-Z]+)", lambda m: m.group(1).capitalize(), text)
    chunk_size=512
    all_entities= []

    def change_device(device):
        model.eval()
        model.to(device)

        tokens = text.split()  # Split text into words
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunks= [" ".join(chunk) for chunk in chunks]

        for chunk in chunks:
            sentence = Sentence(chunk)
            model.predict(sentence)
            all_entities.extend(sentence.get_spans('ner'))
        return list(set([entity.text for entity in all_entities if entity.get_label('ner').value in ("ORG", "MISC")]))

    try:
        torch.cuda.empty_cache()
        gc.collect()
        all_entities= change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            all_entities= change_device("cpu")
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    finally:
        # Cleanup: Release memory after execution
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return all_entities


def get_similar_elements_list(list1, list2):
    if not list1 or not list2:
        return []
    
    model = SentenceTransformer(similarity_check_models[-1])
    if len(list2)> len(list1):
        list1, list2= list2, list1
    similarity_matrix= []
    embeddings1= []
    embeddings2= []

    def change_device(device):    
        model.to(device)
        model.eval
        embeddings1 = model.encode(list1, convert_to_tensor=True)
        embeddings2 = model.encode(list2, convert_to_tensor=True)
        return util.cos_sim(embeddings1, embeddings2)
    try:
        torch.cuda.empty_cache()
        gc.collect()
        similarity_matrix= change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            gc.collect()
            similarity_matrix= change_device("cpu")
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    except Exception as e:
        logger.error(f"Unknown error occured, error: {e}")
    finally:
        # Cleanup: Release memory after execution
        del model, embeddings1, embeddings2
        
    if len(similarity_matrix)<1:
        return []
    
    matches = []
    for i, word in enumerate(list1):
        best_match_idx = similarity_matrix[i].argmax().item()
        best_match = list2[best_match_idx]
        best_score = similarity_matrix[i][best_match_idx].item()  # Convert tensor to float
        matches.append(best_match) if best_score >= JD_VS_CV_SKILLS_SIMILARITY_THRESHOLD else None

    return list(set(matches))


def get_question_type_prediction_batch(text_list, batch_size=64):
    # Load model & tokenizer inside function (so it's released later)
    QUESTION_CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(q_type_models[-1])
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[-1])
    ID2LABEL = QUESTION_CLASSIFIER_MODEL.config.id2label
    results = []

    def change_device(device):
        results = []
        QUESTION_CLASSIFIER_MODEL.to(device)
        QUESTION_CLASSIFIER_MODEL.eval()
        for i in range(0, len(text_list), batch_size):
            batch_text_list = text_list[i:i + batch_size]
            inputs = QUESTION_CLASSIFIER_TOKENIZER(batch_text_list, padding=True, truncation=True, return_tensors="pt").to(device)
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
    df = pd.DataFrame({"text": text_list, "predicted_type": results})
    return df


def get_qa_model_out_raw_batch(text_list, batch_size=4):
    QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_models[-1])
    QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_models[-1], legacy=False)
    inputs = None
    decoded_outputs = []
    df_predicted_types= None

    try:
        df_predicted_types= get_question_type_prediction_batch(text_list)
        df_predicted_types["skill_list"]= df_predicted_types.apply(lambda row: extract_skills_from_text(row["text"]) if row["predicted_type"]== "skills" else [], axis=1)
        df_predicted_types["context"]= df_predicted_types.apply(lambda row:'\n'.join([line for line in CV_DATA["skills"].split("\n") if any(re.search(rf'\b{skill}\b', line, re.IGNORECASE) for skill in row["skill_list"])]) if row["skill_list"] else CV_DATA.get(row["predicted_type"], ""), axis= 1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return pd.DataFrame()
    
    def change_device(device):
        QUESTION_ANSWER_MODEL.to(device)
        QUESTION_ANSWER_MODEL.eval()
        for i in range(0, len(text_list), batch_size):
            df_batch = df_predicted_types[i:i + batch_size]
            batch_text_list= df_batch["text"]
            batch_contexts = df_batch["context"]
            batch_input_texts = [f"You are a candidate filling job application form answer the question based on the given information.\nQuestion: {q}\ncontext: {c}" for q, c in zip(batch_text_list, batch_contexts)]

            inputs = QUESTION_ANSWER_TOKENIZER(
                    batch_input_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=QA_MODEL_MAX_TOKEN_SIZE* batch_size
                ).to(device)
            batch_output_ids = QUESTION_ANSWER_MODEL.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],  # ✅ Ensures padding is handled correctly
                    max_length=512, 
                    num_beams=4,  
                )

            if len(batch_output_ids.shape) == 2:  # Ensuring batch dimension is present
                batch_decoded_outputs = QUESTION_ANSWER_TOKENIZER.batch_decode(batch_output_ids, skip_special_tokens=True)
            else:
                batch_decoded_outputs = [QUESTION_ANSWER_TOKENIZER.decode(batch_output_ids[0], skip_special_tokens=True)]


            decoded_outputs.extend(batch_decoded_outputs)
            del inputs, batch_output_ids, batch_input_texts
    try:
        torch.cuda.empty_cache()
        gc.collect()
        change_device("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA Out of Memory! Switching to CPU.")
        try:
            change_device("cpu")
        except MemoryError:
            logger.critical("CPU MemoryError: System ran out of RAM.")
        except Exception as e:
            logger.critical(f"Unexpected error while using CPU: {e}")
    except Exception as e:
        # logger.exception(f"Error occurred: {e}")
        print(f"Error occurred: {e}")
    finally:
        del QUESTION_ANSWER_MODEL, QUESTION_ANSWER_TOKENIZER
        torch.cuda.empty_cache()
        gc.collect()
        
    df_predicted_types["predcited_ans"]= decoded_outputs
    return df_predicted_types[["text", "predicted_type", "predcited_ans"]]


def get_jd_vs_cv_similarity_score(job_description):
    total_score= 0

    clean_lines = [line for line in [re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' to ', re.sub(r'[^a-zA-Z0-9.,-]+', ' ', s.strip())).strip() for s in re.split(r'\n|\.', job_description) if s.strip()] if line]
    df= get_question_type_prediction_batch(clean_lines)
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

        # inputs = QUESTION_ANSWER_TOKENIZER(input_text, return_tensors="pt").to(device)
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
    text_embedding= []
    options_embeddings= []

    def change_device(device):
        SIMILARITY_CHECK_MODEL.to(device)
        SIMILARITY_CHECK_MODEL.eval()
        # Get predicted option
        text_embedding = SIMILARITY_CHECK_MODEL.encode([text], device= device, show_progress_bar=False)
        options_embeddings = SIMILARITY_CHECK_MODEL.encode(available_options, device= device, show_progress_bar=False)
        similarity_matrix= util.cos_sim(text_embedding, options_embeddings)
        return available_options[similarity_matrix[0].argmax().item()]

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
    except Exception as e:
        logger.error(f"Unknown error occured, error: {e}")
    finally:
        # Cleanup: Release memory after execution
        del SIMILARITY_CHECK_MODEL, text_embedding, options_embeddings, change_device
        torch.cuda.empty_cache()
        gc.collect()
    return most_similar_option


def _old_get_most_similar_option(text, available_options):
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
    except Exception as e:
        logger.error(f"Unknown error occured, error: {e}")
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

