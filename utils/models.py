from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
# import torch
import json
from utils.config import logging
logger = logging.getLogger(__name__)


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


#Question Classifier
try: 
    q_type_models= [
        'model/fine_tuned_question_classifier_model_lite-default',
        'model/fine_tuned_question_classifier_model_lite-Adam',
        'model/fine_tuned_question_classifier_model_lite-AdamW',
        'model/fine_tuned_question_classifier_model_lite-SGD']
    # MOCEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    QUESTION_CLASSIFIER = DistilBertForSequenceClassification.from_pretrained(q_type_models[3])
    QUESTION_CLASSIFIER.eval()
    # QUESTION_CLASSIFER_MODEL.to(MOCEL_DEVICE)  # Move model to GPU if available
    QUESTION_CLASSIFIER_TOKENIZER = DistilBertTokenizerFast.from_pretrained(q_type_models[3])
    ID2LABEL = QUESTION_CLASSIFIER.config.id2label

    if not all(item in ID2LABEL.values() for item in QUESTION_TYPES) and all(ID2LABEL[key] in QUESTION_TYPES for key in ID2LABEL):
        logger.critical("Question classifier model doesn't match the class provided")
except FileNotFoundError:
    logger.error("Question classifier model folder not found.")
except Exception as e:
    logger.error("Unable to load the Question classifier model")


# Load CV data
try: 
    with open("Model_dataset/cv.json", "r") as file:
        CV_DATA= json.load(file)
except FileNotFoundError:
    logger.error("CV file not found.")
except Exception as e:
    logger.error("Unable to load the CV file ")



#Question Answer
try: 
    qa_type_model= "t5-large"
    QUESTION_ANSWER_MODEL = T5ForConditionalGeneration.from_pretrained(qa_type_model)
    QUESTION_ANSWER_MODEL.eval()
    QUESTION_ANSWER_TOKENIZER = T5Tokenizer.from_pretrained(qa_type_model, legacy= False)

except FileNotFoundError:
    logger.error("Question classifier model folder not found.")
except Exception as e:
    logger.error("Unable to load the Question classifier model")

# Similarity check model
try: 
    similarity_check_model= "all-MiniLM-L6-v2"
    SIMILARITY_CHECK_MODEL = SentenceTransformer(similarity_check_model)
except FileNotFoundError:
    logger.error("Question classifier model folder not found.")
except Exception as e:
    logger.error("Unable to load the Question classifier model")
