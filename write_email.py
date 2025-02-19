from utils.config import LINKEDIN_JOB_DETAILS_TABLE, LINKEDIN_JD_VS_CV_THRESHOLD, LINKEDIN_DB_FILE, CV_DATA, CV_GOOGLE_DRIVE_KEY
from utils.utils import read_from_sqlite
import pandas as pd
import re
from utils.model_pipelines import extract_skills_from_text, get_similar_elements_list, get_qa_model_out_raw_batch
from datetime import datetime, timedelta




def get_jd_dataframe(db_file= LINKEDIN_DB_FILE, table= LINKEDIN_JOB_DETAILS_TABLE, max_timeframe= 9999):
    df= read_from_sqlite(db_file, table)
    df= df[
        (df["emails_found"] != '')
        & (df["emails_found"].notna())
        & (df["confidence_score"] >= LINKEDIN_JD_VS_CV_THRESHOLD)
    ]
    df= df[["job_title", "emails_found", "confidence_score", "id", "job_description", "id", "inserted_at"]].reset_index(drop= True)
    df["job_description"]= df["job_description"].apply(lambda text: re.sub(r'(?<!\d)\.(?!\d)|[^A-Za-z0-9\s.]', ' ', text.replace("\n", " ")))
    df["inserted_at"] = pd.to_datetime(df["inserted_at"])
    df= df[df["inserted_at"] >= datetime.now() - timedelta(hours=max_timeframe)]
    df= df.reset_index(drop= True)

    df= df.head(5)

    df["jd_skills"]= df["job_description"].apply(lambda text: extract_skills_from_text(text))

    cv_text= re.sub(r'(?<!\d)\.(?!\d)|[^A-Za-z0-9\s.]', ' ', CV_DATA["skills"].replace("\n", " "))
    cv_text= " ".join([i for i in cv_text.split(" ") if  i])
    cv_skills= extract_skills_from_text(cv_text)

    df["jd_cv_common_skills"]= df["jd_skills"].apply(lambda jd_skills: get_similar_elements_list(jd_skills, cv_skills))

    return df


def email_dataframe(row):
    if not row["jd_cv_common_skills"]:
        return pd.DataFrame()
    df= pd.DataFrame({
            "Email": [],
            "Subject": [],
            "Message": [],
            "Attachment": [],
            "Job_description": [],
            "confidence_score": [],
            "job_id": []
        })

    emails_found= row["emails_found"].split(", ")
    match = re.match(r"([A-Za-z0-9.]+(?: [A-Za-z0-9.]+)*)", row["job_title"])
    job_title= match.group(0) if match else ''
    email_subject= f"Application for {job_title}"
    email_body= "Hi,"
    email_body+= f"\n\nI hope this message finds you well."
    email_body+= f"\nI am writing to express my interest in the {job_title} position. With a {email_vars.iloc[10]} and over {email_vars.iloc[11]} years of experience in Python, data engineering, data exploration, and machine learning, I believe I am well-suited for this role."
    email_body+= f"\nMy experience includes developing and optimizing {' ,'.join(row["jd_cv_common_skills"])}. I am eager to bring my expertise to your team."
    email_body+= f"\nThank you for considering my application. I look forward to the opportunity to discuss how I can contribute."
    email_body+= f"\n\nCurrent CTC: {email_vars.iloc[8]}"
    email_body+= f"\nExpected CTC: {email_vars.iloc[9]}"
    email_body+= f"\nOfficial Notice period: {email_vars.iloc[6]} (negotiable)"
    email_body+= f"\nCurrent Location: {email_vars.iloc[4]}"
    email_body+= f"\npreferred location: Remote or Bangalore, Karnataka, India"
    # email_body+= f"\npreferred location: {email_vars.iloc[5]}"
    email_body+= f"\n\nBest regards,"
    email_body+= f"\n{email_vars.iloc[0]}"
    email_body+= f"\n91+ {email_vars.iloc[3]}"
    # email_body+= f"\n{email_vars.iloc[2]} {email_vars.iloc[3]}"
    email_body+= f"\n{email_vars.iloc[1]}"
    
    for email_send_to in emails_found:
        df= pd.concat([df, pd.DataFrame(
                {
                    "Email": [email_send_to],
                    "Subject": [email_subject],
                    "Message": [email_body],
                    "Attachment": [CV_GOOGLE_DRIVE_KEY],
                    "confidence_score": [row["confidence_score"]],
                    "Job_description": [row["job_description"]],
                    "job_id": [row["id"]],
                }
            )]
        )
    return df



    


if __name__ == "__main__":
    questions= [
        "What is your full name?",
        "What is your email id?",
        "What is your phone country code?",
        "What is your phone number?",
        "What is your current location?",
        "What is your preferred location?",
        "What is your notice period in days?",
        "Is your notice period negotiable?",
        "What is your current ctc?",
        "What is your expected ctc?",
        "what is your heighest degree?",
        "what is your total years of experince?",
    ]
    output_file= "emails/test.xlsx"
    df= get_jd_dataframe("DataBase/Linkedin_6_02_2025.db", LINKEDIN_JOB_DETAILS_TABLE)
    email_vars= get_qa_model_out_raw_batch(questions, 16)["predcited_ans"]
    email_df= pd.concat(df.apply(email_dataframe, axis=1).tolist(), ignore_index=True)
    email_df.to_excel(output_file.replace(".xlsx", f"_{datetime.now().strftime('%Y-%m-%d')}.xlsx"), index= False)

