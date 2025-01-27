import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
import re
from datetime import datetime
from selenium.webdriver.common.by import By
import json
from utils.config import DB_FILE_DATETIME_FORMAT
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
logger = logging.getLogger(__name__)





def set_chrome_settings(headless_browser= True):
    '''
    Configures and returns a Chrome WebDriver instance with predefined settings.
    '''
    logger.debug("nitializing chrome driver")
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--start-maximized")  # Optional: start Chrome maximized

    # chrome_options.add_argument("--log-level=3")  # Suppress ChromeDriver logs
    # chrome_options.add_argument("--silent")       # Silent mode for ChromeDriver
    if headless_browser:
        chrome_options.add_argument("--headless")  # Enable headless mode
        chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration (optional)
        chrome_options.add_argument("--no-sandbox")  # Bypass OS security model (optional)
    chrome_service = Service(
        ChromeDriverManager().install()
        # ,log_output=os.devnull  # Suppress logs from WebDriverManager
        )
    return webdriver.Chrome(service= chrome_service, options= chrome_options)


def dataframe_to_sqlite(db_file, table_name, df):
    """
    Inserts a Pandas DataFrame into an SQLite database table.
    If the table exists but columns do not match,
    a new table with the suffix '_new' is created.

    Args:
        db_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to insert data into.
        df (pd.DataFrame): DataFrame containing data to be inserted.
    
    Returns:
        bool: True if successful, raises an exception otherwise.
    """
    try:
        # Convert any lists in the dataframe columns to JSON strings
        for column in df.columns:
            if df[column].apply(lambda x: isinstance(x, list)).any():
                df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

        # Add a new column for current date and time
        df['inserted_at'] = datetime.now().strftime(DB_FILE_DATETIME_FORMAT)

        # Connect to the database
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            table_exists = cursor.fetchone()
            
            if table_exists:
                # Fetch existing columns
                cursor.execute(f"PRAGMA table_info({table_name});")
                existing_columns = [column[1] for column in cursor.fetchall()]
                
                # Compare columns
                missing_columns = set(df.columns) - set(existing_columns)
                extra_columns = set(existing_columns) - set(df.columns)
                
                if missing_columns or extra_columns:
                    new_table_name = f"{table_name}_new"
                    logger.info(f"Columns mismatch detected. Creating new table '{new_table_name}'.")
                    df.to_sql(new_table_name, conn, if_exists='replace', index=False)
                    logger.info(f"New table '{new_table_name}' created, and data inserted.")
                    return True
                
                # Insert data if columns match
                df.to_sql(table_name, conn, if_exists='append', index=False)
                logger.info(f"Data inserted into existing table '{table_name}'.")
            else:
                # Create table and insert data
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Data inserted into existing table '{table_name}'.")
        
    except Exception as e:
        logger.error(f"Error inserting data into SQLite: {e}")


def read_from_sqlite(db_file, table_name):
    """
    Reads data from an SQLite database table into a Pandas DataFrame.

    Args:
        db_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to read data from.

    Returns:
        pd.DataFrame: DataFrame containing the data from the table.
    """
    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_file) as conn:
            # Read data from the table into a DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        return df
    
    except Exception as e:
        print(f"Error reading data from SQLite: {e}")
        logger.error(f"Error reading data from SQLite: {e}")
        return None


def linkedin_extract_job_details(html_file_path):
    """
    Extracts job details from a LinkedIn job posting HTML file.
    
    Args:
        html_file_path (str): Path to the HTML file.
    
    Returns:
        dict: A dictionary containing job details.
        return keys:
            'job_title', 'job_location', 'job_posted', 'total_applicant', 
            'company_name', 'company_link', 'company_type', 'company_size', 
            'company_size_on_linkedin', 'job_description',
        
    """

    job_location= ""
    job_posted= ""
    total_applicant= ""
    company_name= ""
    company_link= ""
    company_type= ""
    company_size= ""
    company_size_on_linkedin= ""
    job_description= ""


    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    def safe_get_text(tag):
        """Safely extract text from a BeautifulSoup tag."""
        return tag.get_text(strip=True) if tag else ""

    def safe_get_attr(tag, attr):
        """Safely extract an attribute from a BeautifulSoup tag."""
        return tag[attr] if tag and tag.has_attr(attr) else "N/A"

    # Extract job title
    job_title = safe_get_text(soup.find("h1", class_="t-24 t-bold inline"))

    # Extract job other details
    job_details_div = soup.find("div", class_="t-black--light mt2")
    job_details = safe_get_text(job_details_div)
    if job_details:
        details_parts = job_details.split("·")
        job_location = details_parts[0].strip() if len(details_parts) > 0 else ""
        job_posted = details_parts[1].strip() if len(details_parts) > 1 else ""
        total_applicant = int(re.search(r'\d+', details_parts[2].strip()).group()) if len(details_parts) > 2 else ""
    
    # Extract company name and link
    company_div = soup.find('div', class_='job-details-jobs-unified-top-card__company-name')
    if company_div:
        company_link_tag = company_div.find('a')
        company_name = safe_get_text(company_link_tag)
        company_link = safe_get_attr(company_link_tag, 'href')
    
    # Extract company other details
    company_details_div = soup.find("div", class_="t-14 mt5")
    if company_details_div:
        company_details = company_details_div.get_text("\n", strip=True).split("\n")
        company_type = company_details[0].strip() if len(company_details) > 0 else ""
        company_size = company_details[1].strip() if len(company_details) > 1 else ""
        company_size_on_linkedin = company_details[-1].strip() if len(company_details) > 2 else ""

    # Extract hiring team name and link
    hiring_team_div = soup.find('div', class_='display-flex align-items-center mt4')
    if hiring_team_div:
        hiring_team = hiring_team_div.find("a")
        requiter_name = safe_get_text(hiring_team)
        requiter_profile_link = safe_get_attr(hiring_team, 'href')

    # Extract about job
    job_description = '\n'.join(line for line in soup.find("div", class_="jobs-description__content").get_text(strip= False).strip().splitlines() if line.strip())

    job_data = {
        'job_title': job_title,
        'job_location': job_location,
        'job_posted': job_posted,
        'total_applicant': total_applicant,
        'company_name': company_name,
        'company_link': company_link,
        'company_type': company_type,
        'company_size': company_size,
        'company_size_on_linkedin': company_size_on_linkedin,
        # 'requiter_name': requiter_name,
        # 'requiter_profile_link': requiter_profile_link,
        'job_description': job_description,
    }

    return job_data


def linkedin_check_if_easy_apply_avilable(job_page_html_content, driver= False):
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(job_page_html_content, 'html.parser')
    
    # Find the button with the jobs-apply-button class
    apply_button = soup.find('button', class_='jobs-apply-button')

    # Find already applied 
    applied_feedback = soup.find('div', class_='artdeco-inline-feedback artdeco-inline-feedback--success ember-view')
 
    if apply_button:
        aria_label = apply_button.get('aria-label', '')
        if "Easy Apply" in aria_label:
            return 'Y'
        elif "Apply to" in aria_label:
            external_url= 'N'
            if driver:    # access external apply link
                apply_button = driver.find_element(By.CLASS_NAME, "jobs-apply-button")
                apply_button.click()
                driver.switch_to.window(driver.window_handles[-1])
                external_url = driver.current_url
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            return external_url
    elif applied_feedback:
        applied_message = applied_feedback.find('span', class_='artdeco-inline-feedback__message')
        if applied_message:
            return applied_message.get_text(strip=True)
    return ''


