from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import InvalidSessionIdException
from tqdm import tqdm
import threading
import pandas as pd
import uuid
import time
import re
import os
import gc
from Linkedin.Linkedin import Linkedin
from utils.model_pipelines import predict_ans, get_jd_vs_cv_similarity_score
from utils.config import *
from utils.utils import *
logger = logging.getLogger(__name__)
"""
This module contains the `LinkedinJobApply` class, which automates job searching and applying on LinkedIn using Selenium.
"""
"""
TODO:   
    1. Handle text that turn select--> While putting few of the questions like address it turns as select--->_job_form_get_and_insert_qa
"""


class LinkedinJobApply(Linkedin):
    """
    Automates job application processes on LinkedIn, including applying filters, navigating job postings,
    extracting job details, and attempting to submit applications.
    """
    def __init__(self): 
        """
        Initializes the `LinkedinJobApply` class by setting up essential variables like max pages to load,
        max scroll attempts per page, and starting the browser driver.
        """
        super().__init__()
        #create essential variables
        self.max_pages_to_load= LINKEDIN_MAX_PAGES_TO_LOAD_PER_JOB_TITLES
        self.max_scroll_attempts= 10    #per page

    def __del__(self):
        """
        Cleans up resources when the class instance is destroyed.
        """
        super().__del__()


    def run(self):
        """
        Orchestrates the end-to-end job application process on LinkedIn.
        Steps include opening the LinkedIn Jobs page, applying filters, navigating pages, extracting job details,
        and submitting applications.
        """
        logger.info("Starting Job process steps.")
        try:
            # Start heartbeat
            self._is_running = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat)
            self.heartbeat_thread.start()

            #code starts here
            self.browser_driver.get("https://www.linkedin.com/jobs/")
            WebDriverWait(self.browser_driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='Search by title, skill, or company']"))
            )

            for job_title in JOB_TITLES:
                print(f"---------------------------------------Processing jobs for title: {job_title}---------------------------------------")
                logger.debug(f"Processing jobs for title: {job_title}")
                result_number= self._apply_filters(job_title)
                estm_pages= int(result_number/25.5)+1
                print(f"Estimated number of pages: {estm_pages-1}")
                logger.debug(f"Estimated number of pages: {estm_pages-1}")

                if estm_pages > self.max_pages_to_load:
                    estm_pages= self.max_pages_to_load

                logger.debug(f"Searching for jobs in {estm_pages} pages")
                for page_no in tqdm(range(estm_pages), desc="Processing jobs", leave=True, position=0, ncols=80, colour="GREEN"):
                # starts ---->
                    jobs = self._get_jobs_from_current_page()
                    logger.debug(f"Found {len(jobs)} jobs on current page")
                    for j in tqdm(range((len(jobs))), desc=f"Processing jobs on current page: {page_no+1}", leave=False, position=1, dynamic_ncols=True, colour="BLUE"):
                        qa_dataframe= pd.DataFrame()
                        job= jobs[j]
                        job.click()
                        time.sleep(2)
                        reach_daily_easy_apply_limit=  self._easy_apply_limit_reach()

                        logger.debug(f"current job linkedin url: {self.browser_driver.current_url}")
                        is_submited= False
                        job_details= self._get_job_details_to_json(job)

                        #Get JD vs CV score out of 100
                        confidence_score= get_jd_vs_cv_similarity_score(job_details["job_description"])
                        confidence_score= confidence_score if confidence_score else 0
                        job_details["confidence_score"]= confidence_score

                        if job_details["is_easy_apply"]== 'Y' and not reach_daily_easy_apply_limit and confidence_score>= LINKEDIN_JD_VS_CV_THRESHOLD:
                            is_submited, qa_dataframe= self.easy_apply_jobs_apply()
                        
                        job_details["is_submited"]= 'Y' if is_submited else 'N'
                        jd_dataframe= pd.DataFrame([job_details])
                        primary_key= uuid.uuid4().hex
                        jd_dataframe["id"]= primary_key
                        qa_dataframe["job_details_id"]= primary_key if not qa_dataframe.empty else None

                        dataframe_to_sqlite(LINKEDIN_DB_FILE, LINKEDIN_FORM_QA_TABLE, qa_dataframe) if not qa_dataframe.empty else None
                        dataframe_to_sqlite(LINKEDIN_DB_FILE, LINKEDIN_JOB_DETAILS_TABLE, jd_dataframe)
                        del qa_dataframe, jd_dataframe
                        gc.collect()
                # <---- end here
                    if not self._click_next_page_on_job_search():
                        break
                    self.browser_driver.refresh()
                    time.sleep(0.5)
                self.browser_driver.refresh()
                time.sleep(0.5)
                
        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"ime out detected. Might be due to element not found")
        except Exception as e:
            logger.error(f"During processing: {e}")
            print(f"Error during processing: {e}")
            
        finally:
            # Stop heartbeat
            logger.info("End of Job process steps.")
            print("End of Job process steps.")
            self._is_running = False
            if self.heartbeat_thread is not None:
                self.heartbeat_thread.join()

    def _apply_filters(self, job_title):
        """
        Applies search filters on the LinkedIn Jobs page, including job title, location, and Easy Apply filter.
        Args:
            job_title (str): The title of the job to search for.
        Returns:
            int: The total number of jobs found after applying filters.
        """
        result_number= 0
        try:
            if True:    #apply job title and location
                job_title_input = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='Search by title, skill, or company']"))
                )
                job_title_input.clear()
                time.sleep(0.5)
                job_title_input.send_keys(job_title)

                time.sleep(1)

                job_location_input = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='City, state, or zip code']"))
                )
                job_location_input.clear()
                time.sleep(0.5)
                job_location_input.send_keys(JOB_LOCATIONS, Keys.RETURN)

                time.sleep(2)

            if LINKEDIN_APPLY_24_HOURS_FILTER:    #apply application timeline ie 24 hours ago only
                date_posted_filter = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.ID, "searchFilter_timePostedRange"))
                )
                date_posted_filter.click()
                past_24_hours_option = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//span[text()='Past 24 hours']/ancestor::label"))
                )
                past_24_hours_option.click()

                time.sleep(2)
                
                apply_filter_button = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//button[@aria-label[contains(., 'Apply current filter')]]"))
                )
                apply_filter_button.click()

                time.sleep(3)
            
            if LINKEDIN_APPLY_EASY_OPTION:    #  apply easy apply option
                easy_apply_filter = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.ID, "searchFilter_applyWithLinkedin"))
                )
                if easy_apply_filter.get_attribute("aria-checked") == "false":
                    easy_apply_filter.click()
                    time.sleep(2)

            if True:    #Get the number of jobs avilable for the current input
                results_span = WebDriverWait(self.browser_driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, "//div[contains(@class, 'jobs-search-results-list__subtitle')]//span"))
                )
                results_text = results_span.text.strip()
                result_number = ''.join(filter(str.isdigit, results_text))
                print(f"total jobs found {result_number}")

            logger.info("All filter applied. Job found: {result_number}")
        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"ime out detected. Might be due to element not found")
        except Exception as e:
            logger.error(f"Applying filters: {e}")
        finally:
            return int(result_number)

    def _click_next_page_on_job_search(self):
        """
        Attempts to navigate to the next page of job postings. Returns True if successful, False otherwise.
        Returns:
            bool: Whether the next page was successfully loaded.
        """
        try:
            # next_button = WebDriverWait(self.driver, 10).until(
            #     EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='View next page']"))
            # )
            next_button = WebDriverWait(self.browser_driver, 10).until(
                lambda driver: driver.find_element(By.CSS_SELECTOR, "button[aria-label='View next page']")
            )
            if next_button.is_enabled():
                next_button.click()
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-card-container"))
                )
                logger.debug("Clicking next button")
                return True
            logger.info("Next button was disabled, unable to click.")
            return
        
        except NoSuchElementException:
            logger.debug(f"Trying clicking next button, but not found. Might be the last page.")
        except InvalidSessionIdException:
            logger.error(f"Invalid session ID detected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"Time out detected.")
        except Exception as e:
            logger.error(f"Navigating to next page: {e}")

    def _get_jobs_from_current_page(self):
        """
        Retrieves all job cards from the current LinkedIn job search page.
        Returns:
            list: A list of Selenium WebElement objects representing job cards.
        """
        new_jobs = []
        current_total_jobs = 0
        logger.info("Retriving job cards from current page")
        try:
            for i in range(self.max_scroll_attempts):
                new_jobs = self.driver.find_elements(By.CLASS_NAME, "job-card-container")
                if len(new_jobs) > current_total_jobs:
                    current_total_jobs = len(new_jobs)
                    self.browser_driver.execute_script("arguments[0].scrollIntoView(true);", new_jobs[-1])
                    time.sleep(3)
                else:
                    logger.debug("End of scroll")
                    break
            return new_jobs
        except Exception as e:
            logger.error(f"Retriving jobs from current page .{e}")

    def _get_job_details_to_json(self, job):
        '''
        Extracts detailed information about a specific job from its job card.
        Args:
            job (WebElement): The job card element to extract details from.
        return keys:
            'job_title', 'job_location', 'job_posted', 'total_applicant', 'company_name', 'company_link', 
            'company_type', 'company_size', 'company_size_on_linkedin', 'job_description', 
            'job_linkedin_link', 'is_easy_apply', 'emails_found', 'confidence_score'    #, 'applied_Successfully'
        '''
        try:
            job_description_html = self.browser_driver.find_element(By.CSS_SELECTOR, "[class*='jobs-search__job-details--wrapper']").get_attribute('outerHTML')
            temp_file = os.path.join(TEMP_FOLDER, 'job_card.html')
            with open(temp_file, "w") as file:
                file.write(job_description_html)
            job_details = linkedin_extract_job_details(temp_file)


            job_linkedin_link = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")

            job_apply_div = self.browser_driver.find_element(By.CLASS_NAME, "mt4")
            job_apply_content = job_apply_div.get_attribute('innerHTML')

            emails_found= re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', job_details["job_description"].replace("\n", " "))

            job_details["job_linkedin_link"]= job_linkedin_link
            job_details["is_easy_apply"]= linkedin_check_if_easy_apply_avilable(job_apply_content)   #, self.browser_driver)
            job_details["emails_found"]= ', '.join(emails_found) if emails_found else ''

            job_details["confidence_score"]= 'N/A'
            # job_details["applied_Successfully"]= 'N'

            logger.debug("Scraped Job details from current job card")
            return job_details
        except (TimeoutException, InvalidSessionIdException) as e:
            logger.error("Invalid session ID/Time out detected. Restarting the session...")
            super().start_browser_driver_and_login(self.browser_driver.current_url)
        except Exception as e:
            logger.error(f"Processing job: {e}")


    def easy_apply_jobs_apply(self):
        # return None, pd.DataFrame()
        """
        Handles the Easy Apply form submission process for LinkedIn jobs.
        Fills in required fields, clicks through the form, and attempts to submit the application.
        Returns:
            bool: True if the application was successfully submitted, False otherwise.
        """
        '''
        If this function takes large time to submit 
        lets say 5min > terminamte the function and store the
        job application link for a farther verification of what went wrong.
        '''
        logger.info("Trying to access and fill the form.")
        is_submited= False
        df= pd.DataFrame()
        element_type= ''
        
        if self._easy_apply_limit_reach():
            time.sleep(1)
            return
        
        list_of_qa= []
        try: 
            easy_apply_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(@class, 'jobs-apply-button') and contains(@aria-label, 'Easy Apply')]"))
            )
            easy_apply_button.click()
            time.sleep(2)
            
            while True: # while there is a next botton click next:
                # temp_list_of_qa= []
                # Locate the form
                element_type= 'form_elements'
                form_elements = self.browser_driver.find_elements(By.CSS_SELECTOR, 'div[data-test-form-element]')
                for element in form_elements:
                    data= self._job_form_get_and_insert_qa(element)
                    # temp_list_of_qa.append(data) if data else ''
                    list_of_qa.append(data) if data else ''
                
                if self._job_form_check_error_input():
                    logger.warning("Encounter error in the form input")
                    break
                

                # find and click next
                self._job_form_scroll_to_bottom()
                time.sleep(1)
                try:
                    element_type= 'next_button'
                    next_button = self.driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Continue to next step')]")
                    if next_button.is_enabled():
                        next_button.click()
                        time.sleep(2)
                        logger.debug("clicked next option")
                    else:
                        break
                except NoSuchElementException:
                    # if no next--> check for review botton
                    try:
                        element_type= 'review_button'
                        review_button = self.driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Review your application')]")
                        if review_button.is_displayed():
                            review_button.click()
                            logger.debug("clicked review option")
                            break
                    except NoSuchElementException:
                        break
                    except Exception as e:
                        logger.error(f"Unable to click review option: {e}")
                        break
                except Exception as e:
                    logger.error(f"Unable to click next option: {e}")
                    break
                
                
                #only storing when succefull clicked next/review
                # list_of_qa+= temp_list_of_qa

            self._job_form_scroll_to_bottom()
            time.sleep(1)
            # Uncheck the "Follow" checkbox if it is checked by default
            self._job_form_unfollow_comapny()
            time.sleep(1)
            
            if LINKEDIN_SUBMIT_JOB_APPLICATION:
                # After filling in the form, check if the "Submit application" button is available and click it
                is_submited= self._job_form_submit_option()   ################################
                time.sleep(2)
            
            #close the form
            self._job_form_close_div_cross()
            time.sleep(2)
            # click discard if avilabe
            self._job_form_discard_option()
            time.sleep(2)
            
            # print(f"list_of_qa: {list_of_qa}, type= {type(list_of_qa[0])}")
            if list_of_qa:
                df= pd.DataFrame(list_of_qa)
                df["is_submited"]= 'Y' if is_submited else 'N'
            return is_submited, df

        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"Time out detected. Might be due to element not found")
        except NoSuchElementException:
            logger.warning(f"Can't find element type {element_type}")
        except Exception as e:
            logger.error(f"Applying job: {e}")
        finally:
            logger.info(f"Closed job form. submitted- {is_submited}")
    
    def _job_form_get_and_insert_qa(self, element):
        """
        Helper of "easy_apply_jobs_apply"
        Extracts question and answer information from an Easy Apply form element.
        Args:
            element (WebElement): The form element to extract data from.
        Returns:
            dict: A dictionary containing question, input type, pre-filled answers, and available options.
        """
        element_type= ''
        # Initialize variables
        question = ''
        input_type = 'unknown'
        pre_ans = None
        raw_predicted_ans= None
        filtered_predicted_ans = None
        predicted_question_type= None
        available_options = []
        got_model_output= False
        try:
            if True:
                # Extract the question label
                element_type= 'question text'
                question = element.find_element(By.TAG_NAME, 'label').text.strip().split("\n")[0]


            if True:    
                # Check input type
                input_element = None
                element_type= 'answer type'
                if element.find_elements(By.TAG_NAME, 'select'):
                    input_element = element.find_element(By.TAG_NAME, 'select')
                    input_type = 'select'
                elif element.find_elements(By.TAG_NAME, 'input'):
                    input_element = element.find_element(By.TAG_NAME, 'input')
                    input_type = input_element.get_attribute('type')

            if True:
                # Check for pre-selected or pre-filled values and available options
                if input_type == 'select':
                    # Handle dropdown (select element)
                    select = Select(input_element)
                    pre_ans = select.first_selected_option.text.strip()
                    available_options = [option.text.strip() for option in select.options]
                elif input_type == 'radio':
                    try:
                        question = element.find_element(By.XPATH, './/span[@data-test-form-builder-radio-button-form-component__title]').text.split("\n")[0]
                    except Exception as e:
                        question = "Unknown question"
                    # Handle radio buttons
                    element_type= 'radio options'
                    radio_buttons = element.find_elements(By.XPATH, ".//input[@type='radio']")
                    for radio in radio_buttons:
                        # Check if radio is pre-selected
                        if radio.is_selected():
                            pre_ans = radio.get_attribute('value')
                        # Collect all available options
                        available_options.append(radio.get_attribute('value'))
                else:
                    # Handle text-based inputs
                    pre_ans = input_element.get_attribute('value').strip()

            # Build the question dictionary
            question_dict = {
                "question": question,
                "pre_ans": pre_ans if pre_ans else '',
                "raw_predicted_ans": raw_predicted_ans,
                "filtered_predicted_ans": filtered_predicted_ans,
                "input_type": input_type,
                "predicted_question_type": predicted_question_type,
                "available_options": available_options
            }
            #get the valued from the model
            try:
                predicted_question_type, raw_predicted_ans, filtered_predicted_ans = predict_ans(question_dict)
                question_dict["predicted_question_type"]= predicted_question_type
                question_dict["raw_predicted_ans"]= raw_predicted_ans
                question_dict["filtered_predicted_ans"]= filtered_predicted_ans
                got_model_output= True
            except Exception as e:
                logger.error(f"Unable to load model output, error: {e}")
            # print(f"\n\ngot_model_output: {got_model_output}\n\n")
            if got_model_output:
                # putting the ans based on the input type aceepted
                if input_type == 'radio' and available_options:
                    for radio in radio_buttons:
                        if radio.get_attribute('value') == filtered_predicted_ans:
                            element_type= 'radio options'
                            label_for_radio = element.find_element(By.XPATH, f".//label[@for='{radio.get_attribute('id')}']")
                            label_for_radio.click()
                            break
                elif input_type == 'radio' and not available_options:
                    pass
                    # if not pre_ans or pre_ans.lower()== 'select an option':
                elif input_type == 'select' and available_options:
                    try:
                        index_of_first_occurrence= available_options.index(filtered_predicted_ans)
                    except:
                        index_of_first_occurrence= 0
                    select.select_by_index(index_of_first_occurrence)
                elif input_type == 'select' and not available_options:
                    pass
                else:
                    input_element.clear()
                    input_element.send_keys(filtered_predicted_ans)

            logger.debug("Got the QA from the current QA")
            return question_dict
        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"Time out detected.")
        except NoSuchElementException:
            logger.warning(f"Can't find element type {element_type}")
        except Exception as e:
            logger.error(f"Applying job: {e}")

    def _job_form_check_error_input(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        error_message_elements = self.browser_driver.find_elements(By.CLASS_NAME, 'artdeco-inline-feedback__message')
        if error_message_elements:
            for error_element in error_message_elements:
                error_message_text = error_element.text.strip()
                if error_message_text:
                    return True
        return False

    def _job_form_scroll_to_bottom(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            modal_element = self.browser_driver.find_elements(By.XPATH, 
                '//div[@class="artdeco-modal__content jobs-easy-apply-modal__content p0 ember-view"]')
            if modal_element:
                self.browser_driver.execute_script(
                    "arguments[0].style.overflow = 'auto'; arguments[0].scrollTop = arguments[0].scrollHeight;", 
                    modal_element[0]
                )
        except NoSuchElementException:
            logger.debug("Cant find the scroll option for the div")
        except Exception as e:
            logger.error(f"Unable to scroll the div: {e}")

    def _job_form_close_div_cross(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            dismiss_button = self.browser_driver.find_element(By.XPATH, "//button[@aria-label='Dismiss']")
            if dismiss_button:
                ActionChains(self.browser_driver).move_to_element(dismiss_button).click(dismiss_button).perform()
        except NoSuchElementException:
            logger.warning("Unable to click discard button")
        except Exception as e:
            logger.error(f"Cant find the discard button: {e}")
    
    def _job_form_discard_option(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            discard_button = self.browser_driver.find_element(By.XPATH, "//button[span[text()='Discard']]")
            discard_button.click()
        except NoSuchElementException:
            pass
        except Exception as e:
            logger.error(f"Cant find the discard button: {e}")

    def _job_form_unfollow_comapny(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            follow_checkbox = self.driver.find_element(By.XPATH, "//input[@id='follow-company-checkbox']")
            if follow_checkbox.is_selected():
                self.driver.execute_script("arguments[0].click();", follow_checkbox)  # Use JavaScript click
        except NoSuchElementException:
            logger.debug("Unable to click unfollow option")
        except Exception as e:
            logger.error(f"Cant find the unfollow option: {e}")

    def _job_form_submit_option(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            submit_button = self.driver.find_element(By.XPATH, "//button[@aria-label='Submit application']")
            # Check if the button is enabled before clicking
            if submit_button.is_enabled():
                submit_button.click()
                logger.info("Application submitted successfully")
                return True
            else:
                print("Submit button is not enabled.")
        except NoSuchElementException:
            logger.warning("Unable to click submit button")
        except Exception as e:
            logger.error(f"Cant find the submit button: {e}")

    def _easy_apply_limit_reach(self):
        """
        Helper of "easy_apply_jobs_apply" function
        """
        try:
            warning_element = self.browser_driver.find_element(By.CSS_SELECTOR, "div.artdeco-inline-feedback--error[role='alert']")
            warning_element_text= warning_element.text.strip().lower()
            if "limit" in warning_element_text and "today" in warning_element_text:
                logger.info("Hit daily easy apply limit")
                return True
        except NoSuchElementException:
            logger.debug("Cant find daily limit warning")
        except Exception as e:
            logger.error(f"Cant find the submit button: {e}")




if __name__ == "__main__":
    LinkedinJobApply_obj= LinkedinJobApply()
    LinkedinJobApply_obj.run()

