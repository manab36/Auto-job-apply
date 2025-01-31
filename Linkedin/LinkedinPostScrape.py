from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import InvalidSessionIdException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from Linkedin.Linkedin import Linkedin
from tqdm import tqdm
import pandas as pd
import pyperclip
import time
import re
from utils.config import *
from utils.utils import *
logger = logging.getLogger(__name__)




class LinkedinPostScrape(Linkedin):
    def __init__(self): 
        super().__init__()
        self.post_to_process= LINKEDIN_POST_TO_PROCESS
        self.action_chain= ActionChains(self.browser_driver)

    def __del__(self):
        super().__del__()

    def run(self):
        self.browser_driver.get("https://www.linkedin.com/feed/")
        if self._apply_sort_by_recent():
            print("Applied sort by recent filter")
            logger.debug("Applied sort by recent filter")
            time.sleep(3)

        # for i in range(self.post_to_process):
        for post_no in tqdm(range(self.post_to_process), desc="Processing posts", leave= True):
            try:
            # logic goes here --->
                # Load the current post competely
                xpath = f'//div[contains(@data-id, "urn:li:activity") and contains(@class, "relative") and @data-finite-scroll-hotkey-item="{post_no}"]'
                specific_div = WebDriverWait(self.browser_driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                self.action_chain.move_to_element(specific_div).perform()
                WebDriverWait(self.browser_driver, 10).until(EC.visibility_of(specific_div))
                
                # Apply load more option if avilable
                self._apply_more_option(specific_div)
                
                # Get the link of the post
                post_link= self._get_post_link(specific_div) if LINKEDIN_GET_POST_LINK else ""
                
                post_div_card = specific_div.get_attribute('outerHTML')
                post_details_json= self._get_post_details_json(post_div_card)
                if not post_details_json:
                    print(f"link to the post: {post_link}")
                    logger.debug(f"link to the post: {post_link}")
                    continue
                
                post_details_json["post_link"]= post_link
                dataframe_to_sqlite(LINKEDIN_DB_FILE, LINKEDIN_POSTS_DETAILS_TABLE, pd.DataFrame([post_details_json]))
                time.sleep(0.2)
            # <--- logic ends here
            except InvalidSessionIdException as e:
                logger.error(f"Invalid session IDdetected. Restarting the session....")
                #super().start_browser_driver_and_login()
            except TimeoutException:
                logger.error(f"Time out detected. Might be due to element not found")
            except Exception as e:
                logger.error(f"During processing posts: {e}")
                print(f"During processing posts: {e}")

    def _apply_sort_by_recent(self):
        try:
            dropdown_buttons = WebDriverWait(self.browser_driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "button.artdeco-dropdown__trigger"))
            )
            if len(dropdown_buttons) >= 3:
                dropdown_buttons[2].click()  # Click the third button
                WebDriverWait(self.browser_driver, 10).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "div.artdeco-dropdown__content-inner"))
                )
                recent_option = WebDriverWait(self.browser_driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'artdeco-dropdown__item') and text()='Recent']"))
                )
                recent_option.click()
                logger.debug("Applied sort by recent filter")
                return True
        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"Time out detected. Might be due to element not found")
        except Exception as e:
            logger.error(f"During applying sort filter: {e}")
            print(f"Error during applying sort filter: {e}")

    def _apply_more_option(self, specific_div):
        try:
            more_button = specific_div.find_element(By.XPATH, './/button[contains(@class, "feed-shared-inline-show-more-text__see-more-less-toggle")]')
            if more_button.is_displayed():
                more_button.click()
                logger.debug("Clicked 'more..' option")
        except (InvalidSessionIdException, TimeoutException) as e:
            logger.error(f"Invalid session ID/Time out detected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except NoSuchElementException:
            logger.debug("Cant find 'load more..' option")
        except Exception as e:
            logger.error(f"During apply 'load more..' option: {e}")

    def _get_post_link(self, specific_div):
        # logger.warning("While running his function will interupt your copy paste machanism")
        """
        While running his function will interupt your copy paste machanism because of pyperclip
        """
        try:
            more_options_button = specific_div.find_element(
                By.XPATH, './/button[contains(@class, "feed-shared-control-menu__trigger")]'
            )
            more_options_button.click()

            copy_link_element = WebDriverWait(self.browser_driver, 20).until(
                EC.presence_of_element_located((By.XPATH, './/div[contains(@class, "artdeco-dropdown__content-inner")]'))
            ).find_element(
                By.XPATH, ".//li[contains(., 'Copy link to post')]//div[contains(@class, 'feed-shared-control-menu__dropdown-item')]"
            )
            self.action_chain.move_to_element(copy_link_element).click().perform()
            copied_link = pyperclip.paste()
       
            logger.debug("Got the post link")
            return copied_link 
        except InvalidSessionIdException as e:
            logger.error(f"Invalid session IDdetected. Restarting the session....")
            #super().start_browser_driver_and_login()
        except TimeoutException:
            logger.error(f"Time out detected")
        except NoSuchElementException:
            logger.warning("Cant find post additional option")
        except Exception as e:
            """
            If error is causing by
            Pyperclip could not find a copy/paste mechanism for your system
            Try installing xclip:
                sudo apt-get install xclip  # Or xsel
            else avoid using this method
            """
            logger.error(f"During processing: {e}")
            # print(f"Error during processing: {e}")

    def _get_post_details_json(self, html_content):
        logger.debug("Trying to scrape job post details")
        try:
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            url_pattern = r'https?://(?:www\.)?[a-zA-Z0-9./?=&%_-]+'

            poster_name= ''
            post_link= ''
            time_posted= ''
            likes= ''
            text_content= ''
            emails= []
            urls= []

            soup = BeautifulSoup(html_content, 'html.parser')
            poster_name = soup.find('span', class_='zemGKKTqRXFwKgAzmbyOlHWtsjOyCSDjVqCmo')
            if poster_name:
                poster_name= poster_name.get_text(strip=True)

            post_link = soup.find('a', class_='gNDVSUlysFTeyzqdLkmGtXiwIkTNLonGmA update-components-actor__meta-link')
            if post_link:
                post_link = post_link['href']

            time_text = soup.find('span', class_='update-components-actor__sub-description').get_text(strip=True)
            if time_text:
                time_posted = time_text.split('•')[0].strip()
            
            likes = soup.find('span', class_='social-details-social-counts__reactions-count')
            if likes:
                likes = likes.get_text(strip=True)

            text_content = soup.find('div', class_='update-components-text')
            if text_content: 
                text_content= text_content.get_text(separator=' ', strip=True)    #.get_text(strip=True)#
                emails = re.findall(email_pattern, text_content)
                urls = re.findall(url_pattern, text_content)
                
            return {
                "poster_name": poster_name,
                "post_link": post_link,
                "time_posted": time_posted,
                "likes": likes,
                "text_content": text_content,
                "emails_found": emails,
                "links_found": urls
            }
        except Exception as e:
            logger.error(f"While trying to scrape post: {e}")
            print(f"While trying to scrape post: {e}")



if __name__ == "__main__": 
    LinkedinPostScrape_obj= LinkedinPostScrape()
    LinkedinPostScrape.run()
