from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.remote_connection import LOGGER as selenium_logger
from selenium.common.exceptions import NoSuchElementException
import time
import os
import pickle
from utils.config import logging, BROWSER_CAHCHE_FOLDER
from utils.utils import set_chrome_settings
logger = logging.getLogger(__name__)
'''
So that the it encounter no issue run just one tab/window at a time for a single instance of this class !!!
ie dont open another link in another tab/window using the same driver
'''
"""
TODO
    2. change need to be done in '_set_chrome_settings()' function
        if headless_browser:
            chrome_options.add_argument("--headless")  # Enable headless mode
            chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration (optional)
            chrome_options.add_argument("--no-sandbox")  # Bypass OS security model (optional)
        chrome_service = Service(ChromeDriverManager().install())
"""



class Linkedin:
    '''
    Automates LinkedIn login and session management using Selenium, 
    including support for cookies, local storage, and manual login.
    '''
    def __init__(self):
        '''
        Initializes the Linkedin class by setting up essential folders, variables, 
        and starting the browser driver.
        '''
        #create essential folders
        selenium_logger.setLevel(logging.WARNING)
        log_in_cahche_folder= os.path.join(BROWSER_CAHCHE_FOLDER, "Linkedin")
        os.makedirs(log_in_cahche_folder, exist_ok=True)
        
        #create essential variables
        self.cookie_file= os.path.join(log_in_cahche_folder, "cookies.pkl")
        self.local_storage_file= os.path.join(log_in_cahche_folder, "local_storage.pkl")

        #create chrome driver
        self.start_browser_driver_and_login()
        self._is_running= False

    @property
    def browser_driver(self):
        '''
        Returns the active Selenium WebDriver instance.
        '''
        return self.driver
        
    def __del__(self):
        '''
        Cleans up and quits the browser driver on object deletion.
        '''
        if hasattr(self, 'driver') and self.driver:
            logger.info("Quiting browser instances, including all tabs/windos")
            self.driver.quit()  #remove all the tabs

    def start_browser_driver_and_login(self, current_url= "about:blank"):
        '''
        Initializes and starts the browser driver, logs into LinkedIn, 
        and navigates to the specified URL.
        '''
        logger.debug("Starting Browser instance")
        if hasattr(self, 'driver') and self.driver:
            logger.info("Restarting borwser")
            self.driver.quit()
        try:
            self.driver = set_chrome_settings()
        except Exception as e:
            logger.error(f"Error initializing the browser driver: {e}")
            raise RuntimeError(f"Error initializing the browser driver: {e}")
        self._login()
        self.driver.get(current_url) 
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

    def _login(self):
        '''
        Attempts to log into LinkedIn using cookies and local storage if available, 
        or falls back to manual login if needed.
        '''
        try:
            if os.path.exists(self.cookie_file) and os.path.exists(self.local_storage_file):
                logger.info("Cookies found. Attempting to log in using cookies.")
                try:
                    self._login_via_cookies()
                except Exception as e:
                    logger.error(f"Error during login via cookies: {e}. Attempting manual login.")
                    print(f"Error during login via cookies: {e}. Attempting manual login.")
                    self._login_first_time()
            else:
                logger.info("No cookies found. Attempting to log manually.")
                self._login_first_time()
        except Exception as e:
            logger.error(f"Login process failed: {e}.")
            print(f"Login process failed: {e}.")
            raise e  # Re-raise the error to halt further execution
        
    def _login_first_time(self):
        '''
        Performs a manual login to LinkedIn and saves the session data 
        (cookies and local storage) for future use.
        '''
        try:
            self.driver= set_chrome_settings(False)
            self.driver.get("https://www.linkedin.com/login")
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            input("Log in manually and press Enter when done...")
            self._save_session()
            self.driver.close()
            self.driver= set_chrome_settings()
            self._login()       ############## is it redundent ??????

        except Exception as e:
            logger.error(f"Error during manual login: {e}")
            raise RuntimeError(f"Error during manual login: {e}")

    def _save_session(self):
        '''
        Saves cookies and local storage to files for future login sessions.
        '''
        try:
            with open(self.cookie_file, "wb") as cookie_file:
                pickle.dump(self.driver.get_cookies(), cookie_file)
            local_storage = self.driver.execute_script(
                "return Object.entries(window.localStorage).reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});"
            )
            with open(self.local_storage_file, "wb") as storage_file:
                pickle.dump(local_storage, storage_file)
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise RuntimeError(f"Error saving session: {e}")

    def _login_via_cookies(self):
        '''
        Logs into LinkedIn using saved cookies and local storage. 
        Falls back to manual login if unsuccessful.
        '''
        try:
            self.driver.get("https://www.linkedin.com/")
            with open(self.cookie_file, "rb") as cookie_file:
                cookies = pickle.load(cookie_file)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)
            with open(self.local_storage_file, "rb") as storage_file:
                local_storage = pickle.load(storage_file)
                for key, value in local_storage.items():
                    self.driver.execute_script(f"window.localStorage.setItem('{key}', '{value}');")
            self.driver.refresh()
            
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

            if not self._is_logged_in():
                logger.info("Cookie-based login failed. Falling back to manual login.")
                print("Cookie-based login failed. Falling back to manual login.")
                self._login_first_time()
        except Exception as e:
            logger.error(f"Error during cookie-based login: {e}")
            print(f"Error during cookie-based login: {e}")
            self._login_first_time()

    def _is_logged_in(self):
        '''
        Verifies if the user is logged into LinkedIn by checking for specific elements on the page.
        '''
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".feed-identity-module__actor-meta"))
            )
            logger.info("Login successful.")
            return True
        except NoSuchElementException:
            logger.error("Unable to find to element to check is logged in")
        except Exception as e:
            logger.error(f"Login verification failed: {e}")
            print(f"Login verification failed: {e}")
            return False

    def _heartbeat(self):
        '''
        Monitors and prints the current URL every 3 seconds while the browser is running.
        '''
        while self._is_running:
            self.current_url= self.driver.current_url
            # logger.debug(f"Current URL: {self.current_url}")
            time.sleep(3)



if __name__ == "__main__": 
    linkedin_obj= Linkedin()
    print("Browser driver:", linkedin_obj.browser_driver)
