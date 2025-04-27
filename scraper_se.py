from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
import requests

# Setup Chrome options
chrome_options = Options()
#chrome_options.add_argument("--headless")  # Run headless
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("user-agent=Mozilla/5.0")

# Setup webdriver (Make sure chromedriver is installed)
driver = webdriver.Chrome(options=chrome_options)

# Base URL
base_url = "https://www.aastocks.com/tc/stocks/news/aafn/latest-news"

# Create output folder
os.makedirs("news_articles", exist_ok=True)

def scroll_and_get_links(scroll_times=10):
    driver.get(base_url)
    time.sleep(3)

    for i in range(scroll_times):
        print(f"Scrolling {i+1}/{scroll_times}")
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)  # wait for news to load

    # After scrolling, get page source
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.select('div.newshead4.mar4B.lettersp2 > a')


        
    

    links = []
    for a in articles:
        
        url = "https://www.aastocks.com" + a['href']
        if i ==1:
            print(url)
        title = a.get_text(strip=True)
        links.append((title, url))
    return links

def scrape_article(url):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            content_div = soup.select_one('div.newscontent5.fLevel3')
            if content_div:
                return content_div.get_text(separator="\n", strip=True)
            else:
                return ""
        else:
            print(f"❌ Failed to connect {url}. Status code: {res.status_code}")
            return ""
    except Exception as e:
        print(f"❌ Error scraping article: {e}")
        return ""

def handle_content(content):
    if "AASTOCKS新聞" in content:
        content = content.split("AASTOCKS新聞")[0]
    if "新聞來源" in content:
        content = content.split("新聞來源")[0]
    return content

def run_scraper(limit=500, scroll_times=10):
    news_links = scroll_and_get_links(scroll_times=scroll_times)
    #print(news_links)

    for i, (title, url) in enumerate(news_links[:limit]):
        print(f"Scraping {i+1}: {title}")
        content = scrape_article(url)
        content = handle_content(content)
        if content:
            filename = f"news_articles/news_{i+1}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(title + "\n\n" + content)
        time.sleep(1)  # Be polite


    driver.quit()

run_scraper(limit=50, scroll_times=5)
#input("Press ENTER to exit and close the browser...")
