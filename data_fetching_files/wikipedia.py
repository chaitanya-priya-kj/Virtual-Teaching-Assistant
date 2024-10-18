import os
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup


def search_wikipedia(keyword):
    base_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": keyword,
        "utf8": "1"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    search_results = data['query']['search']
    if not search_results:
        print(f"No Wikipedia page found for keyword: {keyword}")
        return []

    titles = [result['title'] for result in search_results]

    return titles

def fetch_page_content(title):
    base_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exlimit": "max",
        "explaintext": "1",
        "titles": title,
        "utf8": "1"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    page = next(iter(data['query']['pages'].values()))
    content = page['extract']

    return content


def scrape_text_with_keywords(keywords):
    scraped_text = ""

    for keyword in keywords:
        page_titles = search_wikipedia(keyword)

        if not page_titles:
            continue

        for title in page_titles:
            page_content = fetch_page_content(title)

            if page_content:
                scraped_text += "\n\n" + page_content

    return scraped_text


def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


if __name__ == "__main__":
    keywords = ["Machine learning", "Artificial intelligence", "Data Science"]
    scraped_text = scrape_text_with_keywords(keywords)

    if scraped_text:
        print("Scraped text before cleaning:")
        # print(scraped_text)

        cleaned_text = clean_text(scraped_text)

        print("\n\nScraped text after cleaning:")
        # print(cleaned_text)

        # Create a folder named study_material if it doesn't exist
        folder_path = 'new_material_text'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Write the scraped text to a text file
        with open(os.path.join(folder_path, 'wikipedia.txt'), 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        print("\nStudy material saved successfully.")
    else:
        print("No text scraped.")
