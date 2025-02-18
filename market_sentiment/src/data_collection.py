import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


def fetch_news_headlines(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all relevant tags
    tags = soup.find_all(["h1", "h2", "h3", "a"])

    # Filter headlines: assuming headlines are longer and contain more than 50 characters
    headlines = [
        tag.get_text(strip=True) for tag in tags if len(tag.get_text(strip=True)) > 50
    ]

    unique_headlines = list(set(headlines))

    # Remove non-headline items (e.g., navigation or language options)
    filtered_headlines = [
        headline
        for headline in unique_headlines
        if not any(
            keyword in headline.lower()
            for keyword in [
                "language",
                "facebook",
                "twitter",
                "linkedin",
                "home",
                "market",
            ]
        )
    ]

    return filtered_headlines


def fetch_news_articles():
    urls = [
        # Economic Times Feeds
        "https://economictimes.indiatimes.com",
        # Business Standard Feeds
        "https://www.business-standard.com",
        # LiveMint Feeds
        "https://www.livemint.com",
        # the hindu
        "https://www.thehindubusinessline.com",
    ]

    all_headlines = []
    for url in urls:
        headlines = fetch_news_headlines(url)
        all_headlines.extend(headlines)

    articles = [{"title": headline} for headline in all_headlines]
    return pd.DataFrame(articles)


if __name__ == "__main__":
    articles = fetch_news_articles()

    # Ensure the directory exists
    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # Save the articles to a CSV file
    articles.to_csv(f"{output_dir}/general_news_articles.csv", index=False)
