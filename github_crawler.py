import os
import requests
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

def fetch_python_repositories(query="machine learning", max_repos=5):
    params = {
        "q": f"{query} language:python",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos
    }
    response = requests.get("https://api.github.com/search/repositories", headers=HEADERS, params=params)
    return response.json()
