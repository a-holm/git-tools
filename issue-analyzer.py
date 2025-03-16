import requests
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time
from itertools import combinations
from collections import defaultdict
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ======== CONFIGURATION ========
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "mannaandpoem"
REPO_NAME = "OpenManus"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"  # Set model here
RATE_LIMIT = 2000  # Requests per minute
# ===============================

# Initialize rate limiting
MIN_INTERVAL = 60 / RATE_LIMIT  # Seconds between requests
last_request_time = None

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def rate_limited_api_call(**kwargs):
    """Make API call with rate limiting"""
    global last_request_time
    
    # Enforce rate limit
    if last_request_time is not None:
        elapsed = time.time() - last_request_time
        if elapsed < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - elapsed
            time.sleep(sleep_time)
    
    # Make API call
    response = client.chat.completions.create(**kwargs)
    last_request_time = time.time()
    return response

CATEGORIES = [
    "Bug", "Feature Request", "Documentation", "Question",
    "Performance Issue", "Integration Request", "UI/UX",
    "API Issue", "Installation Issue", "Security Issue"
]

def fetch_open_issues():
    """Fetch only open issues from the repository."""
    open_issues = []
    page = 1
    
    while True:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
        params = {"state": "open", "per_page": 100, "page": page}  # Changed "all" to "open"
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching issues: {response.status_code}")
            print(response.text)
            break
            
        issues = response.json()
        if not issues:
            break
            
        # Filter out pull requests (GitHub API returns PRs as issues)
        actual_issues = [issue for issue in issues if "pull_request" not in issue]
        open_issues.extend(actual_issues)
        
        page += 1
        
    print(f"Found {len(open_issues)} open issues")
    return open_issues

def categorize_issue_with_ai(issue):
    """Use Gemini (via OpenAI compatibility) to categorize an issue."""
    title = issue["title"]
    body = issue.get("body", "") or ""
    
    content = f"Title: {title}\nBody: {body}"
    
    try:
        response = rate_limited_api_call(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a GitHub issue categorizer. Categorize the following GitHub issue into exactly one of these categories: {', '.join(CATEGORIES)}. Respond with only the category name and nothing else."
                },
                {"role": "user", "content": f"Title: {title}\nBody: {body}"}
            ],
        )
        
        category = response.choices[0].message.content.strip()
        
        # Ensure the returned category is one of our valid categories
        if category not in CATEGORIES:
            closest_category = min(CATEGORIES, key=lambda x: len(set(x.lower()) - set(category.lower())))
            return closest_category
        
        return category
    except Exception as e:
        print(f"Error categorizing issue {issue['number']}: {e}")
        return "Uncategorized"
    
def assess_issue_difficulty(issue):
    """Use Gemini AI to assess if the issue is an easy fix and provide notes."""
    title = issue["title"]
    body = issue.get("body", "") or ""
    
    content = f"Title: {title}\nBody: {body}"
    
    try:
        response = rate_limited_api_call(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at assessing GitHub issues. Determine if the following issue is an easy fix. If it is, provide a brief note on how to fix it. Respond with 'Easy Fix: Yes' or 'Easy Fix: No' followed by 'Notes: ' and your notes if applicable."
                },
                {"role": "user", "content": content}
            ],
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse the response
        if "Easy Fix: Yes" in response_text:
            is_easy_fix = True
            notes = response_text.split("Notes: ")[1] if "Notes: " in response_text else ""
        else:
            is_easy_fix = False
            notes = ""
        
        return is_easy_fix, notes
    except Exception as e:
        print(f"Error assessing issue {issue['number']}: {e}")
        return False, ""

def find_duplicates(issues, threshold=0.85):
    """Find potential duplicate issues using TF-IDF and cosine similarity."""
    issue_texts = [f"{issue['title']} {issue.get('body', '') or ''}" for issue in issues]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(issue_texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find potential duplicates
    potential_duplicates = []
    for i in range(len(issues)):
        for j in range(i+1, len(issues)):
            if similarity_matrix[i, j] > threshold:
                potential_duplicates.append({
                    'issue1_number': issues[i]['number'],
                    'issue1_title': issues[i]['title'],
                    'issue2_number': issues[j]['number'],
                    'issue2_title': issues[j]['title'],
                    'similarity': similarity_matrix[i, j]
                })
    
    return potential_duplicates

def find_ai_suggested_duplicates(issues, max_pairs=100):
    """Use Gemini AI to identify potential duplicate issues."""
    ai_suggested_duplicates = []
    pairs_checked = 0
    
    total_pairs = len(issues) * (len(issues) - 1) // 2
    print(f"Checking all {total_pairs} possible pairs...")
    
    for i, j in combinations(range(len(issues)), 2):
        # print for every 1000 pairs
        if pairs_checked % 1000 == 0:
            print(f"Checked {pairs_checked} pairs so far...")

        issue1 = issues[i]
        issue2 = issues[j]
        
        try:
            response = rate_limited_api_call(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying duplicate GitHub issues. Analyze the two issues below and determine if they are duplicates. Respond with only 'yes' or 'no'."
                    },
                    {
                        "role": "user",
                        "content": f"Issue 1 (#{issue1['number']}):\nTitle: {issue1['title']}\nBody: {issue1.get('body', '') or ''}\n\nIssue 2 (#{issue2['number']}):\nTitle: {issue2['title']}\nBody: {issue2.get('body', '') or ''}"
                    }
                ],
            )
            
            is_duplicate = response.choices[0].message.content.strip().lower()
            
            if is_duplicate == 'yes':
                print(f" potential duplicate found between issues #{issue1['number']} and #{issue2['number']}")
                ai_suggested_duplicates.append({
                    'issue1_number': issue1['number'],
                    'issue1_title': issue1['title'],
                    'issue2_number': issue2['number'],
                    'issue2_title': issue2['title'],
                    'ai_confidence': 'High'
                })
            
            pairs_checked += 1
            
        except Exception as e:
            print(f"Error checking duplicates for issues #{issue1['number']} and #{issue2['number']}: {e}")
    
    return ai_suggested_duplicates

def main():
    print("Fetching issues...")
    issues = fetch_open_issues()
    print(f"Found {len(issues)} issues")
    
    # Categorize issues
    print("Categorizing issues...")
    categorized_issues = []
    category_counts = defaultdict(int)
    
    for i, issue in enumerate(issues):
        print(f"Categorizing issue {i+1}/{len(issues)}: #{issue['number']} - {issue['title']}")
        category = categorize_issue_with_ai(issue)
        category_counts[category] += 1
        
        # Assess issue difficulty
        is_easy_fix, fix_notes = assess_issue_difficulty(issue)
        
        categorized_issues.append({
            'number': issue['number'],
            'title': issue['title'],
            'state': issue['state'],
            'created_at': issue['created_at'],
            'updated_at': issue['updated_at'],
            'url': issue['html_url'],
            'category': category,
            'is_easy_fix': is_easy_fix,  # New column
            'fix_notes': fix_notes      # New column
        })
    
    # Find potential duplicates using TF-IDF
    print("Finding potential duplicates using TF-IDF...")
    tfidf_duplicates = find_duplicates(issues)
    
    # Find potential duplicates using AI
    print("Finding potential duplicates using AI...")
    ai_duplicates = find_ai_suggested_duplicates(issues)
    
    # Output results
    df_issues = pd.DataFrame(categorized_issues)
    df_issues.to_csv("categorized_issues.csv", index=False)
    
    df_tfidf_duplicates = pd.DataFrame(tfidf_duplicates)
    if not df_tfidf_duplicates.empty:
        df_tfidf_duplicates = df_tfidf_duplicates.sort_values('similarity', ascending=False)
        df_tfidf_duplicates.to_csv("tfidf_potential_duplicates.csv", index=False)
    
    df_ai_duplicates = pd.DataFrame(ai_duplicates)
    if not df_ai_duplicates.empty:
        df_ai_duplicates.to_csv("ai_suggested_duplicates.csv", index=False)
    
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} issues ({count/len(issues):.1%})")
    
    print(f"\nFound {len(tfidf_duplicates)} potential duplicate pairs using TF-IDF")
    print(f"Found {len(ai_duplicates)} potential duplicate pairs using AI")
    print("\nResults saved to CSV files")

if __name__ == "__main__":
    main()