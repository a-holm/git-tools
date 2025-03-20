import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time
from itertools import combinations
from collections import defaultdict, Counter
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a filter to skip messages starting with "HTTP Request:"
class SkipHttpRequestFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("HTTP Request:")

# Attach the filter to all handlers of the root logger
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(SkipHttpRequestFilter())

# Optionally, also raise the log level for specific known loggers:
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# ======== CONFIGURATION ========
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "dask"
REPO_NAME = "dask"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODELS = {
    "flash": {
        "name": "gemini-2.0-flash",
        "rate_limit": 2000  # 2000 RPM
    },
    "thinking-exp": {
        "name": "gemini-2.0-flash-thinking-exp-01-21",
        "rate_limit": 10  # 10 RPM
    }
}
# ===============================

# Initialize rate limiting
last_request_times = {
    "flash": None,
    "thinking-exp": None
}

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

CATEGORIES = [
    "Bug", "Feature Request", "Documentation", "Question",
    "Performance Issue", "Integration Request", "UI/UX",
    "API Issue", "Installation Issue", "Security Issue"
]

def rate_limited_api_call(model_type, **kwargs):
    """Make API call with rate limiting for a specific model type."""
    global last_request_times

    model_config = MODELS[model_type]
    min_interval = 60 / model_config["rate_limit"]
    last_time = last_request_times[model_type]
    
    if last_time is not None:
        elapsed = time.time() - last_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    
    response = client.chat.completions.create(
        model=model_config["name"],
        **kwargs
    )
    last_request_times[model_type] = time.time()
    return response

def multiple_flash_calls(messages, samples=3):
    """
    Helper function that makes multiple API calls using the flash model
    and returns a list of response texts.
    """
    responses = []
    for i in range(samples):
        try:
            response = rate_limited_api_call(
                model_type="flash",
                messages=messages
            )
            responses.append(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error("Error during flash API call %d/%d: %s", i+1, samples, e)
    return responses

def fetch_open_issues():
    """Fetch only open issues from the repository."""
    open_issues = []
    page = 1
    while True:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
        params = {"state": "open", "per_page": 100, "page": page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.error("Error fetching issues: %s - %s", response.status_code, response.text)
            break
        issues = response.json()
        if not issues:
            break
        actual_issues = [issue for issue in issues if "pull_request" not in issue]
        open_issues.extend(actual_issues)
        page += 1
    logger.info("Found %d open issues", len(open_issues))
    return open_issues

def categorize_issue_with_ai(issue, samples=3):
    """
    Use multiple flash API calls to categorize an issue with majority vote.
    """
    title = issue["title"]
    body = issue.get("body", "") or ""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a GitHub issue categorizer. Categorize the following GitHub issue "
                f"into exactly one of these categories: {', '.join(CATEGORIES)}. "
                "Respond with only the category name and nothing else."
            )
        },
        {"role": "user", "content": f"Title: {title}\nBody: {body}"}
    ]
    
    responses = multiple_flash_calls(messages, samples=samples)
    valid_responses = [resp for resp in responses if resp in CATEGORIES]
    
    if valid_responses:
        category = Counter(valid_responses).most_common(1)[0][0]
    else:
        first_response = responses[0] if responses else ""
        category = min(CATEGORIES, key=lambda x: len(set(x.lower()) - set(first_response.lower())))
    return category

def assess_issue_difficulty(issue, flash_samples=3):
    """
    Assess if the issue is an easy fix and provide detailed notes on how to solve it.
    
    Step 1: Use multiple flash calls to quickly determine the easy-fix status via a majority vote.
    Step 2: If deemed an easy fix, delegate to the advanced thinking model to provide the complete solution.
    """
    title = issue["title"]
    body = issue.get("body", "") or ""
    content = f"Title: {title}\nBody: {body}"
    
    logger.info("Assessing difficulty for issue #%s", issue.get("number"))
    
    # Step 1: Quick assessment using flash calls
    flash_messages = [
        {
            "role": "system",
            "content": (
                "You are a GitHub issue assessor. Determine if the following issue is an easy fix. "
                "Respond only with 'Easy Fix: Yes' or 'Easy Fix: No'."
            )
        },
        {"role": "user", "content": content}
    ]
    try:
        flash_responses = multiple_flash_calls(flash_messages, samples=flash_samples)
        yes_votes = sum(1 for resp in flash_responses if "easy fix: yes" in resp.lower())
        is_easy_fix = yes_votes >= (flash_samples // 2 + 1)
        logger.info("Flash assessment for issue #%s: %s (Yes votes: %d/%d)",
                    issue.get("number"),
                    "Easy Fix" if is_easy_fix else "Not Easy",
                    yes_votes,
                    flash_samples)
    except Exception as e:
        logger.error("Error during flash assessment for issue #%s: %s", issue.get("number"), e)
        return False, ""
    
    # Step 2: If easy fix, use thinking-exp model for detailed instructions
    final_notes = ""
    if is_easy_fix:
        thinking_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at providing fixes for GitHub issues. "
                    "Given the following issue, provide an in-depth explanation of why it is easy to fix, "
                    "and give a detailed step-by-step guide on how to solve it including code snippets if applicable. "
                    "Your response should begin with 'Notes:' and then provide the detailed instructions."
                )
            },
            {"role": "user", "content": content}
        ]
        try:
            thinking_response = rate_limited_api_call(
                model_type="thinking-exp",
                messages=thinking_messages
            )
            final_notes = thinking_response.choices[0].message.content.strip()
            logger.info("Thinking model provided detailed notes for issue #%s", issue.get("number"))
        except Exception as e:
            logger.error("Error obtaining details from thinking model for issue #%s: %s", issue.get("number"), e)
    
    return is_easy_fix, final_notes

def find_duplicates(issues, threshold=0.85):
    """Find potential duplicate issues using TF-IDF and cosine similarity."""
    issue_texts = [f"{issue['title']} {issue.get('body', '') or ''}" for issue in issues]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(issue_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
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

def find_ai_suggested_duplicates(issues, samples=3):
    """
    Two-stage duplicate detection using flash calls for initial screening and an advanced check with thinking-exp.
    """
    logger.info("Starting two-stage duplicate detection...")
    
    # Stage 1: Initial screening with flash model
    candidates = []
    total_pairs = len(issues) * (len(issues) - 1) // 2
    for idx, (i, j) in enumerate(combinations(range(len(issues)), 2)):
        if idx % 100 == 0:
            logger.info("Processed %d/%d pairs for duplicate detection...", idx, total_pairs)
        issue1 = issues[i]
        issue2 = issues[j]
        messages = [
            {
                "role": "system",
                "content": (
                    "Thoroughly analyze if these are duplicate GitHub issues. "
                    "Consider technical details and context. Respond with only 'yes' or 'no'."
                )
            },
            {
                "role": "user",
                "content": f"Issue 1 (#{issue1['number']}): {issue1['title']}\nIssue 2 (#{issue2['number']}): {issue2['title']}"
            }
        ]
        responses = multiple_flash_calls(messages, samples=samples)
        yes_votes = sum(1 for resp in responses if resp.lower() == 'yes')
        if yes_votes >= (samples // 2 + 1):
            candidates.append((i, j))
    
    # Stage 2: Rigorous check with the advanced thinking model
    verified = []
    for i, j in candidates:
        issue1 = issues[i]
        issue2 = issues[j]
        try:
            response = rate_limited_api_call(
                model_type="thinking-exp",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Thoroughly analyze if these are duplicate GitHub issues. "
                            "Consider technical details and context. Respond with only 'yes' or 'no'."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Issue 1 (#{issue1['number']}):\nTitle: {issue1['title']}\nBody: {issue1.get('body', '')}\n\n"
                            f"Issue 2 (#{issue2['number']}):\nTitle: {issue2['title']}\nBody: {issue2.get('body', '')}"
                        )
                    }
                ]
            )
            if response.choices[0].message.content.strip().lower() == 'yes':
                verified.append({
                    'issue1_number': issue1['number'],
                    'issue1_title': issue1['title'],
                    'issue2_number': issue2['number'],
                    'issue2_title': issue2['title'],
                    'verification': 'Two-Stage AI Verified'
                })
        except Exception as e:
            logger.error("Error in stage 2 duplicate check for issues #%s and #%s: %s", 
                         issue1['number'], issue2['number'], e)
    
    return verified

def main():
    logger.info("Fetching issues...")
    issues = fetch_open_issues()
    logger.info("Found %d issues", len(issues))
    
    logger.info("Categorizing issues...")
    categorized_issues = []
    category_counts = defaultdict(int)
    
    for i, issue in enumerate(issues):
        logger.info("Categorizing issue %d/%d: #%s - %s", 
                    i+1, len(issues), issue['number'], issue['title'])
        category = categorize_issue_with_ai(issue, samples=3)
        category_counts[category] += 1
        
        # Assess issue difficulty using our improved approach:
        # Step 1: Fast flash assessment then detailed instructions from the thinking model.
        is_easy_fix, fix_notes = assess_issue_difficulty(issue, flash_samples=3)
        categorized_issues.append({
            'number': issue['number'],
            'is_easy_fix': is_easy_fix,
            'fix_notes': fix_notes,
            'title': issue['title'],
            'state': issue['state'],
            'created_at': issue['created_at'],
            'updated_at': issue['updated_at'],
            'url': issue['html_url'],
            'category': category
        })
    
    df_issues = pd.DataFrame(categorized_issues)
    df_issues.to_csv("categorized_issues.csv", index=False)
    
    logger.info("Finding potential duplicates using TF-IDF...")
    tfidf_duplicates = find_duplicates(issues)
    
    logger.info("Finding potential duplicates using AI...")
    ai_duplicates = find_ai_suggested_duplicates(issues, samples=3)
    
    df_tfidf_duplicates = pd.DataFrame(tfidf_duplicates)
    if not df_tfidf_duplicates.empty:
        df_tfidf_duplicates = df_tfidf_duplicates.sort_values('similarity', ascending=False)
        df_tfidf_duplicates.to_csv("tfidf_potential_duplicates.csv", index=False)
    
    df_ai_duplicates = pd.DataFrame(ai_duplicates)
    if not df_ai_duplicates.empty:
        df_ai_duplicates.to_csv("ai_suggested_duplicates.csv", index=False)
    
    logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info("%s: %d issues (%.1f%%)", category, count, count/len(issues)*100)
    
    logger.info("Found %d potential duplicate pairs using TF-IDF", len(tfidf_duplicates))
    logger.info("Found %d potential duplicate pairs using AI", len(ai_duplicates))
    logger.info("Results saved to CSV files")

if __name__ == "__main__":
    main()