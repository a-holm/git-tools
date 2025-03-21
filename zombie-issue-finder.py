import requests
import re
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
import base64
from openai import OpenAI
import os
import json
from collections import Counter

# Load environment variables from .env file
load_dotenv()

# ======== CONFIGURATION ========
# GitHub API configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Set this in your environment variables
REPO_OWNER = "dask"
REPO_NAME = "dask"

# Gemini API setup via OpenAI compatibility layer
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Set this in your environment variables
MODEL_NAME = "gemini-2.0-flash"  # Flash model name
THINKING_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"  # Thinking model name for advanced analysis
RATE_LIMIT = 2000  # Requests per minute
# ===============================

# Initialize rate limiting
MIN_INTERVAL = 60 / RATE_LIMIT  # Seconds between requests
last_request_time = None

# Initialize OpenAI client for Gemini
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
base_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

def rate_limited_api_call(**kwargs):
    """Make API call with rate limiting"""
    global last_request_time

    # Enforce rate limit
    if last_request_time is not None:
        elapsed = time.time() - last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
    
    response = client.chat.completions.create(**kwargs)
    last_request_time = time.time()
    return response

def multiple_flash_calls(messages, samples=5, model=MODEL_NAME, response_format={"type": "json_object"}):
    """
    Make multiple API calls using the flash model and return a list of response texts.
    Default samples=5.
    """
    responses = []
    for i in range(samples):
        try:
            response = rate_limited_api_call(
                model=model,
                messages=messages,
                response_format=response_format
            )
            content = response.choices[0].message.content.strip()
            responses.append(content)
        except Exception as e:
            print(f"Error during flash API call {i+1}/{samples}: {e}")
    return responses

def get_open_issues():
    """Fetch all open issues from the repository"""
    url = f"{base_url}/issues"
    params = {"state": "open", "per_page": 100}
    open_issues = []
    
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching issues: {response.status_code}")
            print(response.text)
            break
            
        page_issues = response.json()
        if not page_issues:
            break
            
        # Filter out pull requests (GitHub API returns PRs as issues)
        actual_issues = [issue for issue in page_issues if "pull_request" not in issue]
        open_issues.extend(actual_issues)
        
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    
    print(f"Found {len(open_issues)} open issues")
    return open_issues

def get_all_pull_requests():
    """Fetch all relevant PRs (open PRs and closed PRs that were merged)"""
    all_prs = []
    
    # Fetch open PRs
    print("Fetching open PRs...")
    url = f"{base_url}/pulls"
    params = {"state": "open", "per_page": 100}
    
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching open PRs: {response.status_code}")
            break
            
        page_prs = response.json()
        if not page_prs:
            break
            
        for pr in page_prs:
            if pr["base"]["ref"] != 'main':  # Only include PRs targeting main
                continue
            all_prs.append({
                "number": pr["number"],
                "title": pr["title"],
                "state": "open",
                "merged": False,
                "merged_at": None,
                "created_at": pr["created_at"],
                "base_branch": pr["base"]["ref"],
                "url": pr["html_url"],
                "body": pr.get("body", "")
            })
        
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    
    # Fetch closed and merged PRs
    print("Fetching merged PRs...")
    url = f"{base_url}/pulls"
    params = {"state": "closed", "per_page": 100}
    
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching closed PRs: {response.status_code}")
            break
            
        page_prs = response.json()
        if not page_prs:
            break
            
        for pr in page_prs:
            if pr.get("merged_at"):  # Only include actually merged PRs
                all_prs.append({
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": "closed",
                    "merged": True,
                    "merged_at": pr["merged_at"],
                    "created_at": pr["created_at"],
                    "base_branch": pr["base"]["ref"],
                    "url": pr["html_url"],
                    "body": pr.get("body", "")
                })
        
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    
    open_count = len([p for p in all_prs if p['state'] == 'open'])
    merged_count = len([p for p in all_prs if p['merged']])
    print(f"Found {len(all_prs)} relevant PRs ({open_count} open, {merged_count} merged)")
    return all_prs

def get_pr_commits(pr_number):
    """Get commits for a specific PR"""
    url = f"{base_url}/pulls/{pr_number}/commits"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching commits for PR #{pr_number}: {response.status_code}")
        return []

def get_commit_diff(commit_sha):
    """Get the diff for a specific commit"""
    url = f"{base_url}/commits/{commit_sha}"
    headers_with_diff = headers.copy()
    headers_with_diff["Accept"] = "application/vnd.github.v3.diff"
    
    response = requests.get(url, headers=headers_with_diff)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching diff for commit {commit_sha}: {response.status_code}")
        return ""

def ai_analyze_potential_fixes(issue, pr_data):
    """
    Use Gemini flash model (with multiple calls) to analyze if a PR potentially fixes an issue.
    If flash analysis yields a likely fix with acceptable confidence ("Certain" or "High"),
    then double-check using the advanced thinking model.
    """
    issue_title = issue["title"]
    issue_body = issue.get("body", "") or ""
    issue_number = issue["number"]
    
    pr_title = pr_data["title"]
    pr_body = pr_data.get("body", "") or ""
    pr_number = pr_data["number"]
    
    try:
        commits = get_pr_commits(pr_number)
    except Exception as e:
        time.sleep(30)
        commits = get_pr_commits(pr_number)
    
    commit_diffs = []
    for commit in commits[:10]:
        commit_sha = commit["sha"]
        commit_message = commit["commit"]["message"]
        try:
            diff = get_commit_diff(commit_sha)
        except Exception as e:
            time.sleep(30)
            diff = get_commit_diff(commit_sha)
        
        if len(diff) > 7000:
            diff = diff[:7000] + "... [diff truncated]"
            
        commit_diffs.append({
            "message": commit_message,
            "diff": diff
        })
    
    prompt = f"""
I need you to analyze if a specific GitHub Pull Request (PR) might have already fixed an open issue.

ISSUE #{issue_number}:
Title: {issue_title}
Description: {issue_body[:6000]}{"..." if len(issue_body) > 6000 else ""}

PULL REQUEST #{pr_number}:
Title: {pr_title}
Description: {pr_body[:6000]}{"..." if len(pr_body) > 6000 else ""}

COMMITS & CHANGES:
"""
    for i, commit in enumerate(commit_diffs):
        prompt += f"\nCOMMIT {i+1}:\n"
        prompt += f"Message: {commit['message']}\n"
        prompt += f"Changes:\n{commit['diff'][:5000]}{'...' if len(commit['diff']) > 5000 else ''}\n"
    
    prompt += """
Based on the issue description and the changes in the PR, please analyze:
1. Does this PR appear to fix the issue described?
2. What specific changes in the PR address the issue, if any?
3. How confident are you that this PR resolves the issue (Certain, High, Medium, Low)?

Respond in JSON format with these keys:
{
    "likely_fixes_issue": true/false,
    "confidence": "Certain/High/Medium/Low",
    "reasoning": "Your detailed explanation here",
    "relevant_changes": "Description of specific changes that address the issue"
}
"""
    messages = [{"role": "user", "content": prompt}]
    flash_responses = multiple_flash_calls(messages, samples=5, model=MODEL_NAME, response_format={"type": "json_object"})
    
    parsed_responses = []
    for response_text in flash_responses:
        try:
            parsed = json.loads(response_text)
            parsed_responses.append(parsed)
        except json.JSONDecodeError:
            print(f"Error parsing flash response as JSON for issue #{issue_number} and PR #{pr_number}")
    
    if parsed_responses:
        normalized = [json.dumps(resp, sort_keys=True) for resp in parsed_responses]
        most_common, count = Counter(normalized).most_common(1)[0]
        analysis_json = json.loads(most_common)
    else:
        analysis_json = {
            "likely_fixes_issue": False,
            "confidence": "Low",
            "reasoning": "No valid response from flash calls",
            "relevant_changes": ""
        }
    
    # Only double-check promising candidates (ignore those falling to "Medium" after flash)
    if analysis_json.get("likely_fixes_issue") and analysis_json.get("confidence") in ["Certain", "High"]:
        thinking_prompt = f"""
Double-check the following analysis of whether a Pull Request fixes a GitHub issue.

ISSUE #{issue_number}:
Title: {issue_title}
Description: {issue_body[:6000]}{"..." if len(issue_body) > 6000 else ""}

PULL REQUEST #{pr_number}:
Title: {pr_title}
Description: {pr_body[:6000]}{"..." if len(pr_body) > 6000 else ""}

COMMITS & CHANGES:
"""
        for i, commit in enumerate(commit_diffs):
            thinking_prompt += f"\nCOMMIT {i+1}:\n"
            thinking_prompt += f"Message: {commit['message']}\n"
            thinking_prompt += f"Changes:\n{commit['diff'][:5000]}{'...' if len(commit['diff']) > 5000 else ''}\n"
        
        thinking_prompt += f"""
Flash analysis (aggregated):
{json.dumps(analysis_json, indent=2)}

Now, based on the above information, please provide your final determination on whether the PR fixes the issue. 
Please respond in JSON format with these keys:
{{
   "likely_fixes_issue": true/false,
   "confidence": "Certain/High/Medium/Low",
   "reasoning": "Your detailed explanation",
   "relevant_changes": "Description of changes that address the issue"
}}
"""
        try:
            thinking_response = rate_limited_api_call(
                model=THINKING_MODEL_NAME,
                messages=[{"role": "user", "content": thinking_prompt}],
                response_format={"type": "json_object"}
            )
            thinking_analysis_text = thinking_response.choices[0].message.content.strip()
            thinking_analysis = json.loads(thinking_analysis_text)
            return thinking_analysis
        except Exception as e:
            print(f"Error in double-check thinking model for issue #{issue_number} and PR #{pr_number}: {e}")
            return analysis_json
    else:
        # Even if flash analysis indicated medium confidence, we now ignore it.
        return analysis_json

def find_zombie_issues():
    """Find zombie issues - open issues that may have been fixed by PRs.
       Only candidates with final confidence of 'Certain' or 'High' are retained.
    """
    open_issues = get_open_issues()
    relevant_prs = get_all_pull_requests()
    
    zombie_issues_confident_merged_prs = []
    zombie_issues_confident_open_prs = []
    
    for issue in open_issues:
        issue_number = issue["number"]
        print(f"Analyzing issue #{issue_number}: {issue['title']}")
        
        for pr in relevant_prs:
            # Skip PRs merged before the issue was created
            issue_created_at = datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            if pr["merged"] and datetime.strptime(pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ") < issue_created_at:
                continue
            
            analysis = ai_analyze_potential_fixes(issue, pr)
            
            # Only retain candidates with 'Certain' or 'High' confidence
            if analysis.get("likely_fixes_issue") and analysis.get("confidence") in ["Certain", "High"]:
                candidate = {
                    # Issue information
                    "issue_number": issue["number"],
                    "issue_title": issue["title"],
                    "issue_url": issue["html_url"],
                    "issue_created_at": issue["created_at"],
                    # PR information
                    "pr_number": pr["number"],
                    "pr_title": pr["title"],
                    "pr_url": pr["url"],
                    "pr_state": pr["state"],
                    "pr_merged": pr["merged"],
                    "pr_merged_at": pr.get("merged_at"),
                    "pr_base_branch": pr["base_branch"],
                    "pr_created_at": pr["created_at"],
                    # Analysis results
                    "confidence": analysis.get("confidence"),
                    "reasoning": analysis.get("reasoning"),
                    "relevant_changes": analysis.get("relevant_changes")
                }
                
                if pr["merged"]:
                    zombie_issues_confident_merged_prs.append(candidate)
                else:
                    zombie_issues_confident_open_prs.append(candidate)
                
                print(f"    ✓ Potential fix found with {analysis.get('confidence')} confidence")
                
                # If a high-confidence merged fix targeting main is found, break for this issue.
                if (analysis.get("confidence") in ["Certain", "High"]
                    and pr["merged"] 
                    and pr["base_branch"] == "main"):
                    print(f"    🚨 {analysis.get('confidence')} confidence merged fix for main branch - moving to next issue")
                    break
            else:
                # Ignore candidate if confidence is Medium (or lower)
                print(f"    ✗ Ignoring candidate with {analysis.get('confidence')} confidence")
    
    return zombie_issues_confident_merged_prs, zombie_issues_confident_open_prs

def main():
    print("Finding zombie issues (open issues that are likely fixed by high-confidence PRs)...")
    zombie_issues_merged, zombie_issues_open = find_zombie_issues()
    
    print(f"\nFound {len(zombie_issues_merged)} zombie issues likely fixed by merged PRs (Certain/High confidence)")
    print(f"Found {len(zombie_issues_open)} zombie issues likely fixed by open PRs (Certain/High confidence)")
    
    # Save merged PR candidates
    if zombie_issues_merged:
        df_merged = pd.DataFrame(zombie_issues_merged)
        df_merged.to_csv("zombie_issues_confident_merged_prs.csv", index=False)
        print("Zombie issues fixed by merged PRs saved to zombie_issues_confident_merged_prs.csv")
        print("\nSummary of zombie issues fixed by merged PRs:")
        for i, issue in enumerate(zombie_issues_merged):
            print(f"{i+1}. Issue #{issue['issue_number']}: {issue['issue_title']}")
            print(f"   Fixed by PR #{issue['pr_number']}: {issue['pr_title']}")
            print(f"   Confidence: {issue['confidence']}")
            print()
    else:
        print("No zombie issues fixed by merged PRs found.")
    
    # Save open PR candidates
    if zombie_issues_open:
        df_open = pd.DataFrame(zombie_issues_open)
        df_open.to_csv("zombie_issues_confident_open_prs.csv", index=False)
        print("Zombie issues fixed by open PRs saved to zombie_issues_confident_open_prs.csv")
        print("\nSummary of zombie issues fixed by open PRs:")
        for i, issue in enumerate(zombie_issues_open):
            print(f"{i+1}. Issue #{issue['issue_number']}: {issue['issue_title']}")
            print(f"   Potentially fixed by PR #{issue['pr_number']}: {issue['pr_title']}")
            print(f"   Confidence: {issue['confidence']}")
            print()
    else:
        print("No zombie issues fixed by open PRs found.")

if __name__ == "__main__":
    main()