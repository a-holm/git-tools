import requests
import re
import pandas as pd
from datetime import datetime
import time
import base64
from openai import OpenAI
import os

# ======== CONFIGURATION ========
# GitHub API configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Set this in your environment variables
REPO_OWNER = "mannaandpoem"
REPO_NAME = "OpenManus"

# Gemini API setup via OpenAI compatibility layer
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Set this in your environment variables
MODEL_NAME = "gemini-2.0-flash"  # Set the model here
RATE_LIMIT = 10  # Requests per minute
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
            sleep_time = MIN_INTERVAL - elapsed
            time.sleep(sleep_time)
    
    # Make API call
    response = client.chat.completions.create(**kwargs)
    last_request_time = time.time()
    return response

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
        
        # Check for pagination
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break
    
    print(f"Found {len(open_issues)} open issues")
    return open_issues

def get_merged_pull_requests(limit=30):
    """Fetch recent merged pull requests"""
    url = f"{base_url}/pulls"
    params = {"state": "closed", "per_page": 100}
    merged_prs = []
    
    while True and len(merged_prs) < limit:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching PRs: {response.status_code}")
            break
            
        page_prs = response.json()
        if not page_prs:
            break
            
        # Filter to only include merged PRs
        merged_page_prs = [pr for pr in page_prs if pr.get("merged_at")]
        merged_prs.extend(merged_page_prs[:limit - len(merged_prs)])
        
        # Check for pagination
        if 'next' in response.links and len(merged_prs) < limit:
            url = response.links['next']['url']
        else:
            break
    
    print(f"Found {len(merged_prs)} merged pull requests")
    return merged_prs

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
    """Use Gemini to analyze if a PR potentially fixes an issue"""
    issue_title = issue["title"]
    issue_body = issue.get("body", "") or ""
    issue_number = issue["number"]
    
    pr_title = pr_data["title"]
    pr_body = pr_data.get("body", "") or ""
    pr_number = pr_data["number"]
    
    # Get the commits from this PR
    commits = get_pr_commits(pr_number)
    
    # Get diffs for the commits
    commit_diffs = []
    for commit in commits[:3]:  # Limit to first 3 commits to avoid too much content
        commit_sha = commit["sha"]
        commit_message = commit["commit"]["message"]
        diff = get_commit_diff(commit_sha)
        
        # Limit diff size to avoid token limits
        if len(diff) > 3000:
            diff = diff[:3000] + "... [diff truncated]"
            
        commit_diffs.append({
            "message": commit_message,
            "diff": diff
        })
    
    # Prepare the prompt for Gemini
    prompt = f"""
    I need you to analyze if a specific GitHub Pull Request (PR) might have already fixed an open issue.

    ISSUE #{issue_number}:
    Title: {issue_title}
    Description: {issue_body[:1000]}{"..." if len(issue_body) > 1000 else ""}

    PULL REQUEST #{pr_number}:
    Title: {pr_title}
    Description: {pr_body[:1000]}{"..." if len(pr_body) > 1000 else ""}

    COMMITS & CHANGES:
    """
    
    for i, commit in enumerate(commit_diffs):
        prompt += f"\nCOMMIT {i+1}:\n"
        prompt += f"Message: {commit['message']}\n"
        prompt += f"Changes:\n{commit['diff'][:1500]}{'...' if len(commit['diff']) > 1500 else ''}\n"
    
    prompt += """
    Based on the issue description and the changes in the PR, please analyze:
    1. Does this PR appear to fix the issue described?
    2. What specific changes in the PR address the issue, if any?
    3. How confident are you that this PR resolves the issue (High, Medium, Low)?
    
    Respond in JSON format with these keys:
    {
        "likely_fixes_issue": true/false,
        "confidence": "High/Medium/Low",
        "reasoning": "Your detailed explanation here",
        "relevant_changes": "Description of specific changes that address the issue"
    }
    """
    
    try:
        response = rate_limited_api_call(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = response.choices[0].message.content
        
        # Try to parse the JSON response
        import json
        try:
            analysis_json = json.loads(analysis)
            return analysis_json
        except json.JSONDecodeError:
            print(f"Error parsing AI response as JSON for issue #{issue_number} and PR #{pr_number}")
            # Return a structured response even if JSON parsing fails
            return {
                "likely_fixes_issue": False,
                "confidence": "Low",
                "reasoning": "Failed to parse AI response",
                "relevant_changes": "",
                "raw_response": analysis
            }
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Wait a bit before the next API call
        time.sleep(2)
        return {
            "likely_fixes_issue": False,
            "confidence": "Low",
            "reasoning": f"API error: {str(e)}",
            "relevant_changes": ""
        }


def find_zombie_issues():
    """Find zombie issues - open issues that may have been fixed by merged PRs"""
    open_issues = get_open_issues()
    merged_prs = get_merged_pull_requests(limit=30)  # Limit to 30 most recent PRs
    
    zombie_candidates = []
    
    # For each open issue, check against merged PRs
    for issue in open_issues[:20]:  # Limit to 20 issues to avoid excessive API calls
        issue_number = issue["number"]
        print(f"Analyzing issue #{issue_number}: {issue['title']}")
        
        for pr in merged_prs:
            pr_number = pr["number"]
            
            # Skip obvious non-matches to save API calls
            # If PR came before issue was created, it can't fix it
            if datetime.strptime(pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ") < datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ"):
                continue
                
            print(f"  Checking against PR #{pr_number}")
            
            # Use AI to analyze if this PR might fix the issue
            analysis = ai_analyze_potential_fixes(issue, pr)
            
            # If the AI thinks this PR likely fixes the issue, add it to our candidates
            if analysis.get("likely_fixes_issue") and analysis.get("confidence") in ["High", "Medium"]:
                zombie_candidates.append({
                    "issue_number": issue_number,
                    "issue_title": issue["title"],
                    "issue_url": issue["html_url"],
                    "pr_number": pr_number,
                    "pr_title": pr["title"],
                    "pr_url": pr["html_url"],
                    "pr_merged_at": pr["merged_at"],
                    "confidence": analysis.get("confidence"),
                    "reasoning": analysis.get("reasoning"),
                    "relevant_changes": analysis.get("relevant_changes")
                })
                print(f"    ✓ Potential fix found with {analysis.get('confidence')} confidence")
                break  # Move to next issue once we find a likely fix
            
            # Avoid hitting rate limits
            time.sleep(1)
    
    return zombie_candidates

def main():
    print("Finding zombie issues (open issues that are likely fixed)...")
    zombie_issues = find_zombie_issues()
    
    print(f"\nFound {len(zombie_issues)} potential zombie issues")
    
    # Convert to DataFrame and save to CSV
    if zombie_issues:
        df = pd.DataFrame(zombie_issues)
        df.to_csv("zombie_issues.csv", index=False)
        print("Results saved to zombie_issues.csv")
        
        # Print a summary
        print("\nSummary of zombie issues:")
        for i, issue in enumerate(zombie_issues):
            print(f"{i+1}. Issue #{issue['issue_number']}: {issue['issue_title']}")
            print(f"   Likely fixed by PR #{issue['pr_number']}: {issue['pr_title']}")
            print(f"   Confidence: {issue['confidence']}")
            print()
    else:
        print("No zombie issues found.")

if __name__ == "__main__":
    main()