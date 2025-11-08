import streamlit as st
import praw
import google.generativeai as genai
import pandas as pd
import os
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import certifi

# Load environment variables
load_dotenv()

# Configuration
COMPLAINT_KEYWORDS = [
    "problem", "issue", "pain", "hate", "wish", "annoy", "bug", "broken", 
    "missing", "can't", "cannot", "frustrat", "terrible", "awful", "sucks"
]

def get_credential(key: str) -> str:
    """Get credential from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (preferred for deployment)
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    # Fallback to environment variables
    return os.getenv(key, "")

# Initialize APIs
def init_reddit():
    """Initialize Reddit API client"""
    return praw.Reddit(
        client_id=get_credential("REDDIT_CLIENT_ID"),
        client_secret=get_credential("REDDIT_CLIENT_SECRET"),
        user_agent=get_credential("REDDIT_USER_AGENT") or "ComplaintMiner/1.0"
    )

def init_openai():
    """Initialize Gemini API client"""
    api_key = get_credential("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        # Updated to Gemini 2.0 Flash (latest available model)
        return genai.GenerativeModel('gemini-2.0-flash-exp')
    return None

def init_mongodb():
    """Initialize MongoDB client"""
    mongo_uri = get_credential("MONGODB_URI")
    if mongo_uri:
        try:
            client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=5000  # Faster timeout for initial connection
            )
            # Test connection
            client.admin.command('ping')
            return client
        except Exception as e:
            # Silently fail - don't show warning in sidebar
            return None
    return None

@st.cache_data(ttl=3600)
def search_subreddits(domain: str, limit: int = 10) -> List[str]:
    """Search for subreddits related to a domain or topic"""
    reddit = init_reddit()
    subreddits = set()
    
    try:
        # First, try to find subreddits by searching for the topic/domain name
        # Search in subreddit names and descriptions
        for subreddit in reddit.subreddits.search(domain, limit=limit * 3):
            # Filter out NSFW, private, or banned subreddits
            if not subreddit.over18 and subreddit.subreddit_type == 'public':
                subreddits.add(subreddit.display_name)
                if len(subreddits) >= limit:
                    break
        
        # If we didn't find enough subreddits, search for posts about the topic
        # and collect unique subreddits
        if len(subreddits) < limit:
            try:
                search_query = domain
                for submission in reddit.subreddit("all").search(search_query, limit=100, time_filter='year'):
                    if submission.subreddit.subreddit_type == 'public' and not submission.subreddit.over18:
                        # Check if subreddit name or description is related to the domain
                        sub_name_lower = submission.subreddit.display_name.lower()
                        domain_lower = domain.lower()
                        
                        # Add subreddit if its name contains the search term or is clearly related
                        if (domain_lower in sub_name_lower or 
                            any(word in sub_name_lower for word in domain_lower.split())):
                            subreddits.add(submission.subreddit.display_name)
                            if len(subreddits) >= limit:
                                break
            except Exception as e:
                st.warning(f"Could not search posts: {e}")
        
        # If still not enough, get most relevant subreddits from post search
        if len(subreddits) < limit:
            try:
                for submission in reddit.subreddit("all").search(domain, limit=50, time_filter='month'):
                    if submission.subreddit.subreddit_type == 'public' and not submission.subreddit.over18:
                        subreddits.add(submission.subreddit.display_name)
                        if len(subreddits) >= limit:
                            break
            except Exception as e:
                st.warning(f"Could not expand search: {e}")
                
    except Exception as e:
        st.error(f"Error searching subreddits: {e}")
        return []
    
    result = list(subreddits)[:limit]
    
    # Show a note if we're filtering results
    if result:
        st.caption(f"Found {len(result)} relevant subreddits (filtered for public, SFW communities)")
    
    return result

@st.cache_data(ttl=1800)
def fetch_posts(subreddit_name: str, limit: int = 25) -> List[Dict]:
    """Fetch recent posts from a subreddit"""
    reddit = init_reddit()
    posts = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.new(limit=limit):
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'url': f"https://reddit.com{submission.permalink}",
                'score': submission.score,
                'created_utc': submission.created_utc,
                'subreddit': subreddit_name
            })
            
        # Respect rate limits - small delay between requests
        time.sleep(0.5)
        
    except Exception as e:
        st.warning(f"Could not fetch posts from r/{subreddit_name}: {e}")
    
    return posts

@st.cache_data(ttl=1800)
def fetch_comments(post_id: str, subreddit_name: str, limit: int = 50) -> List[Dict]:
    """Fetch top-level comments from a post"""
    reddit = init_reddit()
    comments = []
    
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Remove "more comments"
        
        for comment in submission.comments[:limit]:
            if hasattr(comment, 'body') and comment.body != '[deleted]':
                comments.append({
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'post_id': post_id,
                    'subreddit': subreddit_name,
                    'post_url': f"https://reddit.com{submission.permalink}"
                })
                
        # Respect rate limits
        time.sleep(0.3)
        
    except Exception as e:
        st.warning(f"Could not fetch comments for post {post_id}: {e}")
    
    return comments

def filter_complaints(comments: List[Dict]) -> List[Dict]:
    """Filter comments that contain complaint keywords"""
    complaint_comments = []
    
    for comment in comments:
        body_lower = comment['body'].lower()
        if any(keyword in body_lower for keyword in COMPLAINT_KEYWORDS):
            complaint_comments.append(comment)
    
    return complaint_comments

@st.cache_data(ttl=3600)
def extract_ideas(comments: List[Dict]) -> List[Dict]:
    """Extract product ideas from complaint comments using Gemini"""
    if not comments:
        return []
    
    model = init_openai()
    if not model:
        st.error("Gemini client not initialized")
        return []
    
    ideas = []
    
    # Limit comments to avoid excessive API calls
    # Free tier: 10 requests per minute (resets every minute)
    max_comments = 30  # Process max 30 comments (3 batches of 10) to stay well under limit
    comments = comments[:max_comments]
    
    # Process comments in batches of 10
    batch_size = 10
    batch_count = len(comments) // batch_size + (1 if len(comments) % batch_size else 0)
    
    st.info(f"Processing {len(comments)} comments in {batch_count} batches. This will take ~{batch_count * 10} seconds due to API rate limits.")
    st.info("Free tier limit: 10 requests/minute. Quota resets every minute.")
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        # Prepare prompt
        comments_text = ""
        for j, comment in enumerate(batch):
            comments_text += f"Comment {j+1} (from r/{comment['subreddit']}, {comment['score']} upvotes):\n{comment['body']}\n\n"
        
        prompt = f"""
        Analyze these Reddit comments and extract product ideas that could solve the complaints mentioned.
        For each complaint that suggests a clear product opportunity, return a JSON object with these fields:
        - idea: A concise product idea (max 100 chars)
        - quality_score: Your assessment of product quality from 0-100 (higher = better quality)
        - rationale: Why this addresses the complaint and why it's high quality (max 150 chars)  
        - source_comment: The original comment text
        - subreddit: The subreddit name
        - upvotes: The comment's upvote score
        - post_url: The post URL
        - relevance_score: How relevant this idea is to the subreddit/complaint context (0-100)
        
        IMPORTANT CRITERIA:
        1. Strong commercial viability - Clear market demand and monetization potential
        2. Technically feasible - Can be built with current technology within reasonable budget
        3. Highly relevant - Must be directly related to the complaint and the subreddit's topic
        4. Specific and actionable - Not vague concepts but concrete product ideas
        
        Evaluate quality based on:
        - Market demand (evident from complaint frequency/intensity)
        - Technical feasibility (realistic to build)
        - Potential user base size
        - Competition gaps
        - Revenue potential
        - Relevance to the complaint context
        
        ONLY include ideas that score 60+ on BOTH quality_score AND relevance_score.
        Reject ideas that are:
        - Too broad or generic
        - Unrelated to the subreddit's topic
        - Technically impossible or extremely difficult
        - Already well-served by existing products
        
        Return ONLY a valid JSON array of objects. Do not include any explanation or markdown formatting.
        Rank ideas by combined quality and relevance internally.
        
        Comments:
        {comments_text}
        """
        
        # Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                st.text(f"Processing batch {batch_num}/{batch_count} (attempt {attempt + 1}/{max_retries})...")
                response = model.generate_content(prompt)
                content = response.text
                
                # Debug: show first 200 chars of response
                if content:
                    st.caption(f"Response preview: {content[:200]}...")
                
                # Extract JSON from response (handle markdown code blocks)
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # Try to parse JSON
                try:
                    batch_ideas = json.loads(content)
                except json.JSONDecodeError as json_err:
                    st.error(f"JSON parsing failed. Raw response: {content[:500]}")
                    st.error(f"JSON error: {json_err}")
                    # Try to find if there's any JSON-like content
                    if '[' in content and ']' in content:
                        # Try to extract just the array part
                        start = content.find('[')
                        end = content.rfind(']') + 1
                        json_substring = content[start:end]
                        try:
                            batch_ideas = json.loads(json_substring)
                            st.info("Successfully extracted JSON from response")
                        except:
                            raise json_err
                    else:
                        raise json_err
                
                # Ensure we have a list
                if not isinstance(batch_ideas, list):
                    batch_ideas = [batch_ideas]
                
                # Ensure we have the required fields and add missing data
                for idea in batch_ideas:
                    # Check for required fields including relevance_score
                    required_fields = ['idea', 'source_comment', 'quality_score', 'relevance_score']
                    if all(field in idea for field in required_fields):
                        # Filter by minimum scores (changed to 60)
                        quality = idea.get('quality_score', 0)
                        relevance = idea.get('relevance_score', 0)
                        
                        if quality >= 60 and relevance >= 60:
                            # Find matching comment for additional data
                            for comment in batch:
                                if comment['body'] == idea['source_comment']:
                                    idea['subreddit'] = comment['subreddit']
                                    idea['upvotes'] = comment['score']
                                    idea['post_url'] = comment['post_url']
                                    break
                            # Calculate combined score for ranking
                            idea['combined_score'] = (quality * 0.6) + (relevance * 0.4)
                            ideas.append(idea)
                
                # Success - break retry loop
                st.text(f"Batch {batch_num} completed successfully! Extracted {len(batch_ideas)} ideas.")
                break
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if '429' in error_msg or 'quota' in error_msg.lower():
                    # Extract wait time from error message if available
                    wait_time = 60
                    if 'retry in' in error_msg.lower():
                        try:
                            # Extract seconds from error message
                            import re
                            match = re.search(r'retry in (\d+\.?\d*)', error_msg.lower())
                            if match:
                                wait_time = float(match.group(1)) + 2
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        st.warning(f"Rate limit hit. Waiting {wait_time:.0f} seconds for quota to reset...")
                        time.sleep(wait_time)
                    else:
                        st.error(f"Batch {batch_num} failed: Rate limit exceeded")
                        st.info("Your API quota will reset in ~1 minute. Please wait and try again.")
                        continue
                else:
                    if attempt < max_retries - 1:
                        st.warning(f"Batch {batch_num} error (will retry): {e}")
                        time.sleep(5)
                    else:
                        st.error(f"Error extracting ideas from batch {batch_num}: {e}")
                        break
        
        # Rate limiting between batches
        if i + batch_size < len(comments):
            wait_seconds = 10
            st.text(f"Waiting {wait_seconds} seconds before next batch (API rate limit)...")
            time.sleep(wait_seconds)
    
    # Sort by combined score (quality 60% + relevance 40%) descending
    if ideas:
        ideas.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        st.info(f"Filtered to {len(ideas)} high-quality, relevant ideas (quality ≥60, relevance ≥60)")
        return ideas[:10]
    else:
        st.warning("No high-quality ideas could be extracted. Try a different domain or adjust search parameters.")
        return []

def save_ideas_to_mongodb(ideas: List[Dict], domain: str):
    """Save top ideas to MongoDB"""
    client = init_mongodb()
    if not client:
        return False
    
    try:
        db = client['product_ideas']
        collection = db['ideas']
        
        # Prepare documents for each idea in the requested format
        documents = []
        for rank, idea in enumerate(ideas, 1):
            document = {
                'prompt': domain,
                'subreddit': idea.get('subreddit', ''),
                'comment': idea.get('source_comment', ''),
                'upvotes': idea.get('upvotes', 0),
                'product_idea': idea.get('idea', ''),
                'quality_score': idea.get('quality_score', 0),
                'relevance_score': idea.get('relevance_score', 0),
                'combined_score': idea.get('combined_score', 0),
                'rank': rank,
                'timestamp': datetime.utcnow()
            }
            documents.append(document)
        
        # Insert all documents
        result = collection.insert_many(documents)
        st.success(f"Saved {len(result.inserted_ids)} ideas to MongoDB")
        return True
        
    except Exception as e:
        st.error(f"Error saving to MongoDB: {e}")
        return False
    finally:
        if client:
            client.close()

def get_historical_runs(limit: int = 10) -> List[Dict]:
    """Get historical runs from MongoDB"""
    client = init_mongodb()
    if not client:
        return []
    
    try:
        db = client['product_ideas']
        collection = db['ideas']
        
        # Get unique domains (prompts) with their latest timestamp
        pipeline = [
            {
                '$group': {
                    '_id': '$prompt',
                    'latest_timestamp': {'$max': '$timestamp'},
                    'total_ideas': {'$sum': 1}
                }
            },
            {'$sort': {'latest_timestamp': -1}},
            {'$limit': limit}
        ]
        
        runs = list(collection.aggregate(pipeline))
        
        # Format for display
        formatted_runs = []
        for run in runs:
            formatted_runs.append({
                'domain': run['_id'],
                'timestamp': run['latest_timestamp'],
                'total_ideas': run['total_ideas']
            })
        
        return formatted_runs
        
    except Exception as e:
        # Silently fail - don't show errors in sidebar
        return []
    finally:
        if client:
            client.close()

def load_run_from_mongodb(domain: str):
    """Load a specific run from MongoDB by domain"""
    client = init_mongodb()
    if not client:
        return None
    
    try:
        db = client['product_ideas']
        collection = db['ideas']
        
        # Get the most recent ideas for this domain
        ideas = list(collection.find(
            {'prompt': domain}
        ).sort('timestamp', -1).limit(10))
        
        if ideas:
            # Convert to the format expected by render_table
            formatted_ideas = []
            for idea in ideas:
                formatted_ideas.append({
                    'subreddit': idea.get('subreddit', ''),
                    'source_comment': idea.get('comment', ''),
                    'upvotes': idea.get('upvotes', 0),
                    'idea': idea.get('product_idea', ''),
                    'quality_score': idea.get('quality_score', 0),
                    'relevance_score': idea.get('relevance_score', 0)
                })
            
            return {
                'domain': domain,
                'ideas': formatted_ideas,
                'timestamp': ideas[0].get('timestamp')
            }
        
        return None
        
    except Exception as e:
        st.error(f"Error loading run: {e}")
        return None
    finally:
        if client:
            client.close()

def render_table(ideas: List[Dict]):
    """Render the results table and provide CSV download"""
    if not ideas:
        st.warning("No product ideas found. Try a different domain or check your API credentials.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(ideas)
    
    # Reorder columns for the requested format
    display_columns = ['subreddit', 'source_comment', 'upvotes', 'idea', 'quality_score', 'relevance_score']
    df = df[display_columns]
    df.columns = ['Subreddit', 'Original Comment', 'Upvotes', 'Product Idea', 'Quality', 'Relevance']
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Ideas Found", len(df))
    with col2:
        st.metric("Unique Subreddits", df['Subreddit'].nunique())
    with col3:
        avg_quality = df['Quality'].mean()
        st.metric("Avg Quality", f"{avg_quality:.0f}/100")
    
    # Add ranking column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    st.subheader("Top Product Ideas (Ranked by Quality & Relevance)")
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                width="small"
            ),
            "Original Comment": st.column_config.TextColumn(
                "Original Comment",
                width="large"
            ),
            "Product Idea": st.column_config.TextColumn(
                "Product Idea",
                width="large"
            ),
            "Upvotes": st.column_config.NumberColumn(
                "Upvotes",
                width="small"
            ),
            "Quality": st.column_config.NumberColumn(
                "Quality",
                width="small",
                help="Product quality score (0-100)"
            ),
            "Relevance": st.column_config.NumberColumn(
                "Relevance",
                width="small",
                help="Relevance to domain/complaint (0-100)"
            )
        }
    )
    
    # CSV download
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Top Ideas as CSV",
        data=csv,
        file_name=f"top_product_ideas_{int(time.time())}.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(
        page_title="Complaint-Driven Product Idea Miner",
        page_icon="",
        layout="wide"
    )
    
    st.title("Complaint-Driven Product Idea Miner")
    st.markdown("Discover product opportunities by analyzing Reddit complaints about any domain or company.")
    
    # Sidebar with history
    with st.sidebar:
        st.header("Search History")
        
        # Get historical runs
        historical_runs = get_historical_runs(limit=20)
        
        if historical_runs:
            st.caption(f"Found {len(historical_runs)} previous searches")
            
            for run in historical_runs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Format timestamp
                    timestamp_str = run['timestamp'].strftime("%m/%d %H:%M") if run['timestamp'] else "Unknown"
                    
                    # Create clickable button for each historical run
                    if st.button(
                        f"{run['domain']}", 
                        key=f"hist_{run['domain']}_{timestamp_str}",
                        help=f"{run['total_ideas']} ideas • {timestamp_str}"
                    ):
                        # Load this run
                        loaded_run = load_run_from_mongodb(run['domain'])
                        if loaded_run:
                            st.session_state['loaded_ideas'] = loaded_run['ideas']
                            st.session_state['loaded_domain'] = loaded_run['domain']
                            st.rerun()
                
                with col2:
                    st.caption(f"{run['total_ideas']}")
            
            # Add a divider
            st.divider()
            
            # Clear history button
            if st.button("Clear History", help="Delete all historical data"):
                if st.button("Confirm Delete", key="confirm_delete"):
                    client = init_mongodb()
                    if client:
                        try:
                            db = client['product_ideas']
                            result = db['ideas'].delete_many({})
                            st.success(f"Deleted {result.deleted_count} records")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing history: {e}")
                        finally:
                            client.close()
        else:
            st.info("No search history yet. Run your first search!")
    
    # Debug: Show what credentials are being loaded
    with st.expander("Debug: Credential Status"):
        st.write("REDDIT_CLIENT_ID:", "Found" if get_credential("REDDIT_CLIENT_ID") else "Missing")
        st.write("REDDIT_CLIENT_SECRET:", "Found" if get_credential("REDDIT_CLIENT_SECRET") else "Missing")
        st.write("GEMINI_API_KEY:", "Found" if get_credential("GEMINI_API_KEY") else "Missing")
        st.write("MONGODB_URI:", "Found" if get_credential("MONGODB_URI") else "Missing")
        
        # Show first/last few characters of each credential (for debugging)
        mongodb_uri = get_credential("MONGODB_URI")
        if mongodb_uri and len(mongodb_uri) > 20:
            st.write(f"MongoDB URI preview: {mongodb_uri[:15]}...{mongodb_uri[-10:]}")
    
    # Check API credentials
    missing_creds = []
    if not get_credential("REDDIT_CLIENT_ID"):
        missing_creds.append("REDDIT_CLIENT_ID")
    if not get_credential("REDDIT_CLIENT_SECRET"):
        missing_creds.append("REDDIT_CLIENT_SECRET")
    if not get_credential("GEMINI_API_KEY"):
        missing_creds.append("GEMINI_API_KEY")
    
    if missing_creds:
        st.error(f"Missing API credentials: {', '.join(missing_creds)}")
        with st.expander("How to set up credentials"):
            st.markdown("""
            **Option 1: Streamlit Secrets (Recommended)**
            1. Create `.streamlit/secrets.toml` in your project directory
            2. Add your credentials in TOML format
            
            **Option 2: Environment Variables**
            1. Create `.env` file from `.env.example`
            2. Add your API credentials (Get free Gemini API key at https://makersuite.google.com/app/apikey)
            
            See the README.md for detailed setup instructions.
            """)
        return
    
    # Check if we're displaying a loaded historical run
    if 'loaded_ideas' in st.session_state:
        st.info(f"Viewing historical run for: {st.session_state.get('loaded_domain', 'Unknown')}")
        render_table(st.session_state['loaded_ideas'])
        if st.button("Clear and Start New Search"):
            del st.session_state['loaded_ideas']
            del st.session_state['loaded_domain']
            st.rerun()
        return
    
    # Input form
    with st.form("domain_form"):
        domain = st.text_input(
            "Enter a domain or company name:",
            placeholder="e.g., spotify.com, notion.so, github.com",
            help="Enter a domain to find related subreddits and complaints"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_subreddits = st.slider("Max subreddits to search", 3, 15, 10)
        with col2:
            posts_per_sub = st.slider("Posts per subreddit", 10, 50, 25)
        
        submitted = st.form_submit_button("Find Product Ideas")
    
    if submitted and domain:
        # Clear any loaded historical data when starting a new search
        if 'loaded_ideas' in st.session_state:
            del st.session_state['loaded_ideas']
        if 'loaded_domain' in st.session_state:
            del st.session_state['loaded_domain']
        
        with st.spinner("Searching for related subreddits..."):
            subreddits = search_subreddits(domain, max_subreddits)
        
        if not subreddits:
            st.warning(f"No subreddits found for domain: {domain}")
            return
        
        st.success(f"Found {len(subreddits)} related subreddits: {', '.join(subreddits)}")
        
        # Fetch posts and comments
        all_comments = []
        progress_bar = st.progress(0)
        
        for i, subreddit in enumerate(subreddits):
            st.text(f"Processing r/{subreddit}...")
            
            posts = fetch_posts(subreddit, posts_per_sub)
            for post in posts:
                comments = fetch_comments(post['id'], subreddit)
                all_comments.extend(comments)
            
            progress_bar.progress((i + 1) / len(subreddits))
        
        st.success(f"Collected {len(all_comments)} total comments")
        
        # Filter complaints
        complaint_comments = filter_complaints(all_comments)
        st.info(f"Found {len(complaint_comments)} complaint-like comments")
        
        if complaint_comments:
            with st.spinner("Analyzing complaints and generating product ideas..."):
                ideas = extract_ideas(complaint_comments)
            
            st.success(f"Generated {len(ideas)} product ideas!")
            
            # Save to MongoDB if configured
            if init_mongodb():
                save_ideas_to_mongodb(ideas, domain)
            
            render_table(ideas)
        else:
            st.warning("No complaints found. Try a different domain or adjust the complaint keywords.")

if __name__ == "__main__":
    main()
