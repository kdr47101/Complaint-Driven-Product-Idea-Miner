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
import re  # <-- added

# Load environment variables
load_dotenv()

# Configuration
COMPLAINT_KEYWORDS = [
    "problem", "issue", "pain", "hate", "wish", "annoy", "bug", "broken", 
    "missing", "can't", "cannot", "frustrat", "terrible", "awful", "sucks"
]
# Ignore generic/broad subreddits that often pollute results
BAD_SUBS = {
    'todayilearned', 'til', 'askreddit', 'pics', 'funny', 'news', 'worldnews',
    'videos', 'gaming', 'aww', 'movies', 'television', 'music', 'mildlyinteresting',
    'technology', 'science', 'space', 'outoftheloop', 'emmawatson'
}
# NEW: NSFW keyword blacklist (name or description match => reject)
NSFW_KEYWORDS = {
    'nsfw','porn','fetish','sex','sexual','erotic','xxx','adult','nude','naked',
    'cum','fap','fuck','bdsm','horny','kink','escort','sexy','strip','boobs',
    'ass','cock','milf','hentai','gayporn','transporn','sissy','thong','deepthroat'
}

def _domain_tokens(domain: str) -> List[str]:
    """Tokenize domain/topic. Drop very short/generic tokens."""
    s = domain.lower()
    tokens = re.findall(r'[a-z0-9]+', s)
    stop = {'www','com','net','org','io','co','the','and','for','app','site'}
    return [t for t in tokens if len(t) >= 3 and t not in stop]

def _has_token(text: str, tokens: List[str]) -> bool:
    tl = text.lower()
    return any(t in tl for t in tokens)

def is_sfw_subreddit(subreddit) -> bool:
    """Return True if subreddit is safe-for-work and not quarantined, and passes keyword heuristic."""
    try:
        # Native flags
        if getattr(subreddit, 'over18', False):
            return False
        if getattr(subreddit, 'quarantine', False) or getattr(subreddit, 'quarantined', False):
            return False
        name_l = getattr(subreddit, 'display_name', '') .lower()
        desc_l = (getattr(subreddit, 'public_description', '') or '').lower()
        # Keyword heuristic (name or description containing NSFW keywords)
        for kw in NSFW_KEYWORDS:
            if kw in name_l or kw in desc_l:
                return False
        return True
    except Exception:
        return False

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
    tokens = _domain_tokens(domain)
    domain_lower = domain.lower()
    
    try:
        # Search in subreddit names and descriptions
        for subreddit in reddit.subreddits.search(domain, limit=limit * 5):
            name_l = subreddit.display_name.lower()
            if name_l in BAD_SUBS:
                continue
            if not is_sfw_subreddit(subreddit):
                continue
            desc = getattr(subreddit, 'public_description', '') or ''
            if tokens and not (_has_token(name_l, tokens) or _has_token(desc, tokens)):
                continue
            if subreddit.subreddit_type == 'public':
                subreddits.add(subreddit.display_name)
                if len(subreddits) >= limit:
                    break
        # Posts (year)
        if len(subreddits) < limit:
            try:
                for submission in reddit.subreddit("all").search(domain, limit=150, time_filter='year'):
                    if getattr(submission, 'over_18', False):
                        continue
                    if not is_sfw_subreddit(submission.subreddit):
                        continue
                    name_l = submission.subreddit.display_name.lower()
                    if name_l in BAD_SUBS:
                        continue
                    if any(kw in name_l for kw in NSFW_KEYWORDS):
                        continue
                    title_l = (submission.title or '').lower()
                    if tokens and not (_has_token(title_l, tokens) or _has_token(name_l, tokens)):
                        continue
                    if submission.subreddit.subreddit_type == 'public':
                        subreddits.add(submission.subreddit.display_name)
                        if len(subreddits) >= limit:
                            break
            except Exception as e:
                st.warning(f"Could not search posts: {e}")
        # Posts (month)
        if len(subreddits) < limit:
            try:
                for submission in reddit.subreddit("all").search(domain, limit=80, time_filter='month'):
                    if getattr(submission, 'over_18', False):
                        continue
                    if not is_sfw_subreddit(submission.subreddit):
                        continue
                    name_l = submission.subreddit.display_name.lower()
                    if name_l in BAD_SUBS:
                        continue
                    if any(kw in name_l for kw in NSFW_KEYWORDS):
                        continue
                    title_l = (submission.title or '').lower()
                    if tokens and not (_has_token(title_l, tokens) or _has_token(name_l, tokens)):
                        continue
                    if submission.subreddit.subreddit_type == 'public':
                        subreddits.add(submission.subreddit.display_name)
                        if len(subreddits) >= limit:
                            break
            except Exception as e:
                st.warning(f"Could not expand search: {e}")
    except Exception as e:
        st.error(f"Error searching subreddits: {e}")
        return []
    # Final SFW filtering pass (defensive)
    filtered = [s for s in subreddits if is_sfw_subreddit(reddit.subreddit(s))]
    result = filtered[:limit]
    if result:
        st.caption(f"Found {len(result)} relevant subreddits (public, SFW, token-matched)")
    return result

@st.cache_data(ttl=1800)
def fetch_posts(subreddit_name: str, limit: int = 25) -> List[Dict]:
    """Fetch recent posts from a subreddit"""
    reddit = init_reddit()
    posts = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        if not is_sfw_subreddit(subreddit):
            return posts
        for submission in subreddit.new(limit=limit):
            if getattr(submission, 'over_18', False):
                continue
            title_l = (submission.title or '').lower()
            if any(kw in title_l for kw in NSFW_KEYWORDS):
                continue
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'url': f"https://reddit.com{submission.permalink}",
                'score': submission.score,
                'created_utc': submission.created_utc,
                'subreddit': subreddit_name
            })
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
        if getattr(submission, 'over_18', False) or not is_sfw_subreddit(submission.subreddit):
            return comments
        title_l = (submission.title or '').lower()
        if any(kw in title_l for kw in NSFW_KEYWORDS):
            return comments
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:limit]:
            if hasattr(comment, 'body') and comment.body != '[deleted]':
                body_l = comment.body.lower()
                if any(kw in body_l for kw in NSFW_KEYWORDS):
                    continue
                comments.append({
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'post_id': post_id,
                    'subreddit': subreddit_name,
                    'post_url': f"https://reddit.com{submission.permalink}",
                    'post_title': submission.title or ""
                })
        time.sleep(0.3)
    except Exception as e:
        st.warning(f"Could not fetch comments for post {post_id}: {e}")
    return comments

def filter_complaints(comments: List[Dict], domain: str) -> List[Dict]:
    """Filter comments that contain complaint keywords and are on-topic"""
    tokens = _domain_tokens(domain)
    complaint_comments = []
    for c in comments:
        body_lower = c['body'].lower()
        post_title_lower = (c.get('post_title') or '').lower()
        # Require complaint signal AND domain token in body or post title
        if any(k in body_lower for k in COMPLAINT_KEYWORDS) and (_has_token(body_lower, tokens) or _has_token(post_title_lower, tokens)):
            complaint_comments.append(c)
    return complaint_comments

@st.cache_data(ttl=3600)
def extract_ideas(comments: List[Dict], domain: str) -> List[Dict]:
    """Extract product ideas from complaint comments using Gemini"""
    if not comments:
        return []
    
    model = init_openai()
    if not model:
        st.error("Gemini client not initialized")
        return []
    
    ideas = []
    max_comments = 30
    comments = comments[:max_comments]
    batch_size = 10
    batch_count = len(comments) // batch_size + (1 if len(comments) % batch_size else 0)
    
    st.info(f"Processing {len(comments)} comments in {batch_count} batches. This will take ~{batch_count * 10} seconds due to API rate limits.")
    st.info("Free tier limit: 10 requests/minute. Quota resets every minute.")
    
    tokens = _domain_tokens(domain)
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        # Prepare prompt with strict on-topic rules
        comments_text = ""
        for j, c in enumerate(batch):
            comments_text += f"Comment {j+1} (from r/{c['subreddit']}, {c['score']} upvotes):\n{c['body']}\n\n"
        
        prompt = f"""
        You are extracting product ideas for the domain/topic: "{domain}".
        ONLY include ideas that are directly related to this domain/topic.
        If an idea is not clearly related, do not include it.

        Return a JSON array; each object has:
        - idea: concise product idea (max 100 chars)
        - relevance_score: 0-100 (strong relation to "{domain}")
        - rationale: brief reason (max 150 chars)
        - source_comment: original comment text
        - subreddit, upvotes, post_url (if present)

        Rules:
        - Exclude off-topic ideas.
        - No speculation beyond comment context.
        - Return ONLY a JSON array (no extra text).

        Comments:
        {comments_text}
        """
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                st.text(f"Processing batch {batch_num}/{batch_count} (attempt {attempt + 1}/{max_retries})...")
                response = model.generate_content(prompt)
                content = response.text or ""
                
                if content:
                    st.caption(f"Response preview: {content[:200]}...")
                
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                try:
                    batch_ideas = json.loads(content)
                except json.JSONDecodeError as json_err:
                    st.error(f"JSON parsing failed. Raw response: {content[:500]}")
                    st.error(f"JSON error: {json_err}")
                    if '[' in content and ']' in content:
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
                
                if not isinstance(batch_ideas, list):
                    batch_ideas = [batch_ideas]
                
                for idea in batch_ideas:
                    required = ['idea', 'source_comment', 'relevance_score']
                    if not all(f in idea for f in required):
                        continue
                    relevance = idea.get('relevance_score', 0) or 0
                    idea_text = (idea.get('idea') or '') + ' ' + (idea.get('rationale') or '') + ' ' + (idea.get('source_comment') or '')
                    if tokens and not _has_token(idea_text, tokens):
                        continue
                    if relevance >= 60:
                        for c in batch:
                            if c['body'] == idea['source_comment']:
                                idea['subreddit'] = c['subreddit']
                                idea['upvotes'] = c['score']
                                idea['post_url'] = c['post_url']
                                break
                        idea.setdefault('subreddit', '')
                        idea.setdefault('upvotes', 0)
                        idea['relevance_score'] = int(relevance)
                        idea['combined_score'] = relevance
                        # quality_score removed
                        ideas.append(idea)
                
                st.text(f"Batch {batch_num} completed successfully! Extracted {len(batch_ideas)} ideas.")
                break
                
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower():
                    wait_time = 60
                    if 'retry in' in error_msg.lower():
                        try:
                            import re as _re
                            match = _re.search(r'retry in (\d+\.?\d*)', error_msg.lower())
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
        
        if i + batch_size < len(comments):
            wait_seconds = 10
            st.text(f"Waiting {wait_seconds} seconds before next batch (API rate limit)...")
            time.sleep(wait_seconds)
    
    if ideas:
        ideas.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        st.info(f"Filtered to {len(ideas)} relevant ideas (relevance ≥60)")
        return ideas[:10]
    else:
        st.warning("No relevant ideas could be extracted. Try a different domain or adjust search parameters.")
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
    # Ensure required columns exist (avoid KeyError)
    required_cols = ['subreddit','source_comment','upvotes','idea','relevance_score']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col in ['upvotes','relevance_score'] else ''
    display_columns = required_cols
    df = df[display_columns]
    df.columns = ['Subreddit', 'Original Comment', 'Upvotes', 'Product Idea', 'Relevance']
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top Ideas Found", len(df))
    with col2:
        st.metric("Unique Subreddits", df['Subreddit'].nunique())
    
    # Add ranking column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    st.subheader("Top Product Ideas (Ranked by Relevance)")
    
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
        
        # Filter complaints (now domain-aware)
        complaint_comments = filter_complaints(all_comments, domain)
        st.info(f"Found {len(complaint_comments)} complaint-like comments")
        
        if complaint_comments:
            with st.spinner("Analyzing complaints and generating product ideas..."):
                ideas = extract_ideas(complaint_comments, domain)  # <-- pass domain
            
            st.success(f"Generated {len(ideas)} product ideas!")
            
            # Save to MongoDB if configured
            if init_mongodb():
                save_ideas_to_mongodb(ideas, domain)
            
            render_table(ideas)
        else:
            st.warning("No complaints found. Try a different domain or adjust the complaint keywords.")

if __name__ == "__main__":
    main()
