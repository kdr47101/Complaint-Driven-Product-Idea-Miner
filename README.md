# Complaint-Driven Product Idea Miner

A Streamlit-powered tool for discovering product opportunities by mining real user complaints from Reddit.

---

## Overview

**Complaint-Driven Product Idea Miner** helps you uncover actionable product ideas by analyzing Reddit discussions for complaints related to any topic, company, or domain. By leveraging Google Gemini's advanced language models and Reddit's vast community data, this app surfaces pain points and unmet needs—turning complaints into innovation.

---

## How It Works

1. **User Input**:  
   Enter a keyword, company name, or domain (e.g., `spotify`, `notion.so`, `ballet shoes`).

2. **Subreddit Discovery**:  
   Google Gemini is used to find subreddits that are relevant to your keyword.

3. **Complaint Mining**:  
   The app fetches top posts from these subreddits and scans their comments for complaint-like language (e.g., "problem", "issue", "hate", "broken", etc.).

4. **AI-Powered Idea Extraction**:  
   Google Gemini analyzes these complaint comments and extracts a subset that represent potential product ideas, assigning each a subjective relevance score (0-100) indicating how closely the idea relates to your original keyword.

5. **Results Table**:  
   The top ideas are displayed in a table, including:
   - Subreddit
   - Original Comment
   - Upvotes
   - Product Idea (concise summary)
   - Relevance Score (to your keyword)

6. **Export**:  
   Download the results as a CSV for further analysis.

---

## Features

- **No Coding Required**: Simple Streamlit web interface.
- **Real User Pain Points**: Surfaces authentic complaints from Reddit communities.
- **AI Filtering**: Uses Google Gemini to extract and rank only the most relevant product ideas.
- **Customizable Search**: Adjust number of subreddits and posts per subreddit.
- **Safe-for-Work**: Filters out NSFW and off-topic subreddits and comments.
- **Exportable Results**: Download your findings as a CSV file.

---

## Requirements

- Python 3.8+
- [Reddit API credentials](https://www.reddit.com/prefs/apps) (Client ID, Client Secret, User Agent)
- [Google Gemini API key](https://makersuite.google.com/app/apikey)
- [MongoDB Atlas connection URI](https://www.mongodb.com/cloud/atlas) (for saving results/history)

**You must provide your own API keys and credentials for this app to function.**

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Complaint-Driven-Product-Idea-Miner
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Credentials

You can use either a `.env` file or Streamlit's `.streamlit/secrets.toml`:

#### Option A: `.env` file

Copy the example and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your credentials
```

#### Option B: Streamlit Secrets (Recommended for deployment)

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your credentials
```

**Required fields:**
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `GEMINI_API_KEY`
- `MONGODB_URI`

---

## Usage

1. **Start the app:**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to [http://localhost:8501](http://localhost:8501).

3. **Enter a keyword or domain** (e.g., `spotify`, `notion.so`, `ballet shoes`).

4. **Adjust search parameters** (number of subreddits, posts per subreddit) as desired.

5. **Click "Find Product Ideas"** and wait for the results.

6. **Review the table** of product ideas, each with subreddit, comment, upvotes, idea, and relevance score.

7. **Download as CSV** if needed.

---

## Data Flow & Filtering

- **Subreddit Selection**:  
  Only public, safe-for-work (SFW), and topically relevant subreddits are considered. NSFW, quarantined, and generic subreddits are excluded.

- **Comment Filtering**:  
  Only comments containing complaint-like keywords are analyzed.

- **AI Extraction**:  
  Google Gemini is prompted to extract only ideas directly relevant to your keyword, and to assign a subjective relevance score (0-100).

- **Result Ranking**:  
  Ideas are ranked by relevance score. Only the top ideas are shown.

---

## API Rate Limits & Ethics

- **Reddit**: The app respects Reddit's API rate limits and uses a descriptive User-Agent.
- **Google Gemini**: Free tier limits apply (see [Gemini API docs](https://ai.google.dev/)).
- **MongoDB**: Used for optional result persistence/history.

**Please use responsibly and respect all API terms of service.**

---

## FAQ

**Q: Why do I need my own API keys?**  
A: Reddit and Google Gemini require authentication for API access. This ensures your usage is private and within their terms.

**Q: Is my data stored?**  
A: Results can be saved to your own MongoDB Atlas database if you provide a URI. Otherwise, data is only processed in memory.

**Q: Can I use this commercially?**  
A: This tool is intended for research, educational, and personal use. Validate all ideas independently before pursuing commercially.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This tool uses AI to generate product ideas from public Reddit comments. All ideas are suggestions only and should be validated through proper research. Always respect user privacy and platform terms of service.
