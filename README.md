# 💬 Product Recommender Chatbot

A smart product recommendation chatbot built with Streamlit and Cohere AI that helps users find the perfect laptop based on their requirements.

## Features

- 🤖 AI-powered product recommendations using Cohere's LLM
- 🔍 Semantic search with embeddings for accurate product matching
- 💬 Interactive chat interface
- 📊 Context-aware responses based on conversation history

## Prerequisites

- Python 3.8 or higher
- Cohere API key ([Get one here](https://dashboard.cohere.ai/))

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "Product Recommender Chatbot"
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**

   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   - Mac/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   DATABASE_FILE=data/products.json
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Project Structure

```
Product Recommender Chatbot/
├── app.py                  # Main Streamlit application
├── data/
│   └── products.json       # Product catalog
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Deployment

This app can be deployed on:
- **Streamlit Cloud** (Recommended)
- **Heroku**
- **AWS/GCP/Azure**
- **Docker containers**

### Deploying to Streamlit Cloud

1. Push your code to GitHub (without venv folder)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your `COHERE_API_KEY` in the Secrets section
5. Deploy!

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `COHERE_API_KEY` | Your Cohere API key | `abc123...` |
| `DATABASE_FILE` | Path to products JSON | `data/products.json` |

## Technologies Used

- **Streamlit** - Web framework
- **Cohere AI** - LLM and embeddings
- **NumPy** - Vector operations
- **Python-dotenv** - Environment management

## License

This project is open source and available under the MIT License.

