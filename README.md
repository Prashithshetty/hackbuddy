# AI-Powered Hackathon Problem Generator

An intelligent system that generates contextual, well-defined hackathon problem statements using the LLaMA model. The system crawls relevant sources, analyzes themes, and generates implementable technical challenges.

## Features

- **Theme-based Generation**: Support for multiple domains including AI/ML, Climate Tech, Healthcare, etc.
- **Intelligent Analysis**: Uses LLaMA model to analyze and generate relevant problems
- **Multi-source Data**: Aggregates data from arXiv, GitHub, tech blogs, and other sources
- **Problem Validation**: Ensures problems are practical and implementable
- **Difficulty Estimation**: Auto-categorizes problems into Easy/Medium/Hard
- **Persistent Storage**: JSON-based problem database with deduplication

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download LLaMA model:
- Download DeepSeek-R1-8b.gguf from hugging face or ollama or lmstudio
- Place it in folder `ai/`
- Update path in line 574 in lama.py 

## Project Structure

```
api/
├── lama.py          # Main application logic
├── database.py      # Database handler for problem storage
├── scraper.py       # Multi-source problem scraper
├── requirements.txt # Project dependencies
└── problems_db.json # Problem database
```

## Usage

Run the main script:
```bash
python lama.py
```

Follow the interactive prompts to:
1. Select a theme
2. Wait for data collection and analysis
3. Review generated problem statement
4. Problems are automatically saved to database

## Supported Themes

- AI & ML
- Climate Tech
- Healthcare
- Innovation
- FinTech
- Logistics
- Sustainability

## Generated Problem Format

Each problem includes:
- Title
- Technical Challenge Description
- Context and Current Trends
- Impact Assessment
- Technical/Business Constraints
- Success Criteria
- Difficulty Rating

## Requirements

- Python 3.8+
- LLaMA model support
- Internet connection for data scraping
- Minimum 8GB RAM recommended

## Dependencies

- llama-cpp-python >= 0.2.0
- beautifulsoup4 >= 4.12.0
- aiohttp >= 3.9.0
- cachetools >= 5.3.0
- requests >= 2.31.0
- feedparser >= 6.0.10
- html2text >= 2020.1.16
- urllib3 >= 2.1.0

## Error Handling

- Logs are stored in `hackathon_generator.log`
- Failed scraping attempts are gracefully handled
- Fallback prompts when data collection fails

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Author

Prashith R Shetty

## Acknowledgments

- LLaMA model community
- arXiv API
- GitHub API
