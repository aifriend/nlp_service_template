# NLP Service Template

Template for deploying machine learning and NLP services as REST APIs, with Docker and AWS Lambda (Serverless) support.

## Overview

A reusable project template for building and deploying ML/NLP microservices. Includes Flask API scaffolding, Docker containerization, and serverless deployment configuration.

## Project Structure

```
├── classifier/         # Classification models
├── controller/         # Request handlers
├── model/              # Data models
├── nlp/                # NLP processing modules
├── services/           # Business logic services
├── tests/              # Test suite
├── api.py              # Flask API entry point
├── GbcMachineLearningService.py  # ML service orchestrator
├── commonsLib.py       # Shared utilities
├── Dockerfile          # Docker configuration
├── serverless.yml      # AWS Lambda deployment
├── requirements.txt    # Python dependencies
└── wsgi.py             # WSGI server config
```

## Tech Stack

- **Language:** Python 3
- **API Framework:** Flask + Gunicorn
- **OCR:** Tesseract
- **Deployment:** Docker, AWS Lambda (Serverless Framework)
- **Port:** 5000

## Getting Started

### Local Development
```bash
pip install -r requirements.txt
python api.py
```

### Docker
```bash
docker build -t nlp-service .
docker run -p 5000:5000 nlp-service
```

### AWS Lambda
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
sls deploy
```

## Running Tests
```bash
cd tests
pytest
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Jose** — [@aifriend](https://github.com/aifriend)
