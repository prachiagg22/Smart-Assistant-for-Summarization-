# Smart-Assistant-for-Summarization-


A **Smart Assistant** powered by Gen AI  to **automatically summarize research ** and extract key insights like abstract, keywords, bullet summaries, and Q&A. This tool helps researchers, students, and academicians save time by digesting long scientific texts quickly and effectively.

##  Features

- Upload PDF 
-  Extract abstract, keywords, and summaries
-  Generate bullet-point summaries
-  Ask questions from the paper and get AI-powered answers
-  Web-based user interface using Streamlit or Flask

##  Tech Stack

- Python 3.8+**
- PyMuPDF / PDFMiner / PyPDF2** – for PDF extraction
- spaCy / NLTK / Transformers** – for NLP processing
- HuggingFace Transformers (BART / T5)** – for summarization
- Sentence-BERT / OpenAI Embeddings** – for semantic Q&A
- Streamlit / Flask / Gradio** – for web interface

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-assistant-summarizer.git
   cd smart-assistant-summarizer

 Project Structure
   smart-assistant-summarizer
│
├── app.py                
├── summarizer.py          
├── extractor.py         
├── qa_module.py           
├── utils.py             
├── requirements.txt
└── README.md

