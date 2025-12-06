# Multi-Agent Research Assistant

A multi-agent system for reading research papers, answering questions, and suggesting new research directions.

## Overview

This project implements a multi-agent chatbot system designed to assist researchers, students, and educators in analyzing academic literature. The system leverages Large Language Models (LLMs) through a coordinated multi-agent architecture with automatic intent classification.

### Key Features

- **Paper Reading & Summarization**: Automatically extract and summarize key findings from uploaded PDF research papers
- **Question Answering**: Ask questions about paper content and receive accurate answers with citations
- **Research Suggestions**: Get AI-generated research direction suggestions based on identified gaps and limitations
- **Automatic Intent Classification**: System automatically routes your query to the appropriate agent
- **Interactive Chat Interface**: User-friendly Streamlit-based web interface

## Getting Started

### Running the Application

**Start the Streamlit web interface:**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🔧 Configuration

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=your-cornell-api-key-here
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-cornell-api-key-here
```