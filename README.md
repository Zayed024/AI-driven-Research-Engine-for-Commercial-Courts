# AI-Driven Research Engine for Commercial Courts

## Overview

This project is a prototype of an **AI-Driven Research Engine** designed to assist judges and judicial officers in India’s **commercial courts**. The goal is to ease the legal research process, leading to faster dispute resolution. The engine aggregates and processes legal data, generates customized research results, and uses **Natural Language Processing (NLP)** techniques to identify key legal principles, precedents, and case outcomes.

This project is built for Smart India Hackathon 2024, Problem Statement ID: 1701, focusing on leveraging **AI technology** to streamline legal research and support the Indian judiciary's efforts to expedite commercial dispute resolution.

## Features

- **PDF Document Processing**: Extracts and processes text from PDF legal documents.
- **NLP-based Analysis**: Uses Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and dependency parsing to analyze legal texts.
- **Text Segmentation**: Segments documents using **TextTiling** to create coherent text chunks.
- **Customizable Query System**: Allows users to ask legal questions, with responses generated based on the content of uploaded documents.
- **Legal Text Embedding**: Generates embeddings for legal texts using the **InLegalBERT** model.
- **Predictive Analytics** (Planned): The system will forecast case outcomes based on historical data and trends.
- **Data Localization** (Planned): Results tailored to the specific laws and rules of different Indian High Courts.
- **Multilingual Support** (Planned): Support for multiple Indian languages using NLP techniques.
- **Ethical AI**: Focuses on transparency, ensuring that the engine acts as a research facilitator, not a decision-maker.

## Problem Statement

The project addresses the problem of **faster resolution of commercial disputes** in India, in line with the **Commercial Courts Act, 2015**. The research engine will help streamline the legal research process for commercial courts by:

- Aggregating data sources like case laws, statutory provisions, and rules.
- Extracting relevant legal information and identifying key principles and precedents.
- Customizing results based on the specific needs of each commercial suit.
- Ensuring localization of results for different Indian High Courts.
- Supporting multiple languages and upholding ethical considerations in the use of AI.

## Technologies Used

- **Python**: Core programming language for the project.
- **Gradio**: Used for building the user-friendly web interface.
- **Cohere API**: For generating natural language responses based on legal queries.
- **spaCy**: For NLP tasks like NER, POS tagging, and dependency parsing.
- **transformers**: For BERT-based embeddings using the **InLegalBERT** model.
- **PDFMiner**: For extracting text from PDF documents.
- **TextTilingTokenizer**: For segmenting legal documents into coherent chunks.
- **Torch**: For handling transformer models and embeddings.
- **nltk**: For tokenization, POS tagging, and additional NLP tasks.

## How It Works

1. **Upload a PDF Document**: The user uploads a legal document in PDF format.
2. **Ask a Legal Question**: The user inputs a legal query based on the document.
3. **NLP and Text Processing**: The engine processes the document, extracts legal information, and segments the text for deeper analysis.
4. **BERT Embeddings**: Legal text is transformed into embeddings using the **InLegalBERT** model.
5. **Generate Response**: The system generates a response based on the document content and the user query, leveraging Cohere’s natural language generation.
6. **Display Results**: The results, including legal principles, precedents, and a custom answer, are displayed in a user-friendly format.

## Planned Enhancements

- **Predictive Analytics**: Implementing machine learning models to forecast case outcomes based on historical data.
- **Multilingual Support**: Adding support for multiple Indian languages to make the engine accessible to diverse users.
- **Data Localization**: Providing localized results based on the specific High Court’s laws and rules.
- **Advanced Customization**: Allowing users to specify the nature of the case (e.g., contract law, intellectual property) for more relevant results.
- **Ethical Transparency**: Ensuring explainability and bias mitigation in AI-based decision-making.

## Setup and Installation

### Prerequisites

- Python 3.8+
- **Cohere API Key**: Get your API key from [Cohere](https://docs.cohere.ai/docs/quickstart).
- **spaCy Model**: Download the spaCy English model (`en_core_web_sm`).
  
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/repo-name.git
    cd repo-name
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your **Cohere API Key** in your environment variables:
    ```bash
    export CO_API_KEY=your-cohere-api-key
    ```

4. Run the application:
    ```bash
    python app.py
    ```

5. The Gradio interface will launch. You can now upload PDF documents and input queries to interact with the research engine.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
