<h1 align="center">
  <img src="img/header_logo.png?{{current_time}}">
</h1>

<p align="center">
    </a>
    <a href="https://github.com/ChristinaSchlager/flashcard_generator">
    <img alt="Github Repo" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
    <a href="https://www.python.org/">
    <img alt="Made with Python 3.11.9" src="https://img.shields.io/badge/Made%20with-Python%203.11.9-1f425f.svg?color=265175">
    </a>
    <a href="https://github.com/explodinggradients/ragas/">
    <img alt="Evaluated by Ragas" src="https://img.shields.io/badge/Evaluated%20by%20-Ragas-cccccc.svg?color=fecb4c">
    </a>
    <a href="https://github.com/confident-ai/deepeval">
    <img alt="Evaluated by DeepEval" src="https://img.shields.io/badge/Evaluated%20by%20-DeepEval-cccccc.svg?color=7C3AED">
    </a>
    <a href="https://github.com/deepset-ai/haystack">
    <img alt="Evaluated by Haystack" src="https://img.shields.io/badge/Evaluated%20by%20-Haystack-cccccc.svg?color=1BB6A6">
    </a>
    <a href="https://github.com/nomic-ai/nomic">
    <img alt="Nomic Embeddings" src="https://img.shields.io/badge/Nomic%20Embeddings%20-Utilized-cccccc.svg">
    </a>
    <a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1">
    <img alt="Llama 3.1 Turbo" src="https://img.shields.io/badge/LLama%203.1%20Turbo%20-Utilized-cccccc.svg">
    </a>
    <a href="https://github.com/sphinx-doc/sphinx">
    <img alt="Documentation with Sphinx" src="https://img.shields.io/badge/Documentation%20with-Sphinx-cccccc.svg?color=0A507A">
    </a>
</p>

## Table of Contents
- [Overview](#overview)
- [RAG Pipeline](#ragpipeline)
- [Prerequisites](#prerequisites)
- [API Integration Details](#api-integration-details)
- [Installation](#installation)
- [Documentation](#documentation)
- [Citation](#citation)

## Overview

This repository contains the source code for the master's thesis "AI Meets Classroom: Optimizing Transformer-Based Language Models for Education". The project evaluates Retrieval Augmented Generation (RAG) pipelines, emphasizing the integration of FATE (Fairness, Accountability, Transparency, and Ethics) principles. A comprehensive suite of evaluation metrics, including those from Ragas, DeepEval, and Haystack, to ensure a robust analysis of the performance of RAG models in educational settings is used.

## RAG Pipeline

<img src="img/RAG_Flashcard_Generator.png?{{current_time}}" width="500">

| Pipelines           | Description                                             |
|-------------------|---------------------------------------------------------|
| Flashcard Generator | An open-source pipeline designed by valuing the principles of the FATE framework. |
| Baseline     | Utilizes OpenAI's models to serve as a comparative baseline in the evaluations process.             |

## Prerequisites

Before running the code, please ensure the following requirements are met:
- Python 3.11.9 installed
- API keys for Together.ai, OpenAI and Nomic Atlas set as environment variables
- All dependencies installed from `requirements.txt`

## API Integration Details

This project integrates several APIs to enhance functionality and achieve comprehensive evaluation metrics. Below is a detailed description of each API used.

### Together AI
- **Purpose**: Facilitates collaborative AI-driven applications and integrations, providing tools for real-time model training and inference across various frameworks while supporting only open-source models.
- **Setup**: Register for an account on Together AI's platform and configure your project with an API key.
- **Documentation**: [Together AI API documentation](https://docs.together.ai/docs/introduction)

### OpenAI
- **Purpose**: Supports advanced text generation and document retrieval functionalities essential for the RAG baseline pipeline for comparison.
- **Setup**: Obtain an API key from OpenAI and set it as an environment variable as described in the Installation section.
- **Documentation**: [OpenAI API documentation](https://platform.openai.com/docs/api-reference/introduction)

### Nomic Atlas
- **Purpose**: Provides open-source language model embeddings and clustering algorithms to enhance semantic search and data organization capabilities.
- **Setup**: Install the Nomic Atlas Python package, configure your API key, and follow the initialization guide to start embedding your data.
- **Documentation**: [Nomic Atlas API documentation](https://docs.nomic.ai/reference/python-api/embeddings)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/ChristinaSchlager/flashcard_generator.git
cd flashcard_generator
pip install -r requirements.txt
```
## Documentation

The documentation for this project is available in PDF format. Please find the details below:

- [**PDF Documentation**](./documentation/_build/latex/flashcard_generator_documentation.pdf)

## Abstract

In the modern educational landscape, digitization has led to a significant increase in textual data from multiple sources. While this increase in data offers a wealth of knowledge, it also presents substantial challenges for educators and students. Moreover, schools have transformed into dynamic, interactive learning environments where the use of Large Language Models (LLMs) has become indispensable, particularly since the release of ChatGPT in 2022. However, the introduction of regulatory frameworks such as the EU AI Act highlights the importance of both functional and non-functional requirements for developing fair, transparent, responsible and ethical AI. Therefore, the implementation of the FATE (Fairness, Accountability, Transparency, and Ethics) framework in the development of a Retrieval Augmented Generation (RAG) model is demonstrated through its application to a Flashcard Generator designed to create educational flashcards. It is shown that the integration of ethical principles not only aligns with, but also enhances the performance of the model. The Flashcard Generator, developed using publicly available data and open-source models, leverages the RAG model to generate question-answer pairs for flashcards. It is evaluated against human-generated ground truth answers and a benchmark RAG model built with GPT-4 Turbo employing evaluation metrics from Ragas, DeepEval, and Haystack. This proof-of-concept serves as a template for future advancements, emphasizing fairness, transparency, and ethics, while maintaining high performance.

## Citation

If you use this project provided within in your research, please cite it as follows:

Schlager, C., "AI Meets Classroom: Optimizing Transformer-Based Language Models for Education." Master's Thesis, Data Science \& Intelligent Analytics, Kufstein University of Applied Sciences, Austria, 2024.

