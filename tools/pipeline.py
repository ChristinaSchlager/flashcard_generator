# requirements
import os
import json
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional

from together import Together
import together

from datasets import load_dataset, Dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from langchain_nomic import NomicEmbeddings

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Ragas Metrics (GPT-4)
from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import ChatOpenAI
from datasets import Features, Sequence, Value
from ragas.metrics import faithfulness, answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall, context_entity_recall
from ragas import evaluate as ragas_evaluate

# Deepval Metrics (GPT-4)
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric, BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

# Haystack Metrics (GPT-3.5 Turbo)
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator, SASEvaluator
from typing import List

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration
WIKIPEDIA_LANGUAGE = "simple"
WIKIPEDIA_DATE = "20220301"
CHROMA_OPENAI_PATH = "./chroma_db_openai"
CHROMA_NOMIC_PATH = "./chroma_db_nomic"
OPENAI_COLLECTION = "openai_rag_chroma_wikipedia"
NOMIC_COLLECTION = "opensource_rag_wikipedia"
TEXT_SPLITTER_CHUNK_SIZE = 5000
TEXT_SPLITTER_CHUNK_OVERLAP = 100
MIN_VALID_LENGTH = 200
VECTORSTORE_THRESHOLD = 0.5
MAX_SECTION_LENGTH = 150
MIN_SECTION_LENGTH = 50
CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

# ============================================================================
# Data Preparation
# ============================================================================
def load_wikipedia_dataset(language: str = WIKIPEDIA_LANGUAGE, date: str = WIKIPEDIA_DATE) -> Dataset:
    """
    Load the Wikipedia dataset from the Hugging Face Hub. This function loads the dataset
    specified by language and date.

    Hugging Face Dataset URL: https://huggingface.co/datasets/wikipedia

    :param language: Language version of Wikipedia to load. Defaults to 'simple' for Simple English.
    :param date: The snapshot date of the Wikipedia dataset to load. Format: YYYYMMDD. Defaults to "20220301".

    :return: A dataset object that allows accessing the data as required.
    """
    logger.info(f"Loading Wikipedia dataset: language={language}, date={date}")
    data = load_dataset("wikipedia", language=language, date=date, streaming=True, trust_remote_code=True)
    return data

def process_documents(
    data: Dict[str, Any],
    chunk_size: int = TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap: int = TEXT_SPLITTER_CHUNK_OVERLAP,
    min_valid_length: int = MIN_VALID_LENGTH,
) -> Tuple[List[Any], List[int], int, int]:
    """
    Process a dataset of documents by splitting each document into chunks, filtering these chunks based
    on a minimum valid length, and annotating them with metadata.

    This function uses a RecursiveCharacterTextSplitter to break down documents into manageable chunks. 
    Chunks that meet the minimum length requirement are kept and annotated with metadata from the original
    document. The function then returns a list of valid chunks and statistics about the processing.

    :param data: A dictionary containing a 'train' key with a list of document dictionaries. Each document
                    should have 'text', 'id', 'url', and 'title' keys.
    :param chunk_size: The size of each chunk in characters.
    :param chunk_overlap: The number of characters to overlap between chunks.
    :param min_valid_length: The minimum number of characters a chunk must have to be considered valid.

    :return: A tuple containing four elements:
             - A list of valid chunks, where each chunk is an object with 'page_content' and 'metadata', which contains ID, url, and title.
             - A list containing the count of chunks created from each document.
             - Total number of chunks created across all documents.
             - Total number of valid chunks that met the length requirement.
    """
    logger.info("Processing documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    texts = []
    chunk_counts = []
    total_chunks = 0
    total_valid_chunks = 0

    for i, example in enumerate(data['train']):
        chunks = text_splitter.create_documents([example["text"]])
        chunk_count = len(chunks)
        chunk_counts.append(chunk_count)
        total_chunks += chunk_count

        valid_chunks = 0
        for chunk in chunks:
            if len(chunk.page_content) >= min_valid_length:
                chunk.metadata = {'id': example["id"], 'url': example["url"], 'title': example["title"]}
                texts.append(chunk)
                valid_chunks += 1

        total_valid_chunks += valid_chunks

    logger.info(f"Total chunks: {total_chunks}, Valid chunks: {total_valid_chunks}")
    return texts, chunk_counts, total_chunks, total_valid_chunks

# ============================================================================
# Data Storage
# ============================================================================

def setup_vectorstore_openai(texts: List[Any], path: str = CHROMA_OPENAI_PATH, collection_name: str = OPENAI_COLLECTION) -> Tuple[Chroma, Any]:
    """
    Sets up a vector store using OpenAI embeddings (model "text-embedding-3-large") with the provided documents. The function initializes
    OpenAI embeddings, sets up a persistent client, checks for the existence of the specified collection
    (or creates it), and populates a vector store with the documents.

    :param texts: A list of documents (texts) to be stored in the vector store.
    :param path: The file path where the vector store data will be located. Defaults to "./chroma_db_openai".
    :param collection_name: The name of the collection to be used or created in the vector store. 
                            Defaults to "openai_rag_chroma_wikipedia".

    :return: A tuple containing a configured instance of the `Chroma` vector store and the `OpenAIEmbeddings` used.
    """
    logger.info(f"Setting up vector store")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536) # to stick to the 1536 dimensions
    persistent_client_openai = chromadb.PersistentClient(path=path)
    collection = persistent_client_openai.create_collection(collection_name, metadata={"hnsw:space": "cosine"}) #cosine similarity instead of default L2

    vectorstore = Chroma.from_documents(
        texts, embeddings, client=persistent_client_openai, collection_name=collection_name
    )
    logger.info("Vector store setup complete")
    return vectorstore, embeddings


def load_vectorstore_openai(path: str = CHROMA_OPENAI_PATH, collection_name: str = OPENAI_COLLECTION) -> Tuple[Chroma, Any]:
    """
    Loads a vector store configured to use OpenAI embeddings with a specified collection.

    This function initializes an OpenAI embeddings object and a persistent client, retrieves the specified
    collection from the database, and initializes a `Chroma` vector store object configured with
    the OpenAI embeddings.

    :param path: The file path where the vector store data is located. Defaults to "./chroma_db_openai".
    :param collection_name: The name of the collection to be used in the vector store.
                            Defaults to "openai_rag_chroma_wikipedia".

    :return: A configured instance of the `Chroma` vector store containing the documents and the `OpenAIEmbeddings` used.
    """
    logger.info(f"Loading vector store")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    persistent_client_openai = chromadb.PersistentClient(path=path)
    collection = persistent_client_openai.get_collection(collection_name)

    vectorstore = Chroma(
        client=persistent_client_openai,
        collection_name= collection_name,
        embedding_function=embeddings
    )
    logger.info("Vector store loaded successfully")
    return vectorstore, embeddings

def setup_vectorstore_nomic(texts: List[Any], path: str = CHROMA_NOMIC_PATH, collection_name: str = NOMIC_COLLECTION) -> Tuple[Chroma, Any]:
    """
    Sets up a vector store using Nomic embeddings (model "nomic-embed-text-v1.5") with documents provided. 
    The function initializes Nomic embeddings, sets up a persistent client, ensures the collection 
    exists (or creates it),  and populates a vector store with the documents.

    :param texts: A list of documents (texts) to be stored in the vector store.
    :param path: The file path where the vector store data will be located. Defaults to "./chroma_db_nomic".
    :param collection: The name of the collection to be used or created in the vector store.
                       Defaults to "opensource_rag_wikipedia".

    :return: A configured instance of the `Chroma` vector store containing the documents and 'Nomic Embeddings' used.
    """
    logger.info(f"Setting up vector store")
    persistent_client_nomic = chromadb.PersistentClient(path=path)
    collection = persistent_client_nomic.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"}) #cosine similarity instead of default L2
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    vectorstore = Chroma.from_documents(
        documents = texts,
        collection_name=collection_name,
        embedding = embeddings,
        client= persistent_client_nomic
    )
    logger.info("Vector store setup complete")
    return vectorstore, embeddings

def load_vectorstore_nomic(path: str = CHROMA_NOMIC_PATH, collection_name: str = NOMIC_COLLECTION) -> Tuple[Chroma, Any]:
    """
    Loads a vector store configured to use the Nomic embeddings with a specified collection.

    This function initializes an embeddings object and a persistent client, retrieves the specified
    collection from the database, and finally initializes and returns a `Chroma` vector store object
    configured with the Nomic embeddings.

    :param path: The file path where the vector store data is located. Defaults to "./chroma_db_nomic".
    :param collection: The name of the collection to be used in the vector store.
    Defaults to "opensource_rag_wikipedia".

    :return: A configured instance of the `Chroma` vector store.
    """
    logger.info(f"Loading vector store")
    persistent_client_nomic = chromadb.PersistentClient(path=path)
    collection = persistent_client_nomic.get_collection(collection_name)
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    vectorstore = Chroma(
        persist_directory= "chroma_db_nomic",
        collection_name=collection_name,
        embedding_function= embeddings
    )
    logger.info("Vector store loaded successfully")
    return vectorstore, embeddings

# ============================================================================
# Retriever
# ============================================================================

def retrieve_documents(topic: str, vectorstore: Chroma, threshold: float = VECTORSTORE_THRESHOLD
) -> List[Any]:
    """
    Retrieves documents from the vector store based on a similarity score threshold.
    
    This function initializes a retriever with the specified threshold for similarity scoring, using the 'similarity_score_threshold' search type.
    It then invokes the retriever with a given topic to fetch documents that meet or exceed the specified similarity score threshold.

    :param topic: The topic query as a string which is used to retrieve relevant documents.
    :param vectorstore: The vector store instance which contains the document embeddings and provides the retrieval mechanism.
    :param threshold: The minimum similarity score threshold. Documents with a similarity score above this threshold will be considered relevant.

    :return: Returns a list of documents or relevant entries that match the topic based on the specified similarity score threshold.
    """
    logger.info(f"Retrieving documents for topic: {topic} with threshold: {threshold}")
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": threshold})
    return retriever.invoke(topic)

def format_documents(docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Formats a list of document objects into a standardized dictionary format.

    This function iterates through each document in the provided list, extracting relevant metadata namely, page content, ID, title, and source (URL).
    Each document is then transformed into a dictionary with keys for 'context', 'id', 'title', and 'source', based on the respective properties and metadata from the document objects.

    :param docs: A list of document objects, each containing content and metadata.
    
    :return: A list of dictionaries where each dictionary represents a document with formatted data.
    """
    logger.info("Formatting documents")
    return [
        {
            "context": doc.page_content,
            "id": doc.metadata['id'],
            "title": doc.metadata['title'],
            "source": doc.metadata['url']
        }
        for doc in docs
    ]

# ============================================================================
# Generation
# ============================================================================
def split_text_into_sections(text: str, max_length: int = MAX_SECTION_LENGTH, min_length: int = MIN_SECTION_LENGTH
) -> List[str]:
    """
    Splits a given text into sections based on specified maximum and minimum lengths.
    
    The function divides the input text into paragraphs, and sequentially adds paragraphs
    to a current section until adding another paragraph would exceed the maximum length.
    If the current section meets the minimum length requirement, it is saved as a separate
    section. This continues until all paragraphs are processed.
    
    :param text: The text to be split into sections.
    :param max_length: The maximum length of each section in characters, defaults to 150.
    :param min_length: The minimum length a section must have to be considered valid, defaults to 50.
    
    :return: A list of text sections that meet the length requirements.
    """
    sections = []
    current_section = ""

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current_section) + len(paragraph) <= max_length:
            current_section += paragraph + " "
        else:
            if len(current_section) >= min_length:
                sections.append(current_section.strip())
            current_section = paragraph + " "
    if len(current_section) >= min_length:
        sections.append(current_section.strip())

    return sections

def calculate_num_pairs(text: str) -> int:
    """
    Calculates the number of sections a text can be divided into using the `split_text_into_sections` function.
    
    This function first divides the text into sections based on predefined criteria in `split_text_into_sections`.
    It then calculates the number of these sections (pairs), returning the total count.

    :param text: The text for which to calculate the number of dividable sections.
    
    :return: The number of sections into which the text was divided.
    """
    sections = split_text_into_sections(text)
    num_pairs = len(sections)
    return num_pairs

def generate_question_answer_pairs_open_source_json(topic: str, vectorstore: Chroma, threshold: float = VECTORSTORE_THRESHOLD) -> str:
    """
    Generates unique question-answer pairs from documents retrieved based on a topic. 

    This function first retrieves documents related to the specified topic with a similarity score above the given threshold.
    It then formats these documents and splits them into smaller sections suitable for generating question-answer pairs.
    Each section is processed to produce unique and contextually relevant question-answer pairs, using the model
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo' via the Together API.

    **Dependencies**:
        - **Together API**: Used to generate question-answer pairs using a model-based approach.
        - **Model**: 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'. This model is specified to generate question-answer
          pairs that are unique and tailored to the educational content based on the document's context.

    :param topic: The topic query to fetch related documents.
    :param vectorstore: The vector store instance used to retrieve document embeddings.
    :param threshold: The similarity score threshold for document retrieval. Defaults to 0.5.

    :return: A JSON string containing a list of unique question-answer pairs generated from the documents.
    """
    logger.info(f"Generating QA pairs")
    client = Together()
    docs = retrieve_documents(topic, vectorstore, threshold)
    formatted_docs = format_documents(docs)

    qa_pairs = []

    for doc in formatted_docs:
        num_pairs = calculate_num_pairs(doc['context'])

        sections = split_text_into_sections(doc['context'])

        pairs_generated = 0

        for section in sections:
            if pairs_generated >= num_pairs:
                break

            if not section.strip():
                continue

            user_prompt = f"""
            Imagine you are a teacher tasked with creating engaging learning materials about the {topic} for your students.
            The focus is on fostering broad knowledge rather than asking specific historical dates.
            Your objective is to generate **unique** question-answer-pair in English that can be used as a flashcard.
            This should be derived from the provided text section in ENGLISH.
            Aim to craft a question and answer that is intriguing and diverse, appropriate for sparking curiosity and discussion.

            Here is the format you should follow:
            {{
                "question": str,
                "answer": str,
                "context": "{section}",
                "source": "{doc['source']}"
            }}

            Ensure the question is derived from the section provided:
            Text section: {section}
            """

            chat_completion, *_ = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Please output valid JSON"},
                    {"role": "user", "content": user_prompt}
                ],
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                response_format={"type": "json_object"},
                temperature=0,
            ).choices

            content = chat_completion.message.content

            try:
                reply = json.loads(content)

                if isinstance(reply, dict):
                    question_unique = all(
                        reply['question'].strip().lower() != existing['question'].strip().lower()
                        for existing in qa_pairs
                    )

                    if question_unique:
                        reply['context'] = section
                        reply['source'] = doc['source']
                        qa_pairs.append(reply)
                        pairs_generated += 1
                        #logger.info(f"Generated QA pair {pairs_generated}")
                    else:
                        logger.warning("Duplicate question detected, generating a new one...")
                else:
                    logger.error(f"Unexpected format in reply: {reply}")

            except json.JSONDecodeError:
               logger.error("Failed to parse JSON response")
               logger.debug(chat_completion)

    logger.info(f"Generated a total of {len(qa_pairs)} QA pairs")
    combined_output = json.dumps(qa_pairs, indent=4)
    return combined_output

def generate_question_answer_pairs_open_ai_json(topic: str, vectorstore: Chroma, threshold: float = VECTORSTORE_THRESHOLD) -> str:
    """
    Generates unique question-answer pairs from documents retrieved based on a topic. 

    This function first retrieves documents related to the specified topic with a similarity score above the given threshold.
    It then formats these documents and splits them into smaller sections suitable for generating question-answer pairs.
    Each section is processed to produce unique and contextually relevant question-answer pairs, using the model
    'gpt-4-turbo' via the OpenAI API.

    **Dependencies**:
        - **OpenAI API**: Used to generate question-answer pairs using a model-based approach.
        - **Model**: 'gpt-4-turbo'. This model is specified for its ability to generate detailed and contextually
          accurate question-answer pairs.

    :param topic: The topic query to fetch related documents.
    :param vectorstore: The vector store instance used to retrieve document embeddings.
    :param threshold: The similarity score threshold for document retrieval. Defaults to 0.5.

    :return: A JSON string containing a list of unique question-answer pairs generated from the documents.
    """
    logger.info(f"Generating QA pairs")
    client = OpenAI()

    docs = retrieve_documents(topic, vectorstore, threshold)
    formatted_docs = format_documents(docs)
    qa_pairs = []

    for doc in formatted_docs:
        num_pairs = calculate_num_pairs(doc['context'])

        sections = split_text_into_sections(doc['context'])

        pairs_generated = 0

        for section in sections:
            
            if not section.strip():
                continue

            if pairs_generated >= num_pairs:
                break

            user_prompt = f"""
            Imagine you are a teacher tasked with creating engaging learning materials about the {topic} for your students.
            The focus is on fostering broad knowledge rather than asking specific historical dates.
            Your objective is to generate **unique** question-answer-pair in English that can be used as a flashcard.
            This should be derived from the provided text section in ENGLISH.
            Aim to craft a question and answer that is intriguing and diverse, appropriate for sparking curiosity and discussion.

            Here is the format you should follow:
            {{
                "question": str,
                "answer": str,
                "context": "{section}",
                "source": "{doc['source']}"
            }}

            Ensure the question is derived from the section provided:
            Text section: {section}
            """

            chat_completion, *_ = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Please output valid JSON"},
                    {"role": "user", "content": user_prompt}
                ],
                model="gpt-4-turbo",
                response_format={"type": "json_object"},
                temperature=0,
            ).choices

            content = chat_completion.message.content

            try:
                reply = json.loads(content)

                if isinstance(reply, dict):
                    question_unique = all(
                        reply['question'].strip().lower() != existing['question'].strip().lower()
                        for existing in qa_pairs
                    )

                    if question_unique:
                        reply['context'] = section
                        reply['source'] = doc['source']
                        qa_pairs.append(reply)
                        pairs_generated += 1
                        #logger.info(f"Generated QA pair {pairs_generated}")
                    else:
                        logger.warning("Duplicate question detected, generating a new one...")
                else:
                    logger.error(f"Unexpected format in reply: {reply}")

            except json.JSONDecodeError:
               logger.error("Failed to parse JSON response")
               logger.debug(chat_completion)

    logger.info(f"Generated a total of {len(qa_pairs)} QA pairs")
    combined_output = json.dumps(qa_pairs, indent=4)
    return combined_output

# ============================================================================
# Evaluation
# ============================================================================
def evaluate_retriever(topic: str, vectorstore: Chroma, k: int = 10) -> pd.DataFrame:
    """
    Evaluates the performance of a vector store's retrieval capabilities by conducting searches using
    both cosine similarity and cosine distance metrics, and then compares the results.

    This function retrieves the top 'k' documents based on the cosine similarity and cosine distance
    metrics for a given topic. It constructs a DataFrame containing the results, including each
    document's ID, URL, context, similarity score, and distance score.

    :param topic: The topic string based on which the documents are retrieved.
    :param vectorstore: An instance of a vector store configured with embeddings used for retrieving documents.
    :param k: The number of top documents to retrieve, defaults to 10.
    
    :return: A pandas DataFrame containing the document details and their respective retrieval scores.
    """
    cosine_similarity_results = vectorstore.similarity_search_with_relevance_scores(topic, k=k)
    cosine_distance_results = vectorstore.similarity_search_with_score(topic, k=k)

    scores = []

    for doc, similarity_score in cosine_similarity_results:
        distance_score = next((score for (d, score) in cosine_distance_results if d.metadata['id'] == doc.metadata['id']), None)
        
        row = {
            'ID': doc.metadata['id'],
            'URL': doc.metadata['url'],
            'Context': doc.page_content,
            'Similarity Score': similarity_score,
            'Distance Score': distance_score
        }
        scores.append(row)
    df = pd.DataFrame(scores)
    logger.info("Retriever evaluation complete")
    return df

def ragas_prepare_data(dataset: pd.DataFrame, ground_truth: bool = True) -> Dataset:
    """
    Prepares a dataset for use of the Ragas evaluation metrics by transforming
    and ensuring proper formatting of its columns. This function resets the dataset index, ensures
    'contexts' are lists of strings, optionally adds a 'ground_truth' field, removes unnecessary
    columns, and casts the dataset to specified features.

    :param dataset: The dataset to prepare, which should contain at least the 'question', 'answer', 'contexts' and 'source' field.
    :param ground_truth: If True, includes the 'ground_truth' field in the final dataset.

    :return: The transformed dataset with the specified features format.
    :raises ValueError: If the data type of the 'contexts' field is neither a string nor a list.
    """
    ragas_data = dataset.reset_index(drop=True)
    ragas_data = Dataset.from_pandas(ragas_data)

    def ensure_list(value):
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Unexpected data type: {type(value)}")

    ragas_data = ragas_data.map(lambda example: {'contexts': ensure_list(example['contexts'])})

    feature_dict = {
        'question': Value('string'),
        'answer': Value('string'),
        'contexts': Sequence(Value('string')),
        'source': Value('string')
    }

    if ground_truth:
        feature_dict['ground_truth'] = Value('string')

    features = Features(feature_dict)
    ragas_data = ragas_data.remove_columns([col for col in ragas_data.column_names if col not in features])
    ragas_data = ragas_data.cast(features)

    logger.info("Data prepared for Ragas evaluation")
    return ragas_data

def ragas_evaluate_data(
    data: Dataset,
    ChatOpenAI_model_name: str = "gpt-4",
    temperature: float = 0,
    ground_truth: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the prepared dataset using various evaluadtion metrics with a specified ChatGPT model as additional large language model.
    It wraps the ChatGPT model in a LangchainLLMWrapper and calculates different metrics based on whether ground truth is considered.

    :param data: The prepared dataset to evaluate.
    :param ChatOpenAI_model_name: The model name of the ChatGPT to use for evaluation.
    :param temperature: The sampling temperature to use in the model inference, defaults to 0.
    :param ground_truth: Specifies whether metrics that require ground truth data should be included.

    :return: The evaluation results with specified metrics.

    """
    llm_instance = LangchainLLMWrapper(ChatOpenAI(model_name=ChatOpenAI_model_name, temperature=temperature))
    if ground_truth:
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, context_entity_recall, answer_correctness]
    else:
        metrics=[faithfulness, answer_relevancy]
    ragas_evaluation_data = ragas_evaluate(data, metrics=metrics, llm=llm_instance)
    logger.info("Ragas evaluation complete")
    return ragas_evaluation_data


def deepeval_prepare_data(dataset: pd.DataFrame, ground_truth: bool = True) -> List[LLMTestCase]:
    """
    Prepares a dataset for using DeepEval evaluation metrics by creating test cases for each entry in the dataset. 
    Each test case is structured to include a question, the actual and expected answers (if ground truth is provided),
    and the retrieval context.

    :param dataset: The dataset to be prepared, typically containing 'question', 'answer', and 'contexts' columns,
                    and optionally a 'ground_truth' column if ground_truth is True.
    :param ground_truth: Specifies whether the 'ground_truth' column should be included in the test cases.

    :return: A list of test cases, each designed for evaluation purposes.
    """
    logger.info("Preparing data for DeepEval evaluation")
    deepeval_data = []
    for index, row in dataset.iterrows():
        if ground_truth:
            test_case = LLMTestCase(
            input=row['question'],
            actual_output=row['answer'],
            expected_output=row['ground_truth'],
            retrieval_context=[row['contexts']]
        )
        else:
            test_case = LLMTestCase(
            input=row['question'],
            actual_output=row['answer'],
            retrieval_context=[row['contexts']]
        )
        deepeval_data.append(test_case)
    logger.info("Data prepared for DeepEval evaluation")
    return deepeval_data

def deepeval_evaluate_data(
    deepeval_data: List[LLMTestCase],
    ground_truth: bool = True,
    model: str = "gpt-4",
) -> None:
    """
    Evaluates a prepared dataset using a suite of DeepEval's metrics designed to assess the performance of language models. 
    It outputs a structured report summarizing the results of each metric evaluation based on whether ground truth is considered.

    :param deepeval_data: The dataset containing test cases prepared for evaluation.
    :param ground_truth: Specifies whether metrics that require ground truth data should be included.
    :param model: The name of the model to be used for evaluation, defaults to "gpt-4".

    :return: Outputs a structured report summarizing the metric evaluations.
    """
    logger.info("Starting DeepEval evaluation")
    metrics = []

    deepeval_faithfulness = FaithfulnessMetric(model=model)
    deepeval_answer_relevancy = AnswerRelevancyMetric(model=model)
    deepeval_contextual_relevancy = ContextualRelevancyMetric(model=model)
    deepeval_bias = BiasMetric(model=model)
    deepeval_toxicity = ToxicityMetric(model=model)
    
    metrics.extend([
        deepeval_faithfulness,
        deepeval_answer_relevancy,
        deepeval_contextual_relevancy,
        deepeval_bias,
        deepeval_toxicity
    ])

    if ground_truth:
        deepeval_contextual_precision = ContextualPrecisionMetric(model=model)
        deepeval_contextual_recall = ContextualRecallMetric(model=model)
        
        metrics.extend([
            deepeval_contextual_precision,
            deepeval_contextual_recall
        ])
    
    deepeval_evaluate(
        test_cases=deepeval_data,
        metrics=metrics
    )
    logger.info("DeepEval evaluation complete")

def haystack_prepare_data(dataset: pd.DataFrame, ground_truth: bool = True) -> List[Dict[str, Any]]:
    """
    Prepares a dataset for using Haystack's evaluation metrics by formatting the input data into a
    specific structure.

    :param dataset: The dataset to be prepared, typically containing 'question', 'answer', and 'contexts' columns,
                    and optionally a 'ground_truth' column if ground_truth is True.
    :param ground_truth: Specifies whether metrics that require ground truth data should be included.

    :return: A list of dictionaries, each containing data for a single instance. Each dictionary includes
        the question, predicted answers, and context(s). If `ground_truth` is True, it also includes
        ground truth answers.
    """
    logger.info("Preparing data for Haystack evaluation")
    haystack_data = []

    for index, row in dataset.iterrows():
        if ground_truth:
            questions = [row['question']]
            predicted_answers = [row['answer']]
            ground_truth_answers = [row['ground_truth']]
            contexts = [row['contexts']]

            test_case = {
                "questions": questions,
                "predicted_answers": predicted_answers,
                "ground_truth_answers": ground_truth_answers,
                "contexts": contexts
            }
        else:
            questions = [row['question']]
            predicted_answers = [row['answer']]
            contexts = [row['contexts']]

            test_case = {
                "questions": questions,
                "predicted_answers": predicted_answers,
                "contexts": contexts
            }

        haystack_data.append(test_case)
    logger.info("Data prepared for Haystack evaluation")
    return haystack_data

def haystack_evaluate_data(data: List[Dict[str, Any]], ground_truth: bool = True
) -> pd.DataFrame:
    """
    Evaluates the prepared dataset by running it through a series of Haystack's evaluation metrics within a
    pipeline. Metrics include context relevance, faithfulness of the answers, and optionally
    semantic answer similarity if ground_truth is True.

    :param data: A list of dictionaries where each dictionary contains 'questions', 'contexts',
        'predicted_answers', and optionally 'ground_truth_answers' if ground_truth is True.
    :param ground_truth: A flag to determine whether semantic similarity evaluation should be
        performed using the ground truth answers provided in the data.

    :return: A pandas DataFrame containing the evaluation results for each data point, including scores
        for context relevance, faithfulness, and optionally semantic similarity along with the overall
        scores for these metrics.
    """
    logger.info("Starting Haystack evaluation")
    evaluation_pipeline = Pipeline()
    context_relevance_evaluator = ContextRelevanceEvaluator()
    faithfulness_evaluator = FaithfulnessEvaluator()

    evaluation_pipeline.add_component("context_relevance_evaluator", context_relevance_evaluator)
    evaluation_pipeline.add_component("faithfulness_evaluator", faithfulness_evaluator)

    if ground_truth:
        semanticanswersimilarity_evaluator = SASEvaluator()
        evaluation_pipeline.add_component("semanticanswersimilarity_evaluator", semanticanswersimilarity_evaluator)

    haystack_evaluation_data = []
    for test_case in data:
        evaluation_input = {
            "context_relevance_evaluator": {
                "questions": test_case['questions'], 
                "contexts": test_case['contexts']
            },
            "faithfulness_evaluator": {
                "questions": test_case['questions'], 
                "contexts": test_case['contexts'], 
                "predicted_answers": test_case['predicted_answers']
            }
        }
        if ground_truth:
            evaluation_input["semanticanswersimilarity_evaluator"] = {
                "predicted_answers": test_case['predicted_answers'],
                "ground_truth_answers": test_case['ground_truth_answers']
            }

        result = evaluation_pipeline.run(evaluation_input)

        for i in range(len(test_case['questions'])):
            individual_data = {
                "question": test_case['questions'][i],
                "predicted_answer": test_case['predicted_answers'][i],
                "context": test_case['contexts'][i],
                "context_relevance_score": result["context_relevance_evaluator"]["individual_scores"][i],
                "faithfulness_score": result["faithfulness_evaluator"]["individual_scores"][i]
            }
            if ground_truth:
                individual_data["semantic_similarity_score"] = result["semanticanswersimilarity_evaluator"]["individual_scores"][i]
                individual_data["ground_truth_answer"] = test_case['ground_truth_answers'][i]
            
            haystack_evaluation_data.append(individual_data)

    df_haystack_evaluation_data = pd.DataFrame(haystack_evaluation_data)

    overall_scores = {
        "context_relevance_score": result["context_relevance_evaluator"]["score"],
        "faithfulness_score": result["faithfulness_evaluator"]["score"]
    }
    if ground_truth:
        overall_scores["semantic_similarity_score"] = result["semanticanswersimilarity_evaluator"]["score"]
    logger.info(f"Overall Scores: {overall_scores}")
    return df_haystack_evaluation_data

# ============================================================================
# Utils
# ============================================================================
def prepare_dataset(flashcard_set: str) -> Dataset:
    """
    Converts a JSON string of flashcard data into a structured dataset.

    :param flashcard_set: A JSON string that represents a set of flashcards. Each flashcard is a dictionary
                          with keys 'question', 'answer', 'context', and 'source'.

    :return: A dataset object with structured fields for questions, answers, contexts, and sources. The dataset
              has features defined for each field to ensure correct data types.
    """
    logger.info("Preparing dataset from JSON string")
    flashcard_set = json.loads(flashcard_set)

    formatted_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'source': []
    }

    for item in flashcard_set:
        formatted_data['question'].append(item['question'])
        formatted_data['answer'].append(item['answer'])
        formatted_data['contexts'].append([item['context']])
        formatted_data['source'].append(item['source'])

    dataset = Dataset.from_dict(formatted_data)

    features = Features({
        'question': Value('string'),
        'answer': Value('string'),
        'contexts': Sequence(Value('string')),
        'source': Value('string')
    })

    dataset = dataset.cast(features)
    logger.info("Dataset preparation complete")
    return dataset

def save_qa_pairs(dataset: Any,
    folder_name: str,
    topic: str,
    pipeline_name: str,
    content: str = "generated_qas",
) -> None:
    """
    Saves the provided dataset as a CSV file formatted for question-answer pairs,
    organized by topic, pipeline name and content.

    :param dataset: The dataset containing the question-answer pairs. It can be a pandas DataFrame
                    or any object that has a `to_pandas()` method.
    :param folder_name: The name of the folder where the CSV file will be saved.
    :param topic: The main topic of the question-answer pairs, used in the file name.
                  The spaces in the topic will be replaced with underscores and converted to lowercase.
    :param pipeline_name: The name of the pipeline used to generate the question-answer pairs.
    :param content: An optional descriptor that precedes the file naming convention.
                    Defaults to 'generated_qas'.

    :return: None
    :raises Exception: If the dataset cannot be converted to a pandas DataFrame.

    Upon successful saving of the file, this function prints "File saved as CSV." to the standard output.
    The file is saved with a semicolon as the delimiter and commas as the decimal separator.
    """
    logger.info(f"Saving QA pairs to CSV: {folder_name}/{content}_{topic}_{pipeline_name}.csv")
    topic = topic.lower().replace(" ", "_")
    if not isinstance(dataset, pd.DataFrame):
        try:
            dataset = dataset.to_pandas()
        except Exception as e:
            logger.error(f"Error converting dataset to DataFrame: {e}")
            return
    os.makedirs(folder_name, exist_ok=True)
    dataset.to_csv(f'{folder_name}/{content}_{topic}_{pipeline_name}.csv', index=True, sep=';', decimal=',', header=True)
    logger.info("File saved as CSV.")

def load_qa_pairs(
    folder_name: str,
    topic: str,
    pipeline_name: str,
    content: str = "generated_qas_gt",
) -> pd.DataFrame:
    """
    Loads question-answer pairs from a CSV file into a pandas DataFrame. The CSV file must be
    named according to a specific naming convention and located in the specified folder.

    :param folder_name: The folder where the CSV file is stored.
    :param topic: The main topic of the question-answer pairs, used in the file name.
                  The spaces in the topic will be replaced with underscores and converted to lowercase.
    :param pipeline_name: The name of the pipeline used to generate the question-answer pairs,
                          used in the file name.
    :param content: An optional descriptor for the file name, defaults to 'generated_qas_gt'.

    :return: A pandas DataFrame containing the loaded question-answer pairs. The 'contexts' column
             is processed to remove square brackets and single quotes.
    """
    logger.info(f"Loading QA pairs from CSV: {folder_name}/{content}_{topic}_{pipeline_name}.csv")
    topic = topic.lower().replace(" ", "_")
    dataset_gt = pd.read_csv(f'{folder_name}/{content}_{topic}_{pipeline_name}.csv', delimiter=';', decimal=',', encoding='utf-8', index_col = 0)
    dataset_gt['contexts'] = dataset_gt['contexts'].apply(lambda x: x.strip("[]").replace("'", ""))
    logger.info("QA pairs loaded successfully")
    return dataset_gt