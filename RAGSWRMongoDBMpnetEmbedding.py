import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Note: Store the OpenAi API key in the Environment Variables

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
# The temperature is used to control the randomness of the output.
# When you set it higher, you'll get more random outputs.
# When you set it lower, towards 0, the values are more deterministic.

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# MongoDB Atlas Connection Details
mongodb_conn_string = (
    "mongodb+srv://<username>:<password>@<cluster>.<server>.mongodb.net/"
)
db_name = "RAGSentenceWindowRetrieval"
collection_name = "SentenceWindow"
index_name = "vector_index"

# Initialize MongoDB python client
mongo_client = MongoClient(mongodb_conn_string)
collection = mongo_client[db_name][collection_name]

# Initialize the MongoDB Atlas Vector Store.
vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name=db_name,
    collection_name=collection_name,
    index_name=index_name,
    embedding_key="embedding",
)

def UploadEmbeddingstoAtlas():
    # Load the input data into Document list.
    documents = SimpleDirectoryReader(input_files=["./input_text.txt"]).load_data(
        show_progress=True
    )

    # Reset w/out deleting the Search Index
    collection.delete_many({})

    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,#The number of sentences on each side of a sentence to capture.
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    sentence_nodes = node_parser.get_nodes_from_documents(documents)
    for i in range(0, len(sentence_nodes)):
        print("==================")
        print(f"Text {str(i+1)}: \n{sentence_nodes[i].text}")
        print("------------------")
        print(f"Window {str(i+1)}: \n{sentence_nodes[i].metadata['window']}")
        print("==================")


    print("Initiated Embedding Creation")
    print("------------------")
    start_time = time.time()

    for node in sentence_nodes:
        node.embedding = Settings.embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )

    print("Embedding Completed In {:.2f} sec".format(time.time() - start_time))

    start_time = time.time()

    # Add nodes to MongoDB Atlas Vector Store.
    vector_store.add(sentence_nodes)

    print(
        "Embedding Saved in MongoDB Atlas Vector in {:.2f} sec".format(
            time.time() - start_time
        )
    )

def AskQuestions():
    # Retrieve Vector Store Index.
    sentence_index = VectorStoreIndex.from_vector_store(vector_store)

    # In advanced RAG, the MetadataReplacementPostProcessor is used to replace the sentence in each node
    # with it's surrounding context as part of the sentence-window-retrieval method.
    # The target key defaults to 'window' to match the node_parser's default
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # For advanced RAG, add a re-ranker to re-ranks the retrieved context for its relevance to the query.
    # Note : Retrieve a larger number of similarity_top_k, which will be reduced to top_n.
    # BAAI/bge-reranker-base
    # link: https://huggingface.co/BAAI/bge-reranker-base
    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")


    # The QueryEngine class is equipped with the generator
    # and facilitates the retrieval and generation steps
    # Set vector_store_query_mode to "hybrid" to enable hybrid search with an additional alpha parameter
    # to control the weighting between semantic and keyword based search.
    # The alpha parameter specifies the weighting between vector search and keyword-based search,
    # where alpha=0 means keyword-based search and alpha=1 means pure vector search.
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=6,
        vector_store_query_mode="hybrid",
        alpha=0.5,
        node_postprocessors=[postproc, rerank],
    )


    # Load the question to ask the RAG into Document list.
    question_documents = []
    with open(file=".\questions.txt", encoding="ascii") as fIn:
        question_documents = set(fIn.readlines())

    question_documents = list(question_documents)

    # Now, run Advanced RAG queries on your data using the Default RAG queries

    for i in range(0, len(question_documents)):
        if question_documents[i].strip():
            print("==================")
            print(f"Question {str(i+1)}: \n{question_documents[i].strip()}")
            print("------------------")
            response = query_engine.query(question_documents[i].strip())
            print(
                f"Advanced RAG Response for Question {str(i+1)}: \n{str(response).strip()}"
            )
            time.sleep(20)
            print("------------------")

            if(str(response).strip()!='Empty Response'):
                window = response.source_nodes[0].node.metadata["window"]
                sentence = response.source_nodes[0].node.metadata["original_text"]

                print(f"Referenced Window for Question {str(i+1)}:\n {window}")
                print("------------------")
                print(f"Original Response Sentence for Question {str(i+1)}: \n{sentence}")
                print("==================")


#Only required to run once for creating and storing the embedding to the MongoDB Atlas Cloud
UploadEmbeddingstoAtlas()

#Run the retrieve Advanced RAG queries responses for queries in question.txt file
AskQuestions()