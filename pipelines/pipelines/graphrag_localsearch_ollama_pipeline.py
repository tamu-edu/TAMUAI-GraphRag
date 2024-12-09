"""
title: GraphRAG Local Search Pipeline
author: Chuck Zuo, Ph.D.
date: 2024-09-09
version: 1.0
license: MIT
description: A pipeline for retrieving and synthesizing information using GraphRAG local search engines.
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import logging
import asyncio

# Define the GraphRAG Localsearch pipeline
class Pipeline:
    class Valves(BaseModel):
        GRAPHRAG_OLLAMA_BASE_URL: str
        GRAPHRAG_OLLAMA_API_KEY: str
        GRAPHRAG_MODEL_NAME: str
        GRAPHRAG_EMBEDDING_MODEL_NAME: str
        GRAPHRAG_INPUT_DIR: str

    def __init__(self):
        self.llm = None
        self.local_search_engine = None
        self.token_encoder = None
        self.name = "GraphRag Localsearch Pipeline"
        self.valves = self.Valves(
            **{
                "GRAPHRAG_OLLAMA_BASE_URL": os.getenv("GRAPHRAG_OLLAMA_BASE_URL", "http://localhost:11434"),
                "GRAPHRAG_OLLAMA_API_KEY": os.getenv("GRAPHRAG_OLLAMA_API_KEY", "your-api-key-here"),
                "GRAPHRAG_MODEL_NAME": os.getenv("GRAPHRAG_MODEL_NAME", "llama3.1"),
                "GRAPHRAG_EMBEDDING_MODEL_NAME": os.getenv("GRAPHRAG_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "GRAPHRAG_INPUT_DIR": os.getenv("GRAPHRAG_INPUT_DIR", "data"),
            }
        )

    async def on_startup(self):
        # GraphRAG related imports
        from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
        from graphrag.query.indexer_adapters import (
            read_indexer_covariates,
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
        from graphrag.query.llm.oai.chat_openai import ChatOpenAI
        from graphrag.query.llm.oai.embedding import OpenAIEmbedding
        from graphrag.query.llm.oai.typing import OpenaiApiType
        from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
        from graphrag.query.structured_search.local_search.search import LocalSearch
        from graphrag.vector_stores.lancedb import LanceDBVectorStore
        import tiktoken
        import pandas as pd

        # Load necessary environment variables
        INPUT_DIR = self.valves.GRAPHRAG_INPUT_DIR
        LANCEDB_URI = f"{INPUT_DIR}/lancedb"
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        RELATIONSHIP_TABLE = "create_final_relationships"
        COVARIATE_TABLE = "create_final_covariates"
        TEXT_UNIT_TABLE = "create_final_text_units"
        COMMUNITY_LEVEL = 2

        # Initialize LLM and token encoder
        self.llm = ChatOpenAI(
            api_key=self.valves.GRAPHRAG_OLLAMA_API_KEY,
            api_base=self.valves.GRAPHRAG_OLLAMA_BASE_URL,
            model=self.valves.GRAPHRAG_MODEL_NAME,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )

        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        # Initialize text embedding model
        self.text_embedder = OpenAIEmbedding(
            api_key=self.valves.GRAPHRAG_OLLAMA_API_KEY,
            api_base=self.valves.GRAPHRAG_OLLAMA_BASE_URL,
            api_type=OpenaiApiType.OpenAI,
            model=self.valves.GRAPHRAG_EMBEDDING_MODEL_NAME,
            deployment_name=self.valves.GRAPHRAG_EMBEDDING_MODEL_NAME,
            max_retries=20,
        )

        # Load embeddings and entity data
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        # Initialize LanceDB for storing embeddings
        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        # Load relationships
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        # Load community reports
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

        # Load text units
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)
        
        # Load covariates
        covariates = None
        covariate_file_path = f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet"
        if os.path.exists(covariate_file_path):
            covariate_df = pd.read_parquet(covariate_file_path)
            if not covariate_df.empty:
                claims = read_indexer_covariates(covariate_df)
                covariates = {"claims": claims}
                print(f"Number of claim records: {len(claims)}")
            else:
                print("Covariate file is empty. Skipping covariates.")
        else:
            print("Covariate file not found. Skipping covariates.")

        # Set up local search engine
        local_context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates if covariates else None,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        local_llm_params = {
            "max_tokens": 4000,
            "temperature": 0.0,
        }

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
        }

        self.local_search_engine = LocalSearch(
            llm=self.llm,
            context_builder=local_context_builder,
            token_encoder=self.token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )

        logging.info("Local search engine initialized.")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logging.info(f"Received message: {user_message}")
        if self.local_search_engine is None:
            logging.error("Local search engine is not initialized")
            return "Error: Search engine not initialized."

        # Perform the local search
        try:
            result = asyncio.run(self.local_search_engine.asearch(user_message)) 
            logging.info("Search completed successfully.")
            return result.response
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return f"Error during search: {str(e)}"


        
        