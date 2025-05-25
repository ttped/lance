# rag_system.py
import os # Added
from dotenv import load_dotenv # Added
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import sqlite3
import json
import re

# Langchain and document processing imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

import pandas as pd # For CSV handling in DocumentProcessor

# Import the Ollama-specific base agent class from agent_ollama.py
from agent_ollama import EnhancedEngineeringAgent # This is the base agent from your agent_ollama.py

load_dotenv() # Added

# Configuration from environment variables, with defaults matching original hardcoded values
SQLITE_DATABASE_FILE = os.getenv("SQLITE_DB_FILE_PATH", "engineering_docs.sqlite")
PROCESSED_DOCUMENTS_TABLE_NAME = os.getenv("SQLITE_TABLE_NAME", "processed_documents")
DEFAULT_VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./ollama_chroma_db")


# --- DocumentProcessor Class ---
class DocumentProcessor:
    """Process various document types and prepare them for RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""], # Common separators
            add_start_index=True # Helps in locating chunks if needed
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type and add common metadata."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        common_metadata = {
            "source": path.name, # Filename as source
            "file_path": str(path.resolve()), # Full path
            "file_type": extension.replace('.', ''), # e.g., "pdf", "csv"
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "size_bytes": path.stat().st_size
        }

        loader_map = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
        }
        
        if extension not in loader_map:
            print(f"Warning: Unsupported file type '{extension}' for file {file_path}. Skipping.")
            return []
        
        loader_class = loader_map[extension]
        
        # Special handling for CSV to improve content representation
        if extension == '.csv':
            try:
                # CSVLoader by default creates one Document per row.
                # We can load and then enhance metadata, or provide content differently.
                # Here, we'll use default behavior and augment metadata.
                csv_loader = CSVLoader(file_path=file_path) # Can specify encoding if needed
                documents = csv_loader.load()
                for i, doc in enumerate(documents):
                    doc.metadata.update(common_metadata)
                    doc.metadata["row"] = i + 1 # Add 1-based row index
                    # Content is typically "col1: val1\n col2: val2..." which is good for LLMs
                return documents
            except Exception as e:
                print(f"Error loading CSV {file_path}: {e}")
                return []

        try:
            loader = loader_class(file_path)
            documents = loader.load()
            # Add common metadata to all loaded document parts (pages, slides, etc.)
            for doc in documents:
                doc.metadata.update(common_metadata)
            return documents
        except Exception as e:
            print(f"Error loading document {file_path} with {loader_class.__name__}: {e}")
            return []

    def process_documents(self, loaded_docs: List[Document], source_file_name: str) -> List[Document]:
        """Process loaded documents into smaller chunks with enhanced metadata."""
        processed_chunks: List[Document] = []
        
        for doc_idx, doc in enumerate(loaded_docs):
            # Split document content into text chunks
            chunks_texts = self.text_splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(chunks_texts):
                # Generate a unique ID for the chunk
                chunk_id = self._generate_chunk_id(source_file_name, doc_idx, chunk_idx, chunk_text)
                
                # Inherit metadata from parent document and add chunk-specific info
                chunk_metadata = doc.metadata.copy() # Start with parent doc's metadata (source, file_path, page, etc.)
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "original_doc_index_in_file": doc_idx, # If a file loaded multiple Document objects (e.g. pages)
                    "chunk_index_in_doc": chunk_idx, # Index of this chunk within its parent Document object
                    "text_length_chars": len(chunk_text),
                    # 'start_index' from RecursiveCharacterTextSplitter can be added here if needed
                })
                
                # Extract any part numbers mentioned in this specific chunk
                part_numbers_in_chunk = self._extract_part_references(chunk_text)
                if part_numbers_in_chunk:
                    chunk_metadata["part_numbers_in_chunk"] = part_numbers_in_chunk
                
                new_chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                processed_chunks.append(new_chunk_doc)
        
        return processed_chunks
    
    def _generate_chunk_id(self, source_file: str, doc_idx: int, chunk_idx: int, text_content: str) -> str:
        """Generate a unique ID for each chunk using file name, indices, and content hash."""
        content_hash = hashlib.md5(text_content.encode('utf-8', errors='ignore')).hexdigest()[:8] # Short hash
        # Sanitize source_file name for ID generation if it contains problematic characters
        safe_source_file_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(source_file).stem)
        return f"{safe_source_file_stem}_doc{doc_idx}_chunk{chunk_idx}_{content_hash}"
    
    def _extract_part_references(self, text: str) -> List[str]:
        """Extract part number like patterns from text."""
        patterns = [
            r'part[-\s]?(\d+)',             # e.g., part-123, part 123
            r'p/n[-\s]?([A-Z0-9-]+)',       # e.g., p/n ABC-123
            r'component[-\s]?([A-Z0-9]+)',  # e.g., component-XYZ, component XYZ
            r'item no\.? ([A-Z0-9-]+)',     # e.g., item no. 123-ABC
        ]
        part_numbers: List[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            part_numbers.extend(matches)
        return list(set(part_numbers)) # Return unique matches


# --- CitationRAG Class ---
class CitationRAG:
    """RAG system with citation support, using a specified embedding model."""

    def __init__(self, vector_store_path: str = None, embedding_model: Any = None): # Modified default
        if embedding_model is None:
            raise ValueError("CitationRAG requires an embedding_model instance (e.g., OllamaEmbeddings).")
        self.embeddings = embedding_model

        _vector_store_path = vector_store_path if vector_store_path is not None else DEFAULT_VECTOR_STORE_PATH # Use loaded default

        Path(_vector_store_path).mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=_vector_store_path,
            embedding_function=self.embeddings
        )
        self.document_processor = DocumentProcessor()
        print(f"CitationRAG (Ollama) initialized. Vector store: '{_vector_store_path}'. Current documents in DB: {self.vector_store._collection.count()}")
    
    def index_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Load, process, and index multiple documents into the vector store."""
        results: Dict[str, Any] = {"processed_files": [], "failed_files": [], "total_chunks_added": 0}
        all_chunks_to_add: List[Document] = []

        for file_path_str in file_paths:
            file_path_obj = Path(file_path_str)
            if not file_path_obj.exists() or not file_path_obj.is_file():
                print(f"File not found or is not a file: {file_path_str}. Skipping.")
                results["failed_files"].append({"file": file_path_str, "error": "File not found or not a file"})
                continue
            
            try:
                print(f"Loading document: {file_path_obj.name}...")
                raw_docs = self.document_processor.load_document(file_path_str)
                if not raw_docs:
                    print(f"No content loaded from {file_path_obj.name}. Skipping processing.")
                    results["failed_files"].append({"file": file_path_str, "error": "No content loaded from file"})
                    continue

                print(f"Processing {len(raw_docs)} sections from {file_path_obj.name} into chunks...")
                chunks = self.document_processor.process_documents(raw_docs, file_path_obj.name)
                
                if chunks:
                    all_chunks_to_add.extend(chunks)
                    results["processed_files"].append({"file": file_path_str, "chunks_created": len(chunks)})
                else:
                    print(f"No chunks created from {file_path_obj.name}.")
                    # Not necessarily a failure if file was empty or unsupported type handled by loader
                    results["failed_files"].append({"file": file_path_str, "error": "No chunks created during processing"})

            except Exception as e:
                print(f"Failed to process/index {file_path_str}: {e}")
                results["failed_files"].append({"file": file_path_str, "error": str(e)})
        
        if all_chunks_to_add:
            print(f"Adding {len(all_chunks_to_add)} total chunks to vector store...")
            self.vector_store.add_documents(all_chunks_to_add)
            self.vector_store.persist() # Persist changes to disk
            results["total_chunks_added"] = len(all_chunks_to_add)
            print(f"Successfully added {len(all_chunks_to_add)} chunks. DB count now: {self.vector_store._collection.count()}")
        else:
            print("No new chunks were prepared to be added to the vector store in this batch.")
            
        return results
    
    def search_with_citations(
        self, 
        query: str, 
        k: int = 3, # Default number of documents to retrieve
        part_numbers: Optional[List[str]] = None # Optional context, not used for filtering here yet
    ) -> List[Dict[str, Any]]:
        """Search documents and return results formatted with citation information."""
        print(f"Searching RAG for: '{query}', k={k}" + (f", with part context: {part_numbers}" if part_numbers else ""))
        
        # Perform similarity search
        # Note: metadata filtering for Chroma can be added here if needed, using the `filter` argument.
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results: List[Dict[str, Any]] = []
        for doc, score in results_with_scores:
            metadata = doc.metadata
            source_file = metadata.get("source", "Unknown Source") # Should be filename
            page_number = metadata.get("page", metadata.get("page_label", None)) # For PDFs
            row_number = metadata.get("row", None) # For CSVs

            # Construct a human-readable citation string
            citation_str = f"[{source_file}"
            if page_number is not None: # Check for None explicitly, as page 0 can be valid
                citation_str += f", Page {page_number}"
            elif row_number is not None:
                 citation_str += f", Row {row_number}"
            citation_str += "]"

            formatted_results.append({
                "content": doc.page_content,
                "score": float(score), # Ensure score is a standard float
                "source": source_file,
                "page": page_number,
                "row": row_number,
                "chunk_id": metadata.get("chunk_id"), # Useful for unique reference
                "file_path": metadata.get("file_path"), # Full path if needed by frontend
                "metadata": metadata, # Include all metadata for flexibility
                "citation_string": citation_str
            })
        
        print(f"Found {len(formatted_results)} RAG results.")
        return formatted_results

    def _load_documents_from_sqlite(self, db_path: str, table_name: str) -> List[Document]:
        """
        Loads all processed texts (pages/slides) from the SQLite database
        and converts them into Langchain Document objects.
        """
        docs_from_db: List[Document] = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Use the passed table_name parameter
            cursor.execute(
                f"SELECT filename, file_type, file_path_original, file_hash_sha256, total_units, content_units_json FROM {table_name}"
            )

            for row in cursor.fetchall():
                (filename, file_type, file_path_original, file_hash,
                 total_units, content_units_json_str) = row

                if not content_units_json_str: # Handle case where JSON might be empty/null
                    print(f"Warning: Empty content_units_json for {filename} in table {table_name}. Skipping this DB entry.")
                    continue
                try:
                    content_units = json.loads(content_units_json_str)
                except json.JSONDecodeError as je:
                    print(f"Warning: Could not decode JSON for {filename} in table {table_name}: {je}. Skipping this DB entry.")
                    continue


                for unit in content_units:
                    unit_text = unit.get("text", "")
                    unit_number = unit.get("unit_number", 0)

                    page_level_metadata = {
                        "source": filename,
                        "original_file_path": file_path_original,
                        "original_file_hash": file_hash,
                        "file_type": file_type,
                        "total_units_in_file": total_units,
                        "unit_number": unit_number,
                        "page": unit_number - 1 if file_type == "pdf" else None, # page is 0-indexed
                        "slide": unit_number if file_type == "pptx" else None, # slide is 1-indexed
                    }

                    doc = Document(page_content=unit_text, metadata=page_level_metadata)
                    docs_from_db.append(doc)
            conn.close()
        except sqlite3.Error as e:
            print(f"SQLite error while loading documents from {db_path}, table {table_name}: {e}")

        print(f"Loaded {len(docs_from_db)} pages/slides from SQLite database '{db_path}', table '{table_name}'.")
        return docs_from_db

    def index_from_sqlite(self, db_path: str, sqlite_table_name: str) -> Dict[str, Any]:
        """
        Loads documents from the SQLite database (using the provided db_path and table_name),
        processes them into chunks, and adds them to the vector store.
        """
        results: Dict[str, Any] = {"indexed_source_files_count": 0, "total_chunks_added": 0, "errors": []}
        all_chunks_to_add: List[Document] = []

        # Pass sqlite_table_name to _load_documents_from_sqlite
        page_level_docs = self._load_documents_from_sqlite(db_path, sqlite_table_name)

        if not page_level_docs:
            print(f"No documents found in SQLite database '{db_path}' (table: '{sqlite_table_name}') to index.")
            return results

        processed_source_files = set()

        # Process each page/slide Document into smaller text chunks
        for page_doc_idx, page_doc in enumerate(page_level_docs):
            # Ensure page_doc is a Document instance from the start
            if not isinstance(page_doc, Document):
                error_detail = f"Item from _load_documents_from_sqlite at index {page_doc_idx} is NOT a Document. Type: {type(page_doc)}, Value: {str(page_doc)[:200]}"
                print(f"CRITICAL ERROR (during page_doc processing): {error_detail}")
                results["errors"].append({
                    "file": "Unknown (item was not a Document during page processing)",
                    "unit": "Unknown",
                    "error": error_detail
                })
                continue # Skip this problematic item

            source_file_name = page_doc.metadata.get("source", f"unknown_source_at_idx_{page_doc_idx}")
            page_unit_number = page_doc.metadata.get("unit_number", 0)
            processed_source_files.add(source_file_name)

            try:
                page_text_content = page_doc.page_content
                # Assuming self.document_processor.text_splitter is initialized
                chunks_of_text = self.document_processor.text_splitter.split_text(page_text_content)

                for chunk_idx, text_of_chunk in enumerate(chunks_of_text):
                    # Assuming self.document_processor._generate_chunk_id is available
                    chunk_id = self.document_processor._generate_chunk_id(
                        source_file_name,
                        page_unit_number, # Using unit_number from metadata for consistency
                        chunk_idx,
                        text_of_chunk
                    )

                    chunk_metadata = page_doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": chunk_id,
                        "chunk_index_within_unit": chunk_idx,
                        "text_length_chars": len(text_of_chunk),
                    })

                    new_chunk_doc = Document(page_content=text_of_chunk, metadata=chunk_metadata)
                    all_chunks_to_add.append(new_chunk_doc) # Appending Document objects

            except Exception as e:
                error_detail = f"Failed to process unit {page_unit_number} from file '{source_file_name}': {str(e)}"
                print(error_detail)
                results["errors"].append({
                    "file": source_file_name,
                    "unit": page_unit_number,
                    "error": str(e)
                })

        # Add all created chunks to the vector store
        if all_chunks_to_add:
            valid_docs_for_filtering = []
            for i, item in enumerate(all_chunks_to_add):
                if not isinstance(item, Document):
                    error_detail = f"Item at index {i} in all_chunks_to_add (PRE-FILTER) is NOT a Document. Type: {type(item)}, Value: {str(item)[:200]}"
                    print(f"CRITICAL ERROR: {error_detail}")
                    results["errors"].append({
                        "file": "Unknown (item was not a Document pre-filter)",
                        "unit": "Unknown",
                        "error": error_detail
                    })
                    continue
                valid_docs_for_filtering.append(item)

            if not valid_docs_for_filtering:
                print("No valid Document objects found to filter and add to ChromaDB.")
                return results

            print(f"Filtering metadata for {len(valid_docs_for_filtering)} valid chunks before adding to ChromaDB...")

            docs_ready_for_chroma = []
            for doc_object in valid_docs_for_filtering: # doc_object here must be a Document
                try:
                    # Assuming filter_complex_metadata is imported
                    filtered_doc = filter_complex_metadata(doc_object) # Expects Document, returns Document
                    docs_ready_for_chroma.append(filtered_doc)
                except Exception as e_filter:
                    source_hint = doc_object.metadata.get('source', 'Unknown') # Safe as doc_object is a Document
                    print(f"Warning: filter_complex_metadata failed for document (source: {source_hint}). Error: {e_filter}")

                    current_metadata = doc_object.metadata
                    manually_filtered_meta = {k: v for k, v in current_metadata.items() if v is not None}

                    print(f"  Attempting manual None removal for metadata of (source: {source_hint})")
                    docs_ready_for_chroma.append(Document(page_content=doc_object.page_content, metadata=manually_filtered_meta))

            if docs_ready_for_chroma:
                print(f"Adding {len(docs_ready_for_chroma)} total chunks from SQLite data to vector store...")
                # Assuming self.vector_store.add_documents is available
                self.vector_store.add_documents(docs_ready_for_chroma)

                if hasattr(self.vector_store, 'persist') and callable(self.vector_store.persist):
                    self.vector_store.persist()
                    print("ChromaDB changes persisted.")

                results["total_chunks_added"] = len(docs_ready_for_chroma)
                db_count = 0
                if hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'count'):
                     db_count = self.vector_store._collection.count()
                print(f"Successfully added {len(docs_ready_for_chroma)} chunks. Vector store now has approx. {db_count} entries.")
            else:
                print("No chunks available to add to vector store after filtering attempts.")

        else:
            print("No new text chunks were generated from SQLite data to be added to the vector store.")

        results["indexed_source_files_count"] = len(processed_source_files)
        return results

# --- OllamaEnhancedRAGAgent Class ---
class OllamaEnhancedRAGAgent(EnhancedEngineeringAgent): # Inherits from the Ollama-specific agent base
    def __init__(self, ollama_model_name: str, ollama_base_url: str, ollama_embed_model:str, db_connection: Any, vector_store_path: str = None): # Modified default
        super().__init__(
            ollama_model_name=ollama_model_name,
            ollama_base_url=ollama_base_url,
            db_connection=db_connection,
            vector_store=None
        )

        self.ollama_embeddings = OllamaEmbeddings(
            model=ollama_embed_model,
            base_url=ollama_base_url
        )

        _vector_store_path = vector_store_path if vector_store_path is not None else DEFAULT_VECTOR_STORE_PATH # Use loaded default
        self.citation_rag = CitationRAG(
            vector_store_path=_vector_store_path,
            embedding_model=self.ollama_embeddings
        )

        self.vector_store = self.citation_rag.vector_store
        print(f"OllamaEnhancedRAGAgent initialized. Using LLM '{ollama_model_name}' and Embeddings '{ollama_embed_model}'. Vector store: '{_vector_store_path}'")

    # Override the search_documents method from the base agent
    def search_documents(self, query: str, part_numbers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Override to use the citation-aware RAG search from CitationRAG,
        which is configured with Ollama embeddings.
        """
        # The 'part_numbers' argument is available if the agent framework passes it.
        # The tool definition in agent_ollama.py should be designed to accept this if needed,
        # or the agent's LLM prompted to include part context in the query string.
        return self.citation_rag.search_with_citations(
            query=query,
            k=3, # Default number of documents to retrieve for the agent
            part_numbers=part_numbers # Pass along for potential use in search_with_citations
        )
