import os
import asyncio
from typing import Optional, List, Dict, Any, Optional 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import json
import requests
from contextlib import asynccontextmanager

import random
from datetime import datetime, timedelta

# Import for StaticFiles
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse # Optional: if you want a specific root path for index.html


# Local imports
from rag_system import OllamaEnhancedRAGAgent # The RAG Agent designed for Ollama
from agent_ollama import QueryType
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Configuration from Environment Variables ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:4b")
OLLAMA_EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL_NAME", "nomic-embed-text")
SQLITE_DB_FILE_PATH = os.getenv("SQLITE_DB_FILE_PATH", "engineering_docs.sqlite")
SQLITE_TABLE_NAME = os.getenv("SQLITE_TABLE_NAME", "processed_documents")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./ollama_chroma_db")
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8000))

# --- Pydantic Models ---  
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    row: Optional[int] = None
    score: float
    excerpt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class HierarchyRequest(BaseModel):
    part_numbers: List[str]

class HierarchyNode(BaseModel):
    id: str
    name: str
    value: Optional[float] = 1.0
    children: List['HierarchyNode'] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

class SunburstDataResponse(BaseModel):
    ids: List[str]
    labels: List[str]
    parents: List[str]
    values: List[float]

class PartsAwaitingMaintenanceData(BaseModel):
    part_name: str
    dates: List[str]
    awp_counts: List[int] = Field(default_factory=list) # Awaiting Parts
    awm_counts: List[int] = Field(default_factory=list) # Awaiting Maintenance
    awpm_counts: List[int] = Field(default_factory=list) # Awaiting Parts & Maintenance

class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessage
    citations: List[Citation] = Field(default_factory=list)
    query_classification: Dict[str, Any] = Field(default_factory=dict)
    part_info: Optional[Dict[str, Any]] = None
    maintenance_data: Optional[PartsAwaitingMaintenanceData] = None

# Update forward references
HierarchyNode.model_rebuild()

class HierarchyResponse(BaseModel):
    hierarchy: HierarchyNode
    parts_info: Dict[str, Any]

# --- Global Agent & Ollama Config ---
agent: Optional[OllamaEnhancedRAGAgent] = None
conversations: Dict[str, List[Dict[str, Any]]] = {}


def build_hierarchy_tree(parts_data: List[Dict[str, Any]]) -> HierarchyNode:
    """Build a complete hierarchy tree from multiple part data."""
    root = HierarchyNode(id="1", name="System", value=100)
    nodes_map = {"1": root}

    # Sort parts by hierarchical ID to ensure parents are created before children
    sorted_parts = sorted(parts_data, key=lambda x: len(x.get('hierarchical_id', '').split('.')))

    for part in sorted_parts:
        hier_id = part.get('hierarchical_id', '')
        if not hier_id or hier_id == "1":
            continue

        parts = hier_id.split('.')

        # Build path to this part
        for i in range(1, len(parts)):
            current_id = '.'.join(parts[:i+1])
            parent_id = '.'.join(parts[:i])

            if current_id not in nodes_map:
                # Determine node name
                if current_id == hier_id:
                    # This is the actual part
                    node_name = f"Part {part.get('part_number', current_id)}"
                else:
                    # This is an intermediate level
                    level = len(current_id.split('.'))
                    level_names = ["System", "Subsystem", "Component", "Subcomponent"]
                    node_name = f"{level_names[min(level-1, len(level_names)-1)]} {current_id}"

                # Create new node
                new_node = HierarchyNode(
                    id=current_id,
                    name=node_name,
                    value=10 if current_id == hier_id else 5,
                    metadata=part if current_id == hier_id else None
                )
                nodes_map[current_id] = new_node

                # Add to parent
                if parent_id in nodes_map:
                    nodes_map[parent_id].children.append(new_node)

    return root

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global agent
    print("🚀 Starting Ollama Engineering Assistant API...")
    if not check_ollama_running():
        print(f"🛑 Ollama service is not running or not reachable at {OLLAMA_BASE_URL}. Please start Ollama (e.g., `ollama serve`) and restart this API.")
    else:
        print(f"Ollama service detected at {OLLAMA_BASE_URL}.")

        required_models = {OLLAMA_MODEL_NAME, OLLAMA_EMBED_MODEL_NAME}
        all_models_ready = True
        for model_name_to_check in required_models:
            print(f"Checking for Ollama model: {model_name_to_check}...")
            if not check_model_available(model_name_to_check):
                print(f"Model '{model_name_to_check}' not found locally at {OLLAMA_BASE_URL}.")
                if not pull_model(model_name_to_check):
                    print(f"Failed to pull model '{model_name_to_check}'. Agent may not function correctly.")
                    all_models_ready = False
            else:
                print(f"Model '{model_name_to_check}' is available.")

        if all_models_ready:
            try:
                agent = OllamaEnhancedRAGAgent(
                    ollama_model_name=OLLAMA_MODEL_NAME,
                    ollama_base_url=OLLAMA_BASE_URL,
                    ollama_embed_model=OLLAMA_EMBED_MODEL_NAME,
                    db_connection=None, # db_connection for the agent itself, not the RAG SQLite
                    vector_store_path=VECTOR_STORE_PATH # Use variable from .env
                )
                print("✅ Ollama RAG Agent initialized successfully.")

                if agent and hasattr(agent, 'citation_rag'):
                    db_file = Path(SQLITE_DB_FILE_PATH) # Use variable
                    if db_file.exists():
                        print(f"Attempting to index documents from SQLite DB: {SQLITE_DB_FILE_PATH}")
                        # Pass the table name from the constant defined in main.py
                        results = agent.citation_rag.index_from_sqlite(SQLITE_DB_FILE_PATH, SQLITE_TABLE_NAME) # Use variables
                        print(f"SQLite document indexing results: {results}")
                    else:
                        print(f"SQLite database file not found at {SQLITE_DB_FILE_PATH}. Skipping indexing at startup.") # Use variable
                        print("Please run the 'document_extractor_sqlite.py' script first to populate the database.")

            except Exception as e:
                print(f"❌ Error during Ollama agent initialization: {e}")
                import traceback
                traceback.print_exc()
                agent = None
        else:
            print("Agent not initialized due to missing models.")
            agent = None

    yield
    print("🔌 Shutting down Ollama Engineering Assistant API...")

# --- Ollama Utility Functions ---
def check_ollama_running():
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False

def check_model_available(model_name: str):
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        res.raise_for_status()
        models = res.json().get("models", [])
        return any(
            m.get("name") == model_name or 
            (m.get("name") and m.get("name").startswith(model_name.split(":")[0]))
            for m in models
        )
    except requests.RequestException:
        return False

def pull_model(model_name: str):
    print(f"Pulling Ollama model '{model_name}' from {OLLAMA_BASE_URL}... This can take a while.") # Uses variable
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name}, stream=True) # Uses variable
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "total" in data and "completed" in data:
                    progress = (data["completed"] / data["total"]) * 100
                    print(f"  {status} - {progress:.2f}% complete", end='\r')
                else:
                    print(f"  {status}")
        print(f"\nModel '{model_name}' pull completed or already present.")
        return True
    except requests.RequestException as e:
        print(f"Error pulling model '{model_name}': {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding pull status for '{model_name}': {e}")
        return False


# --- FastAPI App Initialization ---
app = FastAPI(title="Ollama Engineering Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- FastAPI Endpoints ---
@app.get("/")
async def root_ollama():
    return {
        "message": "Ollama Engineering Assistant API",
        "status": "Agent Initialized" if agent else "Agent NOT Initialized",
        "ollama_service": "Running" if check_ollama_running() else "NOT Running (❗)",
        "ollama_llm_model": OLLAMA_MODEL_NAME,
        "ollama_embedding_model": OLLAMA_EMBED_MODEL_NAME,
        "endpoints": {
            "chat": "/api/chat (POST)",
            "index_docs": "/api/index-documents (POST)", # Assuming you might re-add or have this
            "reindex_from_db": "/api/reindex-from-db (POST)",
            "hierarchy": "/api/hierarchy (POST)",
            "sunburst_data": "/api/sunburst-data (GET)",
            "parts_awaiting_maintenance": "/api/parts-awaiting-maintenance (GET)" # Added new endpoint
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint_ollama(request: ChatRequest):
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Ollama agent not initialized.")
    
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        agent_res = await agent.process_query(request.message)
        assistant_msg = ChatMessage(
            role="assistant",
            content=agent_res.get("answer", "Error processing query.")
        )
        citations_fmt = [Citation(**c) for c in agent_res.get("citations", [])]
        
        # Store conversation
        if conv_id not in conversations:
            conversations[conv_id] = []
        
        conversations[conv_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        conversations[conv_id].append({
            "role": "assistant",
            "content": assistant_msg.content,
            "timestamp": assistant_msg.timestamp.isoformat()
        })
        
        return ChatResponse(
            conversation_id=conv_id,
            message=assistant_msg,
            citations=citations_fmt,
            query_classification=agent_res.get("classification", {}),
            part_info=agent_res.get("part_info"),
            maintenance_data=agent_res.get("maintenance_data")
        )
    except Exception as e:
        print(f"Error in Ollama chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hierarchy", response_model=HierarchyResponse)
async def get_hierarchy_endpoint(request: HierarchyRequest):
    """Get hierarchical information for multiple parts and build a tree structure."""
    global agent
    if not agent:
        raise HTTPException(status_code=503, detail="Ollama agent not initialized.")
    
    if not request.part_numbers:
        raise HTTPException(status_code=400, detail="No part numbers provided.")
    
    try:
        parts_info = {}
        parts_data = []
        
        # Get hierarchy info for each part
        for part_number in request.part_numbers:
            part_info = agent.get_part_hierarchy(part_number)
            parts_info[part_number] = part_info
            parts_data.append(part_info)
        
        # Build complete hierarchy tree
        hierarchy_tree = build_hierarchy_tree(parts_data)
        
        return HierarchyResponse(
            hierarchy=hierarchy_tree,
            parts_info=parts_info
        )
    except Exception as e:
        print(f"Error getting hierarchy: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))   

##@app.post("/api/index-documents")
##async def index_documents_ollama(file_paths: List[str]):
##    # (Logic similar to previous version, uses agent.citation_rag.index_documents)
##    global agent
##    if not agent or not hasattr(agent, 'citation_rag'):
##        raise HTTPException(status_code=503, detail="Ollama RAG agent not initialized.")
##    if not file_paths: raise HTTPException(status_code=400, detail="No file paths.")
##    try:
##        results = agent.citation_rag.index_documents(file_paths)
##        return {"message": "Ollama document indexing initiated.", "details": results}
##    except Exception as e:
##        print(f"Error indexing documents for Ollama: {e}")
##        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex-from-db")
async def reindex_from_db_endpoint():
    global agent
    if not agent or not hasattr(agent, 'citation_rag'):
        raise HTTPException(status_code=503, detail="Ollama RAG agent not initialized.")
    try:
        print("API call to re-index from SQLite DB received.")
        db_file = Path(SQLITE_DB_FILE_PATH)
        if not db_file.exists():
            raise HTTPException(status_code=404, detail=f"SQLite DB file not found: {SQLITE_DB_FILE_PATH}")
        
        results = agent.citation_rag.index_from_sqlite(SQLITE_DB_FILE_PATH, SQLITE_TABLE_NAME)
        return {"message": "Ollama document re-indexing from SQLite initiated.", "details": results}
    except Exception as e:
        print(f"Error re-indexing documents from SQLite via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sunburst-data", response_model=SunburstDataResponse)
async def get_sunburst_data_mockup():
    """
    Provides mockup data structured for a Plotly.js Sunburst chart.
    Parent nodes now have values summed from their children.
    """
    # Children values:
    # Component A1: 10.0
    # Component A2: 15.0
    # Component B1: 20.0
    # Component B2: 5.0
    #
    # Parent sums:
    # Subsystem A (A1+A2): 10 + 15 = 25.0
    # Subsystem B (B1+B2): 20 + 5 = 25.0
    # System (SubA + SubB): 25 + 25 = 50.0
    return SunburstDataResponse(
        ids=[
            "System", "Subsystem A", "Subsystem B",
            "Component A1", "Component A2", "Component B1", "Component B2"
        ],
        labels=[
            "System Root", "Subsystem A", "Subsystem B",
            "Component A1 (SA)", "Component A2 (SA)", "Component B1 (SB)", "Component B2 (SB)"
        ],
        parents=[
            "", "System", "System",
            "Subsystem A", "Subsystem A", "Subsystem B", "Subsystem B"
        ],
        values=[
            50.0,  # System total
            25.0,  # Subsystem A total
            25.0,  # Subsystem B total
            10.0,
            15.0,
            20.0,
            5.0
        ]
    )

@app.get("/api/parts-awaiting-maintenance", response_model=PartsAwaitingMaintenanceData)
async def get_parts_awaiting_maintenance(part_name: str):
    """
    Provides dummy data for parts awaiting maintenance for a given part name.
    Categories: AWP (Awaiting Parts), AWM (Awaiting Maintenance), AWP&M (Awaiting Parts & Maintenance).
    """
    if not part_name:
        raise HTTPException(status_code=400, detail="Part name cannot be empty.")

    # Generate dummy data for the last 10 days
    dates = []
    awp_counts = []
    awm_counts = []
    awpm_counts = []
    
    end_date = datetime.now()
    for i in range(10): # 10 data points
        current_date = end_date - timedelta(days=i)
        dates.append(current_date.strftime("%Y-%m-%d"))
        awp_counts.append(random.randint(0, 15))
        awm_counts.append(random.randint(0, 10))
        awpm_counts.append(random.randint(0, 5))
        
    # Data is generated backwards, so reverse for chronological order
    dates.reverse()
    awp_counts.reverse()
    awm_counts.reverse()
    awpm_counts.reverse()
    
    return PartsAwaitingMaintenanceData(
        part_name=part_name,
        dates=dates,
        awp_counts=awp_counts,
        awm_counts=awm_counts,
        awpm_counts=awpm_counts
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="localhost", port=8000)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
