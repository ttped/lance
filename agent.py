from typing import List, Dict, Any, Optional, Tuple
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from pydantic import BaseModel, Field
import re
from enum import Enum

class QueryType(Enum):
    PART_QUERY = "part_query"
    DOCUMENT_SEARCH = "document_search"
    GENERAL_QUESTION = "general_question"
    HYBRID_QUERY = "hybrid_query"  # Both part and document search

class QueryClassification(BaseModel):
    query_type: QueryType
    part_identifiers: List[str] = Field(default_factory=list)
    requires_rag: bool = False
    search_keywords: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class EnhancedEngineeringAgent:
    def __init__(self, llm: ChatOpenAI, db_connection: Any, vector_store: Any):
        self.llm = llm
        self.db = db_connection
        self.vector_store = vector_store
        self.classification_llm = llm.with_structured_output(QueryClassification)
        
    def classify_query(self, query: str) -> QueryClassification:
        """Classify the type of query and extract relevant information."""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for an engineering knowledge system.
            Analyze the user's query and determine:
            1. Query type:
               - part_query: Questions about specific parts/components (includes part numbers, hierarchies)
               - document_search: Questions requiring document search without specific part references
               - general_question: General engineering questions not requiring specific data
               - hybrid_query: Questions about parts that also need document search
            
            2. Extract any part numbers/names mentioned (look for patterns like "part-123", "component XYZ", etc.)
            3. Determine if RAG document search is needed
            4. Extract key search terms for document retrieval
            5. Provide confidence score (0-1)
            
            Examples:
            - "What is the hierarchy for part-123?" -> part_query
            - "Find all documentation about thermal analysis" -> document_search
            - "What is the difference between steel and aluminum?" -> general_question
            - "Show me the specs for part-456 and any related test reports" -> hybrid_query
            """),
            ("human", "{query}")
        ])
        
        return self.classification_llm.invoke(
            classification_prompt.format_messages(query=query)
        )
    
    def extract_part_numbers(self, text: str) -> List[str]:
        """Extract part numbers from text using regex patterns."""
        patterns = [
            r'part[-\s]?(\d+)',
            r'component[-\s]?([A-Z0-9]+)',
            r'p/n[-\s]?([A-Z0-9-]+)',
            r'part[-\s]?number[-\s]?([A-Z0-9-]+)',
        ]
        
        part_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            part_numbers.extend(matches)
        
        return list(set(part_numbers))  # Remove duplicates
    
    def get_part_hierarchy(self, part_number: str) -> Dict[str, Any]:
        """Retrieve part hierarchy from database."""
        # Implement your database query here
        # This is a placeholder
        return {
            "part_number": part_number,
            "hierarchy": ["system", "subsystem", "component", part_number],
            "metadata": {}
        }
    
    def search_documents(self, query: str, part_numbers: List[str] = None) -> List[Dict[str, Any]]:
        """Search documents using RAG with optional part number filtering."""
        # Construct enhanced query with part numbers if available
        enhanced_query = query
        if part_numbers:
            enhanced_query = f"{query} {' '.join([f'part:{pn}' for pn in part_numbers])}"
        
        # Search vector store
        results = self.vector_store.similarity_search_with_score(
            enhanced_query,
            k=5  # Top 5 documents
        )
        
        # Format results with metadata
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "chunk_id": doc.metadata.get("chunk_id", None)
            })
        
        return formatted_results
    
    def create_tools(self) -> List[Tool]:
        """Create tools for the agent."""
        return [
            Tool(
                name="get_part_hierarchy",
                func=self.get_part_hierarchy,
                description="Get the hierarchy and details for a specific part number"
            ),
            Tool(
                name="search_documents",
                func=self.search_documents,
                description="Search engineering documents using semantic search. Can optionally filter by part numbers."
            ),
            Tool(
                name="extract_part_numbers",
                func=self.extract_part_numbers,
                description="Extract part numbers from user query text"
            )
        ]
    
    def create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with dynamic tool selection."""
        tools = self.create_tools()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an engineering knowledge assistant. 
            Based on the query classification, use the appropriate tools:
            - For part-specific queries: Use get_part_hierarchy
            - For document searches: Use search_documents
            - For hybrid queries: Use both tools
            - For general questions: Answer directly without tools
            
            When using search_documents, always cite your sources with specific document names and pages.
            Format citations as [Source: filename, Page: X] when available.
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self.llm.bind_functions(tools)
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return response with citations."""
        # Classify the query
        classification = self.classify_query(query)
        
        # Initialize response structure
        response = {
            "query": query,
            "classification": classification.dict(),
            "answer": "",
            "citations": [],
            "part_info": None,
            "documents": []
        }
        
        # Handle based on query type
        if classification.query_type == QueryType.GENERAL_QUESTION:
            # Direct LLM response without tools
            direct_response = self.llm.invoke([HumanMessage(content=query)])
            response["answer"] = direct_response.content
            
        else:
            # Use agent for other query types
            agent_executor = self.create_agent_executor()
            result = agent_executor.invoke({"input": query})
            
            response["answer"] = result["output"]
            
            # Extract citations from intermediate steps
            for step in result.get("intermediate_steps", []):
                if step[0].tool == "search_documents":
                    documents = step[1]
                    for doc in documents:
                        citation = {
                            "source": doc["source"],
                            "page": doc.get("page"),
                            "score": doc["score"],
                            "excerpt": doc["content"][:200] + "...",
                            "metadata": doc["metadata"]
                        }
                        response["citations"].append(citation)
                        response["documents"].append(doc)
                
                elif step[0].tool == "get_part_hierarchy":
                    response["part_info"] = step[1]
        
        return response
