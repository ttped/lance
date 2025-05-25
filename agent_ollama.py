# agent_ollama.py

from typing import List, Dict, Any, Optional, Tuple, Union, Set
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from pydantic import BaseModel, Field
from enum import Enum
import re
from datetime import datetime
import json
import random
from datetime import datetime, timedelta

# Ollama specific imports
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import AgentOutputParser
from langchain.tools.render import render_text_description
from langchain_core.messages import HumanMessage


class CustomOllamaFunctionsOutputParser(AgentOutputParser):
    """
    Parses the output of an LLM. Expects JSON for tool usage 
    (with 'tool' and 'tool_input' keys), or a JSON object with a 'response' key
    for a final answer, or plain text for a final answer.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            response_json = json.loads(text) # text is the raw output string from the LLM
            if isinstance(response_json, dict):
                # Check for tool call
                if "tool" in response_json and "tool_input" in response_json:
                    tool = response_json["tool"]
                    tool_input = response_json["tool_input"]
                    return AgentAction(tool=tool, tool_input=tool_input, log=text)
                # Check for LLM wrapping its answer in common keys
                elif "response" in response_json and isinstance(response_json["response"], str):
                    return AgentFinish(return_values={"output": response_json["response"]}, log=text)
                elif "answer" in response_json and isinstance(response_json["answer"], str): # ADDED THIS
                    return AgentFinish(return_values={"output": response_json["answer"]}, log=text)
                elif not response_json: # Handles empty JSON object {}
                    final_answer = "I was unable to find specific information or complete the request."
                    return AgentFinish(return_values={"output": final_answer}, log=text)
                else:
                    # Unexpected JSON dictionary structure
                    print(f"Warning: Received unexpected JSON dictionary structure from LLM: {text}")
                    # Fallback: return the original text, which is a JSON string.
                    # The process_query method will have a final cleanup for this.
                    return AgentFinish(return_values={"output": text}, log=text)
            else:
                # Valid JSON, but not a dictionary (e.g., a JSON-encoded string, list, number)
                print(f"Warning: Received non-dictionary JSON from LLM (could be JSON-encoded string): {text}")
                # Pass it through; process_query will attempt to clean it if it's a string.
                return AgentFinish(return_values={"output": text}, log=text)

        except json.JSONDecodeError:
            # Not JSON, assume it's a direct plain text final answer.
            final_answer = text
            if not text.strip():
                final_answer = "I received an empty response. Could you please clarify your request?"
            return AgentFinish(return_values={"output": final_answer}, log=text)

    @property
    def _type(self) -> str:
        return "custom_ollama_functions_output_parser"


class QueryType(Enum):
    PART_QUERY = "part_query"
    DOCUMENT_SEARCH = "document_search"
    GENERAL_QUESTION = "general_question"
    HYBRID_QUERY = "hybrid_query"

class QueryClassification(BaseModel):
    query_type: QueryType
    part_identifiers: List[str] = Field(default_factory=list)
    requires_rag: bool = False
    search_keywords: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PartMaintenanceInput(BaseModel):
    part_name: str = Field(description="The name or identifier of the part to check maintenance status for.")


class EnhancedEngineeringAgent:
    def __init__(self, ollama_model_name: str, ollama_base_url: str, db_connection: Any, vector_store: Optional[Any] = None):
        # Base LLM for tasks like classification
        # Using format="json" to hint to Ollama to output JSON for structured output tasks.
        # Reliability depends on the specific Ollama model (e.g., gemma3:4b).
        self.llm = ChatOllama(
            model=ollama_model_name,
            base_url=ollama_base_url,
            format="json",
            temperature=0.0 # Classification should be deterministic
        )
        
        # LLM specifically wrapped for function/tool calling.
        # OllamaFunctions tries to make the Ollama model behave like OpenAI functions.
        self.tool_llm = ChatOllama(
            model=ollama_model_name,
            base_url=ollama_base_url,
            format="json", # OllamaFunctions expects the LLM to output JSON for tool calls
            temperature=0.0 # Tool usage should be deterministic
        )

        self.db = db_connection # Mocked for this example
        self.vector_store = vector_store # This will be set by the RAG-enhanced agent version

        # For classification, using the base LLM with with_structured_output.
        # The Pydantic model QueryClassification guides the output.
        # The prompt must strongly guide the LLM to produce the correct JSON structure.
        self.classification_llm = self.llm.with_structured_output(
            QueryClassification,
            #method="json_mode" # causes a error
        )

    def get_parts_awaiting_maintenance_data(self, part_name: str) -> Dict[str, Any]:
        """
        Generates dummy data for parts awaiting maintenance for a given part name.
        This function is called by the 'get_parts_awaiting_maintenance' tool.
        Input can be a string (part_name) or a dict {'part_name': 'XYZ'}.
        """
        actual_part_name = part_name
        if isinstance(part_name, dict) and "part_name" in part_name:
            actual_part_name = part_name["part_name"]
        elif not isinstance(part_name, str):
            # Fallback if input is not as expected
            print(f"Warning: Unexpected input type for part_name in get_parts_awaiting_maintenance_data: {type(part_name)}. Using 'UnknownPart'.")
            actual_part_name = "UnknownPart"

        print(f"Tool 'get_parts_awaiting_maintenance_data' called for part: {actual_part_name}")
        
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
            
        dates.reverse()
        awp_counts.reverse()
        awm_counts.reverse()
        awpm_counts.reverse()
        
        return {
            "part_name": actual_part_name,
            "dates": dates,
            "awp_counts": awp_counts,
            "awm_counts": awm_counts,
            "awpm_counts": awpm_counts
        }
        
    def classify_query(self, query: str) -> QueryClassification:
        """Classify the type of query and extract relevant information using an Ollama model."""
        
        # This prompt is crucial for guiding gemma3:4b (or other Ollama models)
        # to produce the correct JSON structure for the QueryClassification Pydantic model.
        # It explicitly asks for a JSON object and provides examples of the expected structure.
        classification_prompt_template = """
You are an expert query classifier for an engineering knowledge system. Your task is to analyze the user's query
and respond *only* with a valid JSON object that strictly adheres to the following structure:
{{
  "query_type": "string (must be one of: 'part_query', 'document_search', 'general_question', 'hybrid_query')",
  "part_identifiers": ["list of strings (extracted part numbers or names, e.g., '123', 'XYZ-001', '1.1.2.3')"],
  "requires_rag": "boolean (true if document search is needed for an engineering context, false otherwise for general/philosophical questions or if the question can be answered from common knowledge)",
  "search_keywords": ["list of strings (key terms for document retrieval if requires_rag is true, or general keywords from the query if false)"],
  "confidence": "float (a score between 0.0 and 1.0 indicating your confidence in the classification)"
}}

Analyze the following user query:
User Query: "{query}"

Important: Look for hierarchical part numbers like 1.1.2.3, 1.2.4, etc. These should be included in part_identifiers.
Set "requires_rag" to false for questions that are clearly non-engineering, philosophical, common sense, or common knowledge (e.g., "what is love?", "what is the capital of France?"). Only set "requires_rag" to true if the question implies seeking information from specific engineering documents.

Examples of mapping queries to JSON:
1. Query: "What is the hierarchy for part-123?"
   JSON: {{"query_type": "part_query", "part_identifiers": ["123"], "requires_rag": false, "search_keywords": ["hierarchy", "part-123"], "confidence": 0.95}}
2. Query: "Show me part 1.1.2.3"
   JSON: {{"query_type": "part_query", "part_identifiers": ["1.1.2.3"], "requires_rag": false, "search_keywords": ["part 1.1.2.3"], "confidence": 0.95}}
3. Query: "Find all documentation about thermal analysis for component ABC"
   JSON: {{"query_type": "hybrid_query", "part_identifiers": ["ABC"], "requires_rag": true, "search_keywords": ["thermal analysis", "component ABC", "documentation"], "confidence": 0.9}}
4. Query: "What is the difference between steel and aluminum?"
   JSON: {{"query_type": "general_question", "part_identifiers": [], "requires_rag": false, "search_keywords": ["steel", "aluminum", "difference"], "confidence": 0.85}}
5. Query: "What is love?"
   JSON: {{"query_type": "general_question", "part_identifiers": [], "requires_rag": false, "search_keywords": ["love"], "confidence": 0.98}}
6. Query: "Tell me about the reliability of the X12 pump and search for maintenance guides."
   JSON: {{"query_type": "hybrid_query", "part_identifiers": ["X12 pump"], "requires_rag": true, "search_keywords": ["reliability", "X12 pump", "maintenance guides"], "confidence": 0.92}}

Your JSON Response (ONLY the JSON object):"""
        
        prompt = PromptTemplate.from_template(classification_prompt_template)
        
        # It expects the LLM (self.llm which has format="json") to output JSON that Pydantic can parse into QueryClassification.
        chain = prompt | self.llm 
        try:
            response_message = chain.invoke({"query": query})
            json_string_output = response_message.content
            
            # Extract JSON from potential markdown blocks
            match = re.search(r"```json\s*(\{.*?\})\s*```", json_string_output, re.DOTALL)
            if match:
                json_string_output = match.group(1)
            else:
                # Try to find JSON object within the string
                if not (json_string_output.strip().startswith("{") and json_string_output.strip().endswith("}")):
                     match_curly = re.search(r"(\{.*?\})", json_string_output, re.DOTALL)
                     if match_curly:
                         json_string_output = match_curly.group(1)

            parsed_json = json.loads(json_string_output)
            return QueryClassification(**parsed_json)
        except Exception as e:
            print(f"Error during query classification: {e}")
            # Return a default classification
            return QueryClassification(
                query_type=QueryType.GENERAL_QUESTION,
                part_identifiers=[],
                requires_rag=False,
                search_keywords=[],
                confidence=0.5
            )
    
    def extract_part_numbers(self, text: str) -> List[str]:
        """Extract part numbers from text using regex patterns."""
        patterns = [
            r'part[-\s]?(\d+)',             # Example: part-123, part 123
            r'component[-\s]?([A-Z0-9]+)',  # Example: component-XYZ, component XYZ
            r'p/n[-\s]?([A-Z0-9-]+)',       # Example: p/n ABC-123
            r'part number[-\s]?([A-Z0-9-]+)',# Example: part number ABC-123
            r'\b(\d+(?:\.\d+)+)\b',         # Hierarchical numbers: 1.1.2.3, 1.2.4
        ]
        part_numbers: List[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            part_numbers.extend(matches)
        return list(set(part_numbers))

    def get_part_hierarchy(self, part_number: str) -> Dict[str, Any]:
        """Retrieve part hierarchy from database (mocked for Ollama)."""
        print(f"Tool 'get_part_hierarchy' called with: {part_number}")
        
        # Generate mock hierarchy data including hierarchical numbering
        if '.' in part_number:
            # It's already a hierarchical number
            hierarchy_parts = part_number.split('.')
            hierarchy = []
            current = ""
            for i, part in enumerate(hierarchy_parts):
                current = current + ("." if current else "") + part
                level_names = ["System", "Subsystem", "Component", "Subcomponent", "Part"]
                level_name = level_names[min(i, len(level_names)-1)]
                hierarchy.append(f"{level_name}-{current}")
        else:
            # Traditional part number
            hierarchy = ["System-1", "Subsystem-1.1", f"Component-{part_number}"]
        
        return {
            "part_number": part_number,
            "hierarchy": hierarchy,
            "hierarchical_id": part_number if '.' in part_number else f"1.1.{part_number}",
            "status": "Active",
            "description": f"Mock description for part {part_number}",
            "metadata": {
                "source": "MockDB-Ollama", 
                "retrieved_at": datetime.utcnow().isoformat(),
                "parent": hierarchy[-2] if len(hierarchy) > 1 else None,
                "children_count": 3  # Mock number of children
            }
        }
    
    def search_documents(self, query: str, part_numbers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search documents using RAG. This method is a placeholder in the base agent.
        It will be overridden by OllamaEnhancedRAGAgent to use the actual RAG system.
        """
        print(f"Tool 'search_documents' called with query: '{query}'" + (f" and parts: {part_numbers}" if part_numbers else ""))
        if not self.vector_store:
            return [{"content": "Vector store not configured in base agent.", "metadata": {}, "score": 0.0, "source": "SystemInternal"}]
        
        # This is a fallback; the RAG agent should override this.
        # For demonstration, assume vector_store has similarity_search_with_score
        try:
            # This is a simplified call, actual RAG will be more sophisticated
            results = self.vector_store.similarity_search_with_score(query, k=2)
            return [{"content": doc.page_content, "metadata": doc.metadata, "score": score, "source": doc.metadata.get("source","Unknown")} for doc, score in results]
        except Exception as e:
            return [{"content": f"Error in placeholder search_documents: {str(e)}", "metadata": {}, "score": 0.0, "source": "SystemInternal"}]


    def create_tools(self) -> List[Tool]:
        """Create tools for the agent."""
        return [
            Tool(
                name="get_part_hierarchy",
                func=self.get_part_hierarchy,
                description="Use this tool to get the hierarchy and details for a specific part number. The input should be a single string representing the part number."
            ),
            Tool(
                name="search_documents",
                func=self.search_documents,
                description="Use this tool to search engineering documents using a natural language query. The input should be the search query string. You can include part numbers or concepts in your query."
            ),
            Tool(
                name="get_parts_awaiting_maintenance",
                func=self.get_parts_awaiting_maintenance_data,
                description="Use this tool to get data on parts awaiting maintenance for a specific part name. The input should be the part name as a string, for example 'Pump-X12' or 'Sensor-Unit-A'. This tool returns data suitable for plotting a time-series chart.",
                # args_schema=PartMaintenanceInput # Optional: if you want Pydantic validation for tool input
            ),
        ]

    def _extract_tool_history(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> Dict[str, Set[str]]:
        """Extract history of tool calls to prevent loops."""
        tool_history = {}
        for action, _ in intermediate_steps:
            tool_name = action.tool
            if tool_name not in tool_history:
                tool_history[tool_name] = set()
            tool_history[tool_name].add(str(action.tool_input))
        return tool_history
    
    def create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor for Ollama, using OllamaFunctions."""
        tools = self.create_tools()

        # Generate tools_string and tool_names
        tools_string = render_text_description(tools)
        tool_names_list = [tool.name for tool in tools]
        tool_names_str = ", ".join(tool_names_list)
        
        # Prompt for OllamaFunctions. This prompt structure is important for guiding the LLM
        agent_prompt_template = """You are a helpful and methodical engineering assistant.
Your primary goal is to answer the User's Query: {input}

You have access to the following tools:
{tools_string}

**Critical Instructions:**

1.  **TOOL USAGE SCOPE**:
    * If the User's Query clearly pertains to specific engineering topics (like components, materials, part numbers, technical processes, specifications) AND the query classification suggests a document search is appropriate, STRONGLY prefer using the `search_documents` tool.
    * Use `get_part_hierarchy` for specific part number inquiries.
    * If the query is about maintenance status, "parts awaiting maintenance", "maintenance backlog", or similar for a specific part, use `get_parts_awaiting_maintenance`. The tool input for `get_parts_awaiting_maintenance` should be the part name (e.g., "PartX" or "Compressor-123").
    * For clearly non-engineering, philosophical, or common knowledge questions (e.g., "what is love?", "what's the weather like?"), AVOID using engineering-specific tools like `search_documents` or `get_part_hierarchy`. Answer these from general knowledge unless the query explicitly asks for an engineering perspective on that topic.

2.  **AVOID TOOL LOOPS**: 
    * If you've already called a tool with the *exact same input* for the current query, DO NOT call it again unless the previous attempt failed or provided no information.
    * If a tool has provided relevant information, use that information to answer the user.
    * Maximum of 1 call to `get_parts_awaiting_maintenance` per specific part name in a single user query.
    * Maximum of 2 calls total for other unique tools for the current query if absolutely necessary.

3.  **WHEN TO USE A TOOL (TOOL CALL JSON)**: 
    If, based on Instruction 1 and the Decision Process, you determine a tool call is necessary to gather information, respond ONLY with a single valid JSON object strictly matching this format:
    ```json
    {{
      "tool": "string (must be one of: [{tool_names}])",
      "tool_input": "string (the input for the tool. This input MUST be directly and accurately derived from the User's Query. For `search_documents`, use key phrases from the query. Do NOT invent unrelated engineering terms if the query is non-engineering.)"
    }}
    ```

4.  **WHEN TO RESPOND DIRECTLY (FINAL ANSWER JSON)**:
    If you have sufficient information (from previous tool use, as seen in the Scratchpad's "Observation:", or general knowledge) to answer the User's Query directly,
    respond ONLY with a single valid JSON object with a single key "answer":
    ```json
    {{
      "answer": "Your comprehensive, plain text answer here, synthesizing all relevant information. If your answer uses information retrieved from documents (details of which would be in an 'Observation:' from a 'search_documents' tool call), you MUST explicitly state the source document names in parentheses after the relevant piece of information, e.g., 'The device operates at 50Hz (Source: device_manual.pdf).' or 'Key safety measures include X, Y, and Z (Source: safety_guide.txt).' If multiple documents support a statement, you can list them, e.g., (Source: report_A.docx, data_sheet.csv). IMPORTANT: If a tool was used but its output (Observation) is clearly irrelevant to the original User's Query, prioritize answering the User's Query using your general knowledge. In such cases, you should OMIT the irrelevant tool findings from your answer or, at most, very briefly state that a document search did not yield relevant information to the specific query, then proceed with the general knowledge answer."
    }}
    ```

5. **HANDLING INSUFFICIENT DOCUMENTATION**:
    * If `search_documents` returns documents with very low relevance scores (e.g., below 0.5) or
      if the content of retrieved documents is extremely sparse (e.g., just a title, a part number without description),
      your answer MUST reflect this. State that "The documents provided limited/minimal information regarding X..."
    * Do NOT attempt to elaborate or expand on such minimal information using your general knowledge.
    * It is better to state that the information is not available in the documents than to provide potentially misleading information.
    
    Do NOT use plain text directly for final answers if you are not calling a tool. Always wrap your final textual answer in the JSON structure specified above.

**Tool Usage History:**
{tool_history}

**Decision Process:**
1.  Examine the User's Query: "{input}" and the Scratchpad: "{agent_scratchpad}".
2.  Does the Scratchpad contain an "Observation:" with information from a recent tool call?
    * If YES:
        a.  Critically evaluate if the Observation's content is relevant and helpful for answering the original User's Query.
        b.  If the Observation is from `get_part_hierarchy` and contains part details: This information IS considered sufficient for answering a query about that part's hierarchy. You MUST formulate your comprehensive answer using these details and provide it in the FINAL ANSWER JSON format. DO NOT call `get_part_hierarchy` again for the same part in this turn.
        c.  For other tools, or if `get_part_hierarchy` returned an error/no specific info: If the Observation is relevant and sufficient: Formulate your comprehensive answer based on the User's Query and the relevant parts of the Observation. Provide this answer in the FINAL ANSWER JSON format.
        d.  If the Observation is clearly irrelevant (e.g., tool called inappropriately, or a search tool returned no relevant results for the specific query): Disregard the irrelevant tool output. Formulate your answer to the User's Query using your general knowledge (if appropriate for the query type) or state that the information could not be found. Provide this answer in the FINAL ANSWER JSON format. You might state something like, "I searched for documents but did not find specific information relevant to [original query topic]. However, generally speaking, [answer to original query]..." or simply provide the general answer directly if no tool was meant to be used.
3.  If NO relevant Observation exists, or if more information is still needed:
    a.  Re-evaluate the User's Query based on Instruction 1 (TOOL USAGE SCOPE). Is it an engineering query needing tools, or a general/philosophical one best answered from general knowledge?
    b.  If a tool is deemed appropriate and necessary for an engineering query: Respond with the TOOL CALL JSON. Ensure the `tool_input` is directly derived from the User's Query and targeted.
    c.  If no tool is needed or appropriate (especially for non-engineering queries): Formulate your answer using general knowledge. Provide this answer in the FINAL ANSWER JSON format.

User's Query: {input}
Scratchpad:
{agent_scratchpad}

Your response (ensure it's one of the valid JSON formats described above):"""
        
        prompt = ChatPromptTemplate.from_template(agent_prompt_template)
        
        # We will not use bind_tools() if gemma3:4b doesn't support the 'tools' API parameter.
        # self.tool_llm is already ChatOllama(..., format="json", ...), which is suitable
        # for generating JSON based on the prompt.
        # agent_llm_with_tools = self.tool_llm.bind_tools(tools) # COMMENT OUT or REMOVE this line

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: self._format_agent_scratchpad(x.get("intermediate_steps", [])),
                tools_string=lambda x: tools_string,
                tool_names=lambda x: tool_names_str,
                tool_history=lambda x: self._format_tool_history(x.get("intermediate_steps", []))
            )
            | prompt
            | self.tool_llm
            | CustomOllamaFunctionsOutputParser() 
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors="An error occurred. Please try rephrasing your request.",
            max_iterations=3,
            early_stopping_method="generate"  # force / generate
        )

    def _format_tool_history(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        """Format tool usage history for the prompt."""
        if not intermediate_steps:
            return "No tools have been used yet."
        
        tool_history = self._extract_tool_history(intermediate_steps)
        history_str = "Previous tool calls:\n"
        
        for tool_name, inputs in tool_history.items():
            history_str += f"- {tool_name}: called {len(inputs)} time(s)\n"
            for inp in inputs:
                history_str += f"  • Input: {inp[:50]}...\n" if len(inp) > 50 else f"  • Input: {inp}\n"
        
        return history_str

    def _format_agent_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        """Formats the agent's scratchpad (intermediate steps) for the Ollama prompt."""
        log = ""
        for action, observation in intermediate_steps:
            log += action.log
            log += f"\nObservation: {observation}\n"
        return log

    async def _handle_direct_general_question(self, query: str, response_payload: Dict[str, Any]) -> bool:
        """
        Attempts to answer a general question directly using the LLM.
        Ensures the LLM output is in the expected {"answer": "..."} JSON format.
        Returns True if answered, False otherwise.
        """
        print(f"[_handle_direct_general_question] Attempting to answer directly: {query}")
        
        # Prompt specifically designed to get a JSON output with an "answer" key
        direct_answer_prompt_template = """
You are an AI assistant. The user has asked a question that seems to be a general knowledge question.
User's Query: "{query}"

Please provide a concise, helpful answer to the User's Query.
Respond ONLY with a single valid JSON object with a single key "answer":
```json
{{
  "answer": "Your direct answer to the query here."
}}
```
Your JSON Response:"""

        # Format the prompt with the user's query
        formatted_prompt = PromptTemplate.from_template(direct_answer_prompt_template).format(query=query)

        try:
            # Use self.tool_llm as it's configured for JSON output.
            # The formatted_prompt guides it to use the {"answer": ...} structure.
            ai_message = await self.tool_llm.ainvoke(formatted_prompt) 
            
            llm_output_content = ai_message.content.strip()
            print(f"[_handle_direct_general_question] Raw LLM output for direct answer: {llm_output_content}")

            # Attempt to parse the LLM's JSON output
            parsed_json = json.loads(llm_output_content)
            
            if "answer" in parsed_json and isinstance(parsed_json["answer"], str):
                answer_content = parsed_json["answer"].strip()
                if answer_content:
                    response_payload["answer"] = answer_content
                    print(f"[_handle_direct_general_question] Successfully answered directly.")
                    return True
                else:
                    print(f"[_handle_direct_general_question] LLM provided an empty answer string within the JSON.")
                    return False
            else:
                print(f"[_handle_direct_general_question] LLM output JSON did not contain an 'answer' key with a string value.")
                return False
        except json.JSONDecodeError:
            print(f"[_handle_direct_general_question] Failed to decode JSON from LLM output: {llm_output_content}")
            return False
        except Exception as e:
            print(f"[_handle_direct_general_question] Error during direct LLM call or processing: {e}")
            return False

    def _process_agent_output(self, agent_result: Dict[str, Any], response_payload: Dict[str, Any]):
        """
        Processes the output from the AgentExecutor, populating the answer,
        citations, and part_info in the response_payload.
        """
        # ---- START DEBUG PRINTS ----
        print(f"\n🔍 [_process_agent_output] Raw agent_result received by _process_agent_output:")
        print(f"Output: {agent_result.get('output')}") # This is the 'final answer' from the agent
        intermediate_steps = agent_result.get("intermediate_steps", [])
        print(f"Intermediate steps type: {type(intermediate_steps)}, Count: {len(intermediate_steps)}")
        if intermediate_steps:
            for i, step_tuple in enumerate(intermediate_steps):
                if not (isinstance(step_tuple, tuple) and len(step_tuple) == 2):
                    print(f"  Step {i} is not a valid (action, observation) tuple: {step_tuple}")
                    continue
                action, observation = step_tuple
                print(f"  Step {i}:")
                action_tool = "N/A"
                action_tool_input = "N/A"
                if hasattr(action, 'tool'): action_tool = action.tool
                if hasattr(action, 'tool_input'): action_tool_input = action.tool_input
                print(f"    Action Tool: {action_tool}")
                print(f"    Action Tool Input: {action_tool_input}")
                obs_summary = str(observation)[:200] + "..." if len(str(observation)) > 200 else str(observation)
                print(f"    Observation (summary): {obs_summary}")
        else:
            print("  No intermediate steps found in agent_result.")
        print("---- END DEBUG PRINTS ----\n")
        
        answer = agent_result.get("output", "I apologize, I couldn't formulate a response.")

        # The 'output' from AgentExecutor (parsed by CustomOllamaFunctionsOutputParser)
        # should already be the clean textual answer if parsing was successful.
        # This additional cleaning is a fallback.
        if isinstance(answer, str):
            cleaned_answer = answer.strip()
            if not cleaned_answer or cleaned_answer == "{}": # Should be less likely now
                cleaned_answer = "I was unable to find specific information for your query. Could you please rephrase or provide more details?"
            # No need to re-parse JSON here if CustomOllamaFunctionsOutputParser did its job for {"answer": ...}
            response_payload["answer"] = cleaned_answer
        else:
            # If answer is not a string (e.g., if parser passed raw dict through on error)
            response_payload["answer"] = str(answer) if answer else "I apologize, I couldn't formulate a response."


        # Populate citations and part_info from intermediate steps
        if intermediate_steps:
            for step_idx, step_content in enumerate(intermediate_steps):
                if not (isinstance(step_content, tuple) and len(step_content) == 2):
                    continue # Skip malformed steps already logged above
                action, observation = step_content
                tool_called = getattr(action, 'tool', '')
                
                if tool_called == "search_documents":
                    print(f"[_process_agent_output] Processing 'search_documents' observation from step {step_idx} for citations.")
                    documents_from_tool = observation if isinstance(observation, list) else []
                    if not documents_from_tool:
                         print(f"[_process_agent_output] 'search_documents' observation in step {step_idx} is empty or not a list.")

                    # Populate documents_retrieved field
                    current_doc_retrieved_ids = {doc.get("metadata", {}).get("chunk_id") for doc in response_payload.get("documents_retrieved", []) if isinstance(doc, dict)}
                    for doc_dict in documents_from_tool:
                        if isinstance(doc_dict, dict) and doc_dict.get("metadata", {}).get("chunk_id") not in current_doc_retrieved_ids:
                            response_payload["documents_retrieved"].append(doc_dict)
                            current_doc_retrieved_ids.add(doc_dict.get("metadata", {}).get("chunk_id"))


                    # Populate citations list
                    current_citation_sources_pages = {(c.get("source"), c.get("page"), c.get("row")) for c in response_payload.get("citations", [])}
                    for doc_idx_obs, doc_dict in enumerate(documents_from_tool):
                        if isinstance(doc_dict, dict):
                            citation_tuple = (doc_dict.get("source"), doc_dict.get("page"), doc_dict.get("row"))
                            if citation_tuple not in current_citation_sources_pages:
                                print(f"[_process_agent_output] Adding citation for doc {doc_idx_obs} from observation: {doc_dict.get('source')}")
                                citation = {
                                    "source": doc_dict.get("source", "Unknown"),
                                    "page": doc_dict.get("page"), 
                                    "row": doc_dict.get("row"),
                                    "score": doc_dict.get("score", 0.0),
                                    "excerpt": doc_dict.get("content", "")[:250] + "..." if doc_dict.get("content") else "",
                                    "metadata": doc_dict.get("metadata", {}) # Contains chunk_id
                                }
                                response_payload["citations"].append(citation)
                                current_citation_sources_pages.add(citation_tuple)
                
                elif tool_called == "get_part_hierarchy":
                    print(f"[_process_agent_output] Processing 'get_part_hierarchy' observation from step {step_idx}.")
                    if isinstance(observation, dict): 
                        response_payload["part_info"] = observation # Assuming only one part_info needed
                elif tool_called == "get_parts_awaiting_maintenance":
                    print(f"[_process_agent_output] Processing 'get_parts_awaiting_maintenance' observation from step {step_idx}.")
                    if isinstance(observation, dict) and "part_name" in observation and "dates" in observation:
                        response_payload["maintenance_data"] = observation

    async def _invoke_agent_and_process_results(self, query: str, classification: QueryClassification, response_payload: Dict[str, Any]):
        """
        Creates the agent executor, invokes it, and processes its results.
        """
        agent_executor = self.create_agent_executor()
        agent_input = {"input": query} # Intermediate steps will be added by the RunnablePassthrough

        try:
            result = await agent_executor.ainvoke(agent_input)
            self._process_agent_output(result, response_payload)
        except Exception as e:
            full_error_message = f"Error during agent execution: {e}"
            import traceback
            traceback.print_exc()
            response_payload["answer"] = "I encountered an error while processing your request with my tools."



    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using classification and the Ollama agent executor."""
        print(f"\n🚀 [process_query] Received query: {query}")
        classification = self.classify_query(query)
        print(f"[process_query] Classification result: {classification.model_dump()}")
        
        response_payload: Dict[str, Any] = {
            "query": query,
            "classification": classification.model_dump(), 
            "answer": "",
            "citations": [],
            "part_info": None,
            "documents_retrieved": [],
            "maintenance_data": None
        }
        
        proceed_to_agent_executor = True 

        if classification.query_type == QueryType.GENERAL_QUESTION:
            if not classification.requires_rag and classification.confidence >= 0.85: # Tuned confidence
                print("[process_query] Attempting direct handling for GENERAL_QUESTION.")
                if await self._handle_direct_general_question(query, response_payload):
                    proceed_to_agent_executor = False 
                else:
                    print("[process_query] Direct handling failed or provided no answer, proceeding to agent executor.")
            else:
                print("[process_query] GENERAL_QUESTION but requires_rag or low confidence, proceeding to agent executor.")
        
        if proceed_to_agent_executor:
            print("[process_query] Proceeding to agent executor.")
            await self._invoke_agent_and_process_results(query, classification, response_payload)
        
        print(f"[process_query] Final response_payload (summary):")
        print(f"  Answer: {str(response_payload['answer'])[:200]}...")
        print(f"  Citations count: {len(response_payload['citations'])}")
        print(f"  Part Info: {'Present' if response_payload['part_info'] else 'None'}")
        print(f"  Docs Retrieved count: {len(response_payload['documents_retrieved'])}")
        print(f"  Maintenance Data: {'Present' if response_payload['maintenance_data'] else 'None'}")
        return response_payload

