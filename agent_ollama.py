# agent_ollama.py

from typing import List, Dict, Any, Optional, Tuple, Union, Set
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompt_values import ChatPromptValue
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
                    
                    # Special handling for final_answer tool
                    if tool == "final_answer":
                        return AgentFinish(return_values={"output": tool_input}, log=text)
                    
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
    GET_MAINTENANCE_DATA = "get_maintenance_data"
    GET_PART_HIERARCHY = "get_part_hierarchy"

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

    def get_maintenance_data(self, part_name: str) -> Dict[str, Any]:
        """
        Generates dummy data for parts awaiting maintenance for a given part name.
        This function is called by the 'get_maintenance_data' tool.
        Input can be a string (part_name) or a dict {'part_name': 'XYZ'}.
        """
        actual_part_name = part_name
        if isinstance(part_name, dict) and "part_name" in part_name:
            actual_part_name = part_name["part_name"]
        elif not isinstance(part_name, str):
            # Fallback if input is not as expected
            print(f"Warning: Unexpected input type for part_name in get_maintenance_data: {type(part_name)}. Using 'UnknownPart'.")
            actual_part_name = "UnknownPart"

        print(f"Tool 'get_maintenance_data' called for part: {actual_part_name}")
        
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
        
        # todo: auto populate query type
        # Much more concise classification prompt
        classification_prompt_template = """Classify this engineering query into JSON format:
Query: "{query}"

Response as JSON only:
{{
"query_type": "one of: part_query, document_search, general_question, hybrid_query, get_maintenance_data",
"part_identifiers": ["extracted part names/numbers"],
"requires_rag": "true/false",
"search_keywords": ["key terms"],
"confidence": "0.0-1.0"
}}

Classification Rules:
- get_maintenance_data: if mentions maintenance, AWP, AWM, AWPM, trends
- part_query: if asks about specific parts/hierarchy  
- document_search: if asks for RAG/document search
- general_question: for non-engineering questions

Examples:
"Show me maintenance data for TIE Fighter" → {{"query_type": "get_maintenance_data", "part_identifiers": ["TIE Fighter"], "requires_rag": false, "search_keywords": ["maintenance", "TIE Fighter"], "confidence": 0.95}}
"What is part 123?" → {{"query_type": "part_query", "part_identifiers": ["123"], "requires_rag": false, "search_keywords": ["part", "123"], "confidence": 0.95}}

JSON only:"""
        
        prompt = PromptTemplate.from_template(classification_prompt_template)
        
        # Rest of the method remains the same
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
            results = self.vector_store.similarity_search_with_score(query, k=5)
            return [{"content": doc.page_content, "metadata": doc.metadata, "score": score, "source": doc.metadata.get("source","Unknown")} for doc, score in results]
        except Exception as e:
            return [{"content": f"Error in placeholder search_documents: {str(e)}", "metadata": {}, "score": 0.0, "source": "SystemInternal"}]

    def final_answer(self, answer: str) -> str:
        """Provide the final answer to the user's query."""
        return answer

    def create_tools(self) -> List[Tool]:
        """Create tools for the agent. Descriptions should be extremely clear and self-contained.
		Should describe when to use the tool, input expectations, and output nature.
		"""
        return [
            Tool(
                name="get_part_hierarchy",
                func=self.get_part_hierarchy,
                description='''Use this tool to retrieve the hierarchical structure and detailed information for a specific engineering part. 
                Input MUST be a single string representing the exact part number (e.g., '1.1.2.3', 'Compressor-X45'). 
                Output is a JSON object containing part details, hierarchy list, and metadata.'''
            ),
            Tool(
                name="search_documents",
                func=lambda query: self.search_documents(query, k=5),  # Ensure k=5 is used
                description='''Use this tool to search relevant engineering documents, manuals, or reports based on a natural language query.
                Input MUST be a string containing the search query. 
                The query can include part numbers, technical concepts, or problem descriptions. 
                Output is a list of relevant document chunks with their sources and content.
                This tool returns UP TO 5 relevant documents ranked by similarity.'''
            ),
            Tool(
                name="get_maintenance_data",
                func=self.get_maintenance_data,
                description='''Use this tool when the user asks about maintenance data, AWP (awaiting parts), AWM (awaiting manufacturing), AWPM counts, maintenance trends, or time-series maintenance information for a part.
                TRIGGER WORDS: "maintenance", "AWP", "AWM", "AWPM", "awaiting", "trends", "maintenance data", "time-series"
                Input MUST be a single string representing the part name (e.g., 'Pump-X12', 'Sensor-Unit-A', 'TIE Fighter'). 
                Output is a JSON object containing dates and counts for AWP/AWM/AWPM, suitable for generating maintenance trend charts.'''
            ),
            Tool(
                name="final_answer",
                func=self.final_answer,
                description='''Use this tool to provide your final answer to the user's query. 
                Input MUST be a complete, well-formatted answer string that synthesizes all the information gathered from previous tool calls. 
                This should be the LAST tool you call after gathering all necessary information.'''
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

**Query Analysis:**
- Classified as: {query_type}
- Part identifiers found: {part_identifiers}
- Based on this classification, select the appropriate tool accordingly.

You have access to the following tools:
{tools_string}

**Critical Instructions:**

1. **TOOL SELECTION PRIORITY**: 
   - If the query mentions "maintenance", "AWP", "AWM", "AWPM", "maintenance_data", or "get_maintenance_data", use get_maintenance_data tool FIRST
   - If the query asks for part hierarchy or part information, use get_part_hierarchy tool FIRST  
   - If the query asks for document search or RAG, use search_documents tool FIRST
   - Pay close attention to any "IMPORTANT" instructions in the User's Query

2. **ALWAYS end with final_answer**: After gathering information using other tools, you MUST call the "final_answer" tool to provide your response to the user.

3. **Tool Usage Flow**:
   - First, use the MOST APPROPRIATE tool based on the query type (see priority above)
   - Then, ALWAYS call final_answer with your complete response
   - Never provide an answer without using the final_answer tool

4. **Tool Response Format**: 
   Always respond with a valid JSON object:
   ```json
   {{
     "tool": "tool_name", 
     "tool_input": "appropriate input for the tool"
   }}
   ```

5. **HANDLING INSUFFICIENT DOCUMENTATION/DATA**:
   * If the content of retrieved documents/data is extremely sparse or clearly not what was asked for, your answer MUST reflect this. State that "The search/tool provided limited/minimal information regarding X..." or "The retrieved data for Y was not specific to the request."
   * Do NOT attempt to elaborate or expand on such minimal/irrelevant information using your general knowledge if the query was specific.
   * It is better to state that the specific information is not available in the documents/data than to provide potentially misleading information.
   
   Do NOT use plain text directly for final answers if you are not calling a tool. Always wrap your final textual answer in the JSON structure specified above.

**Tool Usage History:**
{tool_history}

**Decision Process:**
1. Examine the User's Query and the Scratchpad.
2. If you haven't gathered the necessary information yet, call the appropriate information-gathering tool
3. Once you have sufficient information, call the final_answer tool with your complete response.
4. Remember: final_answer should always be your last tool call

Scratchpad:
{agent_scratchpad}

Your response (ensure it's one of the valid JSON formats described above):"""
        
        prompt = ChatPromptTemplate.from_template(agent_prompt_template)
		
		# Define a small function to print the LLM's input
        def print_llm_input_passthrough(prompt_value: ChatPromptValue):
            print("\n================ LLM Input (Full Prompt) ================")
            print(prompt_value.to_string())
            print("================ End LLM Input ================\n")
            return prompt_value

        # Create a parser instance that can track state
        output_parser = CustomOllamaFunctionsOutputParser()

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: self._format_agent_scratchpad(x.get("intermediate_steps", [])),
                tools_string=lambda x: tools_string,
                tool_names=lambda x: tool_names_str,
                tool_history=lambda x: self._format_tool_history(x.get("intermediate_steps", [])),
                query_type=lambda x: x.get("query_type", "unknown"),  # ADD THIS
                part_identifiers=lambda x: x.get("part_identifiers", [])  # ADD THIS
            )
            | prompt
            | RunnableLambda(print_llm_input_passthrough)
            | self.tool_llm
            | output_parser
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors="An error occurred. Please try rephrasing your request.",
            max_iterations=3,
            early_stopping_method="force"
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
        """Formats the agent's scratchpad."""
        log = ""
        if not intermediate_steps: 
            return log

        has_info_gathering_results = False
        called_final_answer = False
        
        for i, (action, observation) in enumerate(intermediate_steps):
            log += f"Previous Action Log (Iteration {i+1}): {action.log}\n"
            
            if hasattr(action, 'tool') and action.tool == "final_answer":
                called_final_answer = True
            elif hasattr(action, 'tool') and action.tool in ["get_part_hierarchy", "search_documents", "get_maintenance_data"]:
                has_info_gathering_results = True
            
            obs_summary = ""
            max_obs_length = 3000  # Increased from 1500 to allow more context

            if isinstance(observation, str):
                obs_summary = (observation[:max_obs_length - 3] + "...") if len(observation) > max_obs_length else observation
            elif isinstance(observation, (list, dict)): # This handles search_documents results (list of dicts)
                try: 
                    obs_str = json.dumps(observation, ensure_ascii=False) # Added ensure_ascii=False for wider char support
                except TypeError: 
                    obs_str = str(observation)
                obs_summary = (obs_str[:max_obs_length - 3] + "...") if len(obs_str) > max_obs_length else obs_str
            elif observation is None: 
                obs_summary = "No observation was returned from the tool."
            else: # Fallback for other types
                try:
                    str_repr = str(observation)
                    obs_summary = (str_repr[:max_obs_length - 3] + "...") if len(str_repr) > max_obs_length else str_repr
                except Exception:
                    obs_summary = f"Observation received (type: {type(observation).__name__}, summary might be limited)."
            log += f"Observation (Iteration {i+1}): {obs_summary}\n\n"
        
        if log:
            if called_final_answer:
                log += "You have already provided the final answer. No further action needed."
            elif has_info_gathering_results:
                log += """You have gathered information from tools. Now you MUST call the final_answer tool with your complete response.
        Example: {"tool": "final_answer", "tool_input": "Based on the search results, [your complete answer here]"}"""
            else:
                log += "Based on the User's Query, select the appropriate tool to gather information."
        
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
        print(f"\n [_process_agent_output] Raw agent_result received by _process_agent_output:")
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

        if isinstance(answer, str):
            cleaned_answer = answer.strip()
            if not cleaned_answer or cleaned_answer == "{}": 
                cleaned_answer = "I was unable to find specific information for your query. Could you please rephrase or provide more details?"
            response_payload["answer"] = cleaned_answer
        else:
            response_payload["answer"] = str(answer) if answer else "I apologize, I couldn't formulate a response."


        if intermediate_steps:
            # Use a set to track unique chunk_ids already processed into citations for this response_payload
            processed_chunk_ids_for_this_response = {
                c.get("metadata", {}).get("chunk_id") 
                for c in response_payload.get("citations", []) 
                if isinstance(c, dict) and c.get("metadata") and c.get("metadata", {}).get("chunk_id")
            }

            for step_idx, step_content in enumerate(intermediate_steps):
                if not (isinstance(step_content, tuple) and len(step_content) == 2):
                    continue 
                action, observation = step_content
                tool_called = getattr(action, 'tool', '')
                
                if tool_called == "search_documents":
                    print(f"[_process_agent_output] Processing 'search_documents' observation from step {step_idx} for citations.")
                    documents_from_tool = observation if isinstance(observation, list) else []
                    if not documents_from_tool:
                         print(f"[_process_agent_output] 'search_documents' observation in step {step_idx} is empty or not a list.")

                    # Populate documents_retrieved field (de-duplicated by chunk_id)
                    current_doc_retrieved_ids = {
                        doc.get("metadata", {}).get("chunk_id") 
                        for doc in response_payload.get("documents_retrieved", []) 
                        if isinstance(doc, dict) and doc.get("metadata") and doc.get("metadata",{}).get("chunk_id")
                    }
                    for doc_dict in documents_from_tool: # doc_dict is a chunk from search results
                        if isinstance(doc_dict, dict):
                            retrieved_chunk_id = doc_dict.get("chunk_id") # chunk_id from search result
                            
                            # Add to documents_retrieved if not already there (based on chunk_id)
                            if retrieved_chunk_id and retrieved_chunk_id not in current_doc_retrieved_ids:
                                response_payload["documents_retrieved"].append(doc_dict) # Appending the whole doc_dict
                                current_doc_retrieved_ids.add(retrieved_chunk_id)

                            # Create citation if this chunk_id hasn't been made into a citation yet for this response
                            if retrieved_chunk_id and retrieved_chunk_id not in processed_chunk_ids_for_this_response:
                                print(f"[_process_agent_output] Adding citation for chunk_id: {retrieved_chunk_id} from source: {doc_dict.get('source')}, page: {doc_dict.get('page')}")
                                
                                new_excerpt_length = 500  # Increased excerpt length
                                raw_content = doc_dict.get("content", "")
                                excerpt = (raw_content[:new_excerpt_length] + "...") if len(raw_content) > new_excerpt_length else raw_content
                                
                                # Ensure metadata in the citation object also contains the chunk_id
                                # The doc_dict itself should contain metadata from the RAG system
                                citation_metadata = doc_dict.get("metadata", {}) 
                                if not isinstance(citation_metadata, dict): # Should be a dict from RAG
                                    citation_metadata = {}
                                # The chunk_id is usually at the top level of doc_dict from search_with_citations,
                                # but also expected inside its metadata field. We ensure it's in the citation's metadata.
                                if "chunk_id" not in citation_metadata and retrieved_chunk_id:
                                    citation_metadata["chunk_id"] = retrieved_chunk_id

                                citation = {
                                    "source": doc_dict.get("source", "Unknown"),
                                    "page": doc_dict.get("page"),
                                    "row": doc_dict.get("row"),
                                    "score": doc_dict.get("score", 0.0),
                                    "excerpt": excerpt,
                                    "metadata": citation_metadata # Contains chunk_id and other original metadata
                                }
                                response_payload["citations"].append(citation)
                                processed_chunk_ids_for_this_response.add(retrieved_chunk_id)
                
                elif tool_called == "get_part_hierarchy":
                    print(f"[_process_agent_output] Processing 'get_part_hierarchy' observation from step {step_idx}.")
                    if isinstance(observation, dict): 
                        response_payload["part_info"] = observation 
                elif tool_called == "get_maintenance_data":
                    print(f"[_process_agent_output] Processing 'get_maintenance_data' observation from step {step_idx}.")
                    if isinstance(observation, dict) and "part_name" in observation and "dates" in observation:
                        response_payload["maintenance_data"] = observation

    async def _invoke_agent_and_process_results(self, query: str, classification: QueryClassification, response_payload: Dict[str, Any]):
        """Creates the agent executor, invokes it, and processes its results."""
        
        # Add debug logging
        print(f"[DEBUG] Query classification type: {classification.query_type}")
        print(f"[DEBUG] Part identifiers: {classification.part_identifiers}")
        
        # Provide hints based on classification
        # Provide stronger, more directive hints
        hint = ""
        if classification.query_type == QueryType.GET_MAINTENANCE_DATA and classification.part_identifiers:
            hint = f"\n\nIMPORTANT: You MUST use the get_maintenance_data tool for part '{classification.part_identifiers[0]}' to answer this query. Do NOT use other tools first."
        elif classification.query_type == QueryType.PART_QUERY and classification.part_identifiers:
            hint = f"\n\nIMPORTANT: You MUST use the get_part_hierarchy tool for part '{classification.part_identifiers[0]}' to answer this query."
        elif "rag" in query.lower() and classification.requires_rag:
            hint = "\n\nIMPORTANT: You MUST use the search_documents tool for this RAG search."
        
        augmented_query = query + hint
        agent_input = {
            "input": augmented_query,
            "query_type": classification.query_type.value,  # Add this
            "part_identifiers": classification.part_identifiers  # And this
        }
        
        agent_executor = self.create_agent_executor()
        
        try:
            result = await agent_executor.ainvoke(agent_input)
            self._process_agent_output(result, response_payload)
        except Exception as e:
            full_error_message = f"Error during agent execution: {e}"
            import traceback
            traceback.print_exc()
            response_payload["answer"] = "I encountered an error while processing your request with my tools."


    async def process_query(self, query: str) -> Dict[str, Any]:
        """
		Process a query using classification and the Ollama agent executor.
		
		First classifies the correct tool to use then routes it to that tool.
		"""
        print(f"\n [process_query] Received query: {query}")
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
            if not classification.requires_rag: # Tuned confidence
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

