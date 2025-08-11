

import traceback
import json



@dataclass
class Tool:
    name: str
    description: str
    function: callable
    parameters: Dict[str, Any]


    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LLM prompt"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {}
        }

@dataclass
class Todo:
    id: int
    tool_name: str
    task: str
    tool_parameters: Dice[str, Any]


class Agent():
    """
    Main agent to take initial query and run other tools.
    """
    def __init__(self):

        self.rag = RAGSystem()
        self.tools: Dict[str, Tool] = {}
        self.todo_list: List[Todo] = []
        self.current_todo_id = 0


    def run_agent(self, query: str):
        """ """

        prompt = self._get_tool_prompt(query, self.tools)

        todos = self._create_todo_items(query = prompt, )

        tools = self.parse_tool_info(prompt)
    

    def register_tool(self, tool: Tool):
        """Register a new tool with the agent"""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for the LLM prompt"""
        return [tool.to_dict() for tool in self.tools.values()]


    def _get_todolist_prompt(self, query: str):
        """ Get prompt """

        tool_list = json.dumps(self.get_available_tools(), indent=2)
        
        prompt = """
            Your goal is to decide on a list of todo items formatted as a step by step set of tasks. Each item should specify 
            1. What the task is
            2. what tool to use
            3. what parameters to use for the tool

            The available tools are {tool_list} 

            The user query is {query}


            You must format your answer as a json with the structure
            [{"name": "todo_item1_name",
              "summary": a summary of what must be done for this task,
              "tool": SearchTool,
              "parameters": {
                "search_string": what to look for
                "top_k": 5
              }
            },
            {"name": "todo_item2_name",
              "summary": a summary of what must be done for this task,
              "tool": MathTool,
              "parameters": {
                "search_string": what to look for
                "top_k": 5
              }
            },
            ]

            create a logical sequency so adress the users query
        """
        return prompt

    def _parse_todo_generation(self, llm_response: str) -> Optional[List[Dict[str, Any]]]:
        """Parse the LLM's todo list generation response"""
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                data = json.loads(json_str)
                return data.get("todos", [])
            else:
                print(f"No valid JSON found in response: {llm_response}")
                return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse todo generation JSON: {e}")
            print(f"Response was: {llm_response}")
            return None

    def _create_todo_list(self, query):
        """ create todo list from prompt"""

        prompt = self._get_todolist_prompt(query)

        todo_response = self.rag_system.get_model_response(prompt)

        todo_data = self._parse_todo_generation(todo_response)

        todo_items = []
        for i, todo in enumerate(todo_data):
            todo_item = Todo(
                id=str(i + 1),
                task=todo.get("task", f"Task {i + 1}"),
                tool_name=todo.get("tool_name", "general_knowledge"),
                parameters=todo.get("parameters", {})
            )
            todo_items.append(todo_item)
        
        self.todo_list = todo_items
        return todo_items


    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        tool = self.tools[tool_name]
        try:
            result = tool.function(**parameters)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def execute_todo_item(self, todo_item: TodoItem, verbose: bool = False) -> Dict[str, Any]:
        """Execute a single todo item"""
        if verbose:
            print(f"Executing todo: {todo_item.task}")
        
        todo_item.status = "in_progress"
        
        result = self.execute_tool(todo_item.tool_name, todo_item.parameters)
        
        if result["success"]:
            todo_item.status = "completed"
            todo_item.result = result
            if verbose:
                print(f"✓ Completed: {todo_item.task}")
        else:
            todo_item.status = "failed"
            todo_item.result = result
            if verbose:
                print(f"✗ Failed: {todo_item.task} - {result.get('error', 'Unknown error')}")
        
        return result
    
    def execute_all_todos(self, verbose: bool = False) -> List[Dict[str, Any]]:
        """Execute all todos in sequence"""
        results = []
        
        for todo_item in self.todo_list:
            if verbose:
                print(f"\n📋 Todo {todo_item.id}: {todo_item.task}")
                print(f"   Tool: {todo_item.tool_name}")
                print(f"   Parameters: {todo_item.parameters}")
            
            result = self.execute_todo_item(todo_item, verbose)
            results.append({
                "todo_id": todo_item.id,
                "task": todo_item.task,
                "status": todo_item.status,
                "result": result
            })
        
        return results
    
