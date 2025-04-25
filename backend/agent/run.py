import os
import json
import re
from uuid import uuid4
from typing import Optional

# from agent.tools.message_tool import MessageTool
from agent.tools.message_tool import MessageTool
from agent.tools.sb_deploy_tool import SandboxDeployTool
from agent.tools.sb_expose_tool import SandboxExposeTool
from agent.tools.web_search_tool import WebSearchTool
from dotenv import load_dotenv
from utils.config import config

from agentpress.thread_manager import ThreadManager
from agentpress.response_processor import ProcessorConfig
from agent.tools.sb_shell_tool import SandboxShellTool
from agent.tools.sb_files_tool import SandboxFilesTool
from agent.tools.sb_browser_tool import SandboxBrowserTool
from agent.tools.data_providers_tool import DataProvidersTool
from agent.prompt import get_system_prompt
from utils import logger
from utils.billing import check_billing_status, get_account_id_from_thread

load_dotenv()

async def run_agent(
    thread_id: str,
    project_id: str,
    stream: bool,
    thread_manager: Optional[ThreadManager] = None,
    native_max_auto_continues: int = 25,
    max_iterations: int = 150,
    model_name: str = "xai/grok-latest",
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low',
    enable_context_manager: bool = True
):
    """Run the development agent with specified configuration."""
    
    thread_manager = ThreadManager()

    client = await thread_manager.db.client

    # Get account ID from thread for billing checks
    account_id = await get_account_id_from_thread(client, thread_id)
    if not account_id:
        raise ValueError("Could not determine account ID for thread")

    # Get sandbox info from project
    project = await client.table('projects').select('*').eq('project_id', project_id).execute()
    if not project.data or len(project.data) == 0:
        raise ValueError(f"Project {project_id} not found")
    
    project_data = project.data[0]
    sandbox_info = project_data.get('sandbox', {})
    if not sandbox_info.get('id'):
        raise ValueError(f"No sandbox found for project {project_id}")
    
    # Initialize tools with project_id instead of sandbox object
    # This ensures each tool independently verifies it's operating on the correct project
    thread_manager.add_tool(SandboxShellTool, project_id=project_id, thread_manager=thread_manager)
    thread_manager.add_tool(SandboxFilesTool, project_id=project_id, thread_manager=thread_manager)
    thread_manager.add_tool(SandboxBrowserTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
    thread_manager.add_tool(SandboxDeployTool, project_id=project_id, thread_manager=thread_manager)
    thread_manager.add_tool(SandboxExposeTool, project_id=project_id, thread_manager=thread_manager)
    thread_manager.add_tool(MessageTool) # we are just doing this via prompt as there is no need to call it as a tool
 
    # Add more tools if API keys are available
    if config.TAVILY_API_KEY:
        thread_manager.add_tool(WebSearchTool)
        
    # Add data providers tool if RapidAPI key is available
    if config.RAPID_API_KEY:
        thread_manager.add_tool(DataProvidersTool)

    system_message = { "role": "system", "content": get_system_prompt() }

    iteration_count = 0
    continue_execution = True
    
    while continue_execution and iteration_count < max_iterations:
        iteration_count += 1
        # logger.debug(f"Running iteration {iteration_count}...")

        # Billing check on each iteration - still needed within the iterations
        can_run, message, subscription = await check_billing_status(client, account_id)
        if not can_run:
            error_msg = f"Billing limit reached: {message}"
            # Yield a special message to indicate billing limit reached
            yield {
                "type": "status",
                "status": "stopped",
                "message": error_msg
            }
            break
        # Check if last message is from assistant using direct Supabase query
        latest_message = await client.table('messages').select('*').eq('thread_id', thread_id).in_('type', ['assistant', 'tool', 'user']).order('created_at', desc=True).limit(1).execute()  
        if latest_message.data and len(latest_message.data) > 0:
            message_type = latest_message.data[0].get('type')
            if message_type == 'assistant':
                print(f"Last message was from assistant, stopping execution")
                continue_execution = False
                break
            
        # Get the latest message from messages table that its type is browser_state
        latest_browser_state = await client.table('messages').select('*').eq('thread_id', thread_id).eq('type', 'browser_state').order('created_at', desc=True).limit(1).execute()
        temporary_message = None
        if latest_browser_state.data and len(latest_browser_state.data) > 0:
            try:
                content = json.loads(latest_browser_state.data[0]["content"])
                screenshot_base64 = content["screenshot_base64"]
                # Create a copy of the browser state without screenshot
                browser_state = content.copy()
                browser_state.pop('screenshot_base64', None)
                browser_state.pop('screenshot_url', None) 
                browser_state.pop('screenshot_url_base64', None)
                temporary_message = { "role": "user", "content": [] }
                if browser_state:
                    temporary_message["content"].append({
                        "type": "text",
                        "text": f"The following is the current state of the browser:\n{browser_state}"
                    })
                if screenshot_base64:
                    temporary_message["content"].append({
                        "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{screenshot_base64}",
                            }
                    })
                else:
                    print("@@@@@ THIS TIME NO SCREENSHOT!!")
            except Exception as e:
                print(f"Error parsing browser state: {e}")
                # print(latest_browser_state.data[0])
        
        max_tokens = 64000 if "sonnet" in model_name.lower() else None

        response = await thread_manager.run_thread(
            thread_id=thread_id,
            system_prompt=system_message,
            stream=stream,
            llm_model=model_name,
            llm_temperature=0,
            llm_max_tokens=max_tokens,
            tool_choice="auto",
            max_xml_tool_calls=1,
            temporary_message=temporary_message,
            processor_config=ProcessorConfig(
                xml_tool_calling=True,
                native_tool_calling=False,
                execute_tools=True,
                execute_on_stream=True,
                tool_execution_strategy="parallel",
                xml_adding_strategy="user_message"
            ),
            native_max_auto_continues=native_max_auto_continues,
            include_xml_examples=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            enable_context_manager=enable_context_manager
        )
            
        if isinstance(response, dict) and "status" in response and response["status"] == "error":
            yield response 
            break
            
        # Track if we see ask or complete tool calls
        last_tool_call = None
        
        async for chunk in response:
            # print(f"CHUNK: {chunk}") # Uncomment for detailed chunk logging

            # Check for XML versions like <ask> or <complete> in assistant content chunks
            if chunk.get('type') == 'assistant' and 'content' in chunk:
                try:
                    # The content field might be a JSON string or object
                    content = chunk.get('content', '{}')
                    if isinstance(content, str):
                        assistant_content_json = json.loads(content)
                    else:
                        assistant_content_json = content
                        
                    # The actual text content is nested within
                    assistant_text = assistant_content_json.get('content', '')
                    if isinstance(assistant_text, str): # Ensure it's a string
                         # Check for the closing tags as they signal the end of the tool usage
                        if '</ask>' in assistant_text or '</complete>' in assistant_text:
                           xml_tool = 'ask' if '</ask>' in assistant_text else 'complete'
                           last_tool_call = xml_tool
                           print(f"Agent used XML tool: {xml_tool}")
                except json.JSONDecodeError:
                    # Handle cases where content might not be valid JSON
                    print(f"Warning: Could not parse assistant content JSON: {chunk.get('content')}")
                except Exception as e:
                    print(f"Error processing assistant chunk: {e}")
                    
            yield chunk
        
        # Check if we should stop based on the last tool call
        if last_tool_call in ['ask', 'complete']:
            print(f"Agent decided to stop with tool: {last_tool_call}")
            continue_execution = False
