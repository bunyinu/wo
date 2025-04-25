import os
from typing import Optional

# Removed Daytona imports as per requirements

from agentpress.tool import Tool
from utils.logger import logger
from utils.config import config
from utils.files_utils import clean_path
from agentpress.thread_manager import ThreadManager

from dotenv import load_dotenv
load_dotenv()

logger.debug("Initializing sandbox configuration without Daytona")

class MockSandbox:
    """Mock sandbox class to replace Daytona functionality"""
    
    def __init__(self, sandbox_id):
        self.id = sandbox_id
        self.instance = type('MockInstance', (), {'state': 'RUNNING'})
        self.process = type('MockProcess', (), {
            'create_session': lambda session_id: logger.debug(f"Mock creating session {session_id}"),
            'execute_session_command': lambda session_id, request: logger.debug(f"Mock executing command in session {session_id}")
        })
        self.fs = type('MockFS', (), {
            'upload_file': lambda target_path, content: logger.debug(f"Mock uploading file to {target_path}"),
            'list_files': lambda parent_dir: []
        })
        logger.debug(f"Initialized mock sandbox with ID: {sandbox_id}")
    
    def get_preview_link(self, port):
        """Mock method to get preview link"""
        return f"http://localhost:{port}"

async def get_or_start_sandbox(sandbox_id: str):
    """Mock function to retrieve a sandbox by ID"""
    logger.info(f"Mock getting or starting sandbox with ID: {sandbox_id}")
    return MockSandbox(sandbox_id)

def start_supervisord_session(sandbox):
    """Mock function to start supervisord in a session"""
    logger.info(f"Mock starting supervisord session")
    return True

def create_sandbox(password: str, project_id: str = None):
    """Mock function to create a new sandbox"""
    logger.debug("Mock creating new sandbox environment")
    
    sandbox_id = project_id or "mock-sandbox-" + os.urandom(4).hex()
    sandbox = MockSandbox(sandbox_id)
    
    logger.debug(f"Mock sandbox created with ID: {sandbox.id}")
    return sandbox

class SandboxToolsBase(Tool):
    """Base class for all sandbox tools that provides project-based sandbox access."""
    
    # Class variable to track if sandbox URLs have been printed
    _urls_printed = False
    
    def __init__(self, project_id: str, thread_manager: Optional[ThreadManager] = None):
        super().__init__()
        self.project_id = project_id
        self.thread_manager = thread_manager
        self.workspace_path = "/workspace"
        self._sandbox = None
        self._sandbox_id = None
        self._sandbox_pass = None

    async def _ensure_sandbox(self):
        """Ensure we have a valid sandbox instance, retrieving it from the project if needed."""
        if self._sandbox is None:
            try:
                # Get database client
                client = await self.thread_manager.db.client
                
                # Get project data
                project = await client.table('projects').select('*').eq('project_id', self.project_id).execute()
                if not project.data or len(project.data) == 0:
                    raise ValueError(f"Project {self.project_id} not found")
                
                project_data = project.data[0]
                sandbox_info = project_data.get('sandbox', {})
                
                if not sandbox_info.get('id'):
                    # Create a mock sandbox ID if none exists
                    sandbox_id = "mock-sandbox-" + os.urandom(4).hex()
                    logger.info(f"No sandbox found for project {self.project_id}, creating mock sandbox with ID {sandbox_id}")
                    
                    # Update project with mock sandbox info
                    await client.table('projects').update({
                        'sandbox': {'id': sandbox_id, 'pass': 'mockpass'}
                    }).eq('project_id', self.project_id).execute()
                    
                    self._sandbox_id = sandbox_id
                    self._sandbox_pass = 'mockpass'
                else:
                    # Store sandbox info
                    self._sandbox_id = sandbox_info['id']
                    self._sandbox_pass = sandbox_info.get('pass')
                
                # Get or start the sandbox
                self._sandbox = await get_or_start_sandbox(self._sandbox_id)
                
            except Exception as e:
                logger.error(f"Error retrieving sandbox for project {self.project_id}: {str(e)}", exc_info=True)
                raise e
        
        return self._sandbox

    @property
    def sandbox(self):
        """Get the sandbox instance, ensuring it exists."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not initialized. Call _ensure_sandbox() first.")
        return self._sandbox

    @property
    def sandbox_id(self) -> str:
        """Get the sandbox ID, ensuring it exists."""
        if self._sandbox_id is None:
            raise RuntimeError("Sandbox ID not initialized. Call _ensure_sandbox() first.")
        return self._sandbox_id

    def clean_path(self, path: str) -> str:
        """Clean and normalize a path to be relative to /workspace."""
        cleaned_path = clean_path(path, self.workspace_path)
        logger.debug(f"Cleaned path: {path} -> {cleaned_path}")
        return cleaned_path
