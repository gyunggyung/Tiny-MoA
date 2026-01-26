"""
Cowork File Skills
==================
에이전트가 워크스페이스 내에서 파일을 조작할 수 있도록 돕는 툴셋.
"""

import os
from typing import Dict, Any
from src.tiny_moa.cowork.workspace import WorkspaceContext

class CoworkFileSkill:
    def __init__(self, workspace: WorkspaceContext):
        self.workspace = workspace

    def get_tool_definitions(self) -> list:
        return [
            {
                "name": "workspace_list",
                "description": "List all files in the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recursive": {"type": "boolean", "default": False}
                    }
                }
            },
            {
                "name": "workspace_read",
                "description": "Read the content of a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"}
                    },
                    "required": ["filename"]
                }
            },
            {
                "name": "workspace_write",
                "description": "Write or overwrite a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["filename", "content"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "workspace_list":
            res = self.workspace.list_files(recursive=args.get("recursive", False))
            return {"success": True, "result": res}
        
        elif tool_name == "workspace_read":
            res = self.workspace.read_file(args.get("filename"))
            return {"success": not res.startswith("Error"), "result": res}
            
        elif tool_name == "workspace_write":
            res = self.workspace.write_file(args.get("filename"), args.get("content"))
            return {"success": not res.startswith("Error"), "result": res}
            
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
