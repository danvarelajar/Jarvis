import asyncio
import re
from typing import Dict, List, Optional
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client



# Since sse_client is a context manager, we need a way to manage multiple of them.
# A common pattern is to have a background task that manages the lifecycle.

from mcp.client.streamable_http import streamablehttp_client

class PersistentConnection:
    def __init__(self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None, transport: str = "sse"):
        self.server_name = server_name
        self.url = url
        self.headers = headers or {}
        self.transport = transport
        self.session: Optional[ClientSession] = None
        self._exit_stack = None

    async def start(self):
        # This would need to run in a loop to handle reconnections
        while True:
            try:
                if self.transport == "http":
                    # HTTP Transport (Streamable HTTP)
                    async with streamablehttp_client(self.url, headers=self.headers) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            self.session = session
                            print(f"Connected to {self.server_name} via HTTP")
                            await session.initialize()
                            while True:
                                await asyncio.sleep(1)
                else:
                    # SSE Transport
                    async with sse_client(self.url, headers=self.headers) as (read, write):
                        async with ClientSession(read, write) as session:
                            self.session = session
                            print(f"Connected to {self.server_name} via SSE")
                            await session.initialize()
                            while True:
                                await asyncio.sleep(1)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Connection error for {self.server_name}: {e}")
                self.session = None
                await asyncio.sleep(5) # Wait before reconnecting

import json
import os

CONFIG_FILE = "/app/data/mcp_config.json"

class GlobalConnectionManager:
    def __init__(self):
        self.connections: Dict[str, PersistentConnection] = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    for name, details in config.get("mcpServers", {}).items():
                        self.add_server(
                            name, 
                            details["url"], 
                            details.get("headers"), 
                            transport=details.get("transport", "sse"),
                            save=False
                        )
                print(f"Loaded config from {CONFIG_FILE}")
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        config = {"mcpServers": {}}
        for name, conn in self.connections.items():
            config["mcpServers"][name] = {
                "url": conn.url,
                "headers": conn.headers,
                "transport": conn.transport
            }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {CONFIG_FILE}")
        except Exception as e:
            print(f"Failed to save config: {e}")

    def add_server(self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None, transport: str = "sse", save: bool = True):
        if server_name not in self.connections:
            connection = PersistentConnection(server_name, url, headers, transport)
            self.connections[server_name] = connection
            asyncio.create_task(connection.start())
            if save:
                self.save_config()

    def get_session(self, server_name: str) -> Optional[ClientSession]:
        conn = self.connections.get(server_name)
        return conn.session if conn else None
    
    def get_all_sessions(self) -> Dict[str, ClientSession]:
        return {name: conn.session for name, conn in self.connections.items() if conn.session}

    async def list_tools(self, server_name: str = None) -> List[dict]:
        tools = []
        if server_name:
            conn = self.connections.get(server_name)
            if conn and conn.session:
                try:
                    result = await conn.session.list_tools()
                    tools.extend([t.model_dump() for t in result.tools])
                except Exception as e:
                    print(f"Error listing tools for {server_name}: {e}")
        else:
            for name, conn in self.connections.items():
                if conn.session:
                    try:
                        result = await conn.session.list_tools()
                        tools.extend([t.model_dump() for t in result.tools])
                    except Exception as e:
                        print(f"Error listing tools for {name}: {e}")
        return tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        conn = self.connections.get(server_name)
        if not conn or not conn.session:
            raise ValueError(f"Server {server_name} not found or not connected")
        
        return await conn.session.call_tool(tool_name, arguments)

# Smart Routing Logic
def parse_server_route(message: str) -> Optional[str]:
    """
    Scans for @{server_name} in the message.
    Returns the server_name if found, else None.
    """
    match = re.search(r'@\{([a-zA-Z0-9_-]+)\}', message)
    if match:
        return match.group(1)
    return None

# Singleton instance
connection_manager = GlobalConnectionManager()
