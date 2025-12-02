import asyncio
import re
from typing import Dict, List, Optional, Callable, Any
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CreateMessageResult



# Since sse_client is a context manager, we need a way to manage multiple of them.
# A common pattern is to have a background task that manages the lifecycle.

from mcp.client.streamable_http import streamablehttp_client

import time

class PersistentConnection:
    def __init__(self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None, transport: str = "sse", sampling_callback: Optional[Callable[[Any], Any]] = None):
        self.server_name = server_name
        self.url = url
        self.headers = headers or {}
        self.transport = transport
        self.sampling_callback = sampling_callback
        self.session: Optional[ClientSession] = None
        self._exit_stack = None
        
        # Caching
        self.tools_cache = None
        self.tools_cache_timestamp = 0
        
        self.resources_cache = None
        self.resources_cache_timestamp = 0
        
        self.prompts_cache = None
        self.prompts_cache_timestamp = 0
        
        self.CACHE_TTL = 300  # 5 minutes

    async def start(self):
        # Exponential backoff parameters
        backoff_delay = 1
        max_backoff = 60
        
        while True:
            try:
                if self.transport == "http":
                    # HTTP Transport (Streamable HTTP)
                    async with streamablehttp_client(self.url, headers=self.headers) as (read, write, _):
                        async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                            self.session = session
                            print(f"Connected to {self.server_name} via HTTP")
                            await session.initialize()
                            
                            # Reset backoff on successful connection
                            backoff_delay = 1
                            
                            while True:
                                await asyncio.sleep(1)
                else:
                    # SSE Transport
                    # Set timeout to None to prevent read timeouts on long-lived connections
                    async with sse_client(self.url, headers=self.headers, timeout=None) as (read, write):
                        async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                            self.session = session
                            print(f"Connected to {self.server_name} via SSE")
                            await session.initialize()
                            
                            # Reset backoff on successful connection
                            backoff_delay = 1
                            
                            while True:
                                await asyncio.sleep(1)
            except Exception as e:
                import traceback
                # Only print full traceback for unexpected errors, keep logs cleaner for connection issues
                # traceback.print_exc() 
                print(f"Connection error for {self.server_name}: {e}")
                self.session = None
                # Clear cache on disconnection
                self.tools_cache = None
                self.resources_cache = None
                self.prompts_cache = None
                
                print(f"Reconnecting to {self.server_name} in {backoff_delay} seconds...")
                await asyncio.sleep(backoff_delay)
                
                # Increase backoff
                backoff_delay = min(backoff_delay * 2, max_backoff)

    async def get_tools(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.tools_cache and (current_time - self.tools_cache_timestamp < self.CACHE_TTL):
            return self.tools_cache
            
        try:
            result = await self.session.list_tools()
            self.tools_cache = [t.model_dump() for t in result.tools]
            self.tools_cache_timestamp = current_time
            return self.tools_cache
        except Exception as e:
            print(f"Error listing tools for {self.server_name}: {e}")
            return []

    async def get_resources(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.resources_cache and (current_time - self.resources_cache_timestamp < self.CACHE_TTL):
            return self.resources_cache
            
        try:
            result = await self.session.list_resources()
            self.resources_cache = [r.model_dump() for r in result.resources]
            self.resources_cache_timestamp = current_time
            return self.resources_cache
        except Exception as e:
            print(f"Error listing resources for {self.server_name}: {e}")
            return []

    async def get_prompts(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.prompts_cache and (current_time - self.prompts_cache_timestamp < self.CACHE_TTL):
            return self.prompts_cache
            
        try:
            result = await self.session.list_prompts()
            self.prompts_cache = [p.model_dump() for p in result.prompts]
            self.prompts_cache_timestamp = current_time
            return self.prompts_cache
        except Exception as e:
            print(f"Error listing prompts for {self.server_name}: {e}")
            return []

import json
import os

CONFIG_FILE = "/app/data/mcp_config.json"

class GlobalConnectionManager:
    def __init__(self):
        self.connections: Dict[str, PersistentConnection] = {}
        self.sampling_callback: Optional[Callable[[Any], Any]] = None
        # Defer loading config until we have the callback, or just load it and update later?
        # Better to load it, but we can't start connections without the callback if we want them to have it.
        # So we'll load config but not start connections yet? 
        # Or we can just set the callback later and restart connections?
        # For simplicity, let's allow setting the callback and then loading/reloading.
        
    def set_sampling_callback(self, callback: Callable[[Any], Any]):
        self.sampling_callback = callback
        # If we already have connections, we might need to restart them to pick up the callback.
        # But usually this is called at startup before adding servers.

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
        # Stop existing if any
        if server_name in self.connections:
            # Ideally we should stop it, but we don't have a stop method exposed easily yet.
            # The loop in start() runs forever. We'd need to cancel the task.
            # For now, we'll just overwrite it, leaving the old one potentially running (leak).
            # TODO: Fix connection cleanup.
            pass

        connection = PersistentConnection(server_name, url, headers, transport, sampling_callback=self.sampling_callback)
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
            if conn:
                tools.extend(await conn.get_tools())
        else:
            for name, conn in self.connections.items():
                tools.extend(await conn.get_tools())
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
