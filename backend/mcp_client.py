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
        self._task: Optional[asyncio.Task] = None
        
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
                            
                            # Prefetch tools to populate cache
                            asyncio.create_task(self.get_tools())
                            
                            # Reset backoff on successful connection
                            backoff_delay = 1
                            
                            while True:
                                await asyncio.sleep(1)
                else:
                    # SSE Transport
                    # Set timeout to None to prevent read timeouts on long-lived connections
                    
                    import logging
                    import sys
                    
                    async def log_response_body(response):
                        print(f"HOOK CALLED for {response.url}", flush=True)
                        if response.status_code >= 400:
                            body = await response.aread()
                            print(f"Response Body for {response.url}: {body}", flush=True)
                            logging.error(f"Response Body for {response.url}: {body}")

                    def custom_client_factory(headers, auth, timeout):
                        # Enforce infinite timeout for SSE connections to prevent read timeouts
                        # httpx defaults to 5s if not specified, and sometimes timeout=None isn't enough depending on context
                        return httpx.AsyncClient(
                            headers=headers, 
                            auth=auth, 
                            timeout=httpx.Timeout(None, connect=5.0),
                            event_hooks={'response': [log_response_body]}
                        )

                    async with sse_client(self.url, headers=self.headers, timeout=None, httpx_client_factory=custom_client_factory) as (read, write):
                        async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                            self.session = session
                            print(f"Connected to {self.server_name} via SSE")
                            await session.initialize()
                            
                            # Prefetch tools to populate cache
                            asyncio.create_task(self.get_tools())
                            
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

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            print(f"Stopped connection to {self.server_name}")

    async def get_tools(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.tools_cache and (current_time - self.tools_cache_timestamp < self.CACHE_TTL):
            return self.tools_cache
            
        try:
            result = await self.session.list_tools()
            self.tools_cache = []
            for t in result.tools:
                tool_dump = t.model_dump()
                # Prefix tool name with server name
                tool_dump['name'] = f"{self.server_name}__{t.name}"
                self.tools_cache.append(tool_dump)
                
            self.tools_cache_timestamp = current_time
            return self.tools_cache
        except Exception as e:
            import traceback
            traceback.print_exc()
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
SECRETS_FILE = "/app/data/secrets.json"

class GlobalConnectionManager:
    def __init__(self):
        self.connections: Dict[str, PersistentConnection] = {}
        self.sampling_callback: Optional[Callable[[Any], Any]] = None
        self.gemini_api_key: Optional[str] = None
        
    def set_sampling_callback(self, callback: Callable[[Any], Any]):
        self.sampling_callback = callback
        # If we already have connections, we might need to restart them to pick up the callback.
        # But usually this is called at startup before adding servers.

    async def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    for name, details in config.get("mcpServers", {}).items():
                        await self.add_server(
                            name, 
                            details["url"], 
                            details.get("headers"), 
                            transport=details.get("transport", "sse"),
                            save=False
                        )
                print(f"Loaded config from {CONFIG_FILE}")
            except Exception as e:
                print(f"Failed to load config: {e}")

        # Load secrets
        if os.path.exists(SECRETS_FILE):
            try:
                with open(SECRETS_FILE, 'r') as f:
                    secrets = json.load(f)
                    self.gemini_api_key = secrets.get("geminiApiKey")
                print(f"Loaded secrets from {SECRETS_FILE}")
            except Exception as e:
                print(f"Failed to load secrets: {e}")

    def save_config(self):
        # Save MCP Servers
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

        # Save Secrets
        secrets = {"geminiApiKey": self.gemini_api_key}
        try:
            with open(SECRETS_FILE, 'w') as f:
                json.dump(secrets, f, indent=2)
            print(f"Saved secrets to {SECRETS_FILE}")
        except Exception as e:
            print(f"Failed to save secrets: {e}")

    async def add_server(self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None, transport: str = "sse", save: bool = True):
        # Stop existing if any
        if server_name in self.connections:
            print(f"Stopping existing connection for {server_name}...")
            await self.connections[server_name].stop()

        connection = PersistentConnection(server_name, url, headers, transport, sampling_callback=self.sampling_callback)
        self.connections[server_name] = connection
        connection._task = asyncio.create_task(connection.start())
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
    Scans for @{server_name} or @server_name in the message.
    Returns the server_name if found, else None.
    """
    # Matches @{name} or @name, preceded by start of string or whitespace
    match = re.search(r'(?:^|\s)@(?:\{([a-zA-Z0-9_-]+)\}|([a-zA-Z0-9_-]+))', message)
    if match:
        return match.group(1) or match.group(2)
    return None

# Singleton instance
connection_manager = GlobalConnectionManager()
