# Jarvis - MCP Client

Jarvis is a web-based client for the Model Context Protocol (MCP). It allows you to connect to various MCP servers and interact with them using natural language through a chat interface.

## Features

-   **MCP Client:** Connect to any standard MCP server via SSE (Server-Sent Events).
-   **LLM Integration:** Powered by OpenAI `gpt-4o-mini` (configurable in the UI).
-   **Tool Use:** Automatically discovers and utilizes tools provided by connected MCP servers.
-   **Modern UI:** Built with React, Vite, and TailwindCSS.
-   **Dockerized:** Easy deployment with Docker Compose.

## Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your machine.
-   A [Google Gemini API Key](https://aistudio.google.com/app/apikey).

## Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd Jarvis
    ```

2.  **Start the application:**

    Run the following command to build and start the services:

    ```bash
    docker-compose up --build
    ```

    This will:
    -   Build the frontend (React/Vite).
    -   Set up the backend (FastAPI).
    -   Start the server on port 3000.

3.  **Access the application:**

    Open your browser and navigate to:

    [http://localhost:3000](http://localhost:3000)

## Configuration

### API Key Setup
1.  When you first load the application, you will be prompted to enter your **OpenAI API Key**.
2.  Enter the key and click "Save".
3.  The key is stored locally in `data/secrets.json` (mapped to your host machine).

### Connecting to MCP Servers
1.  Use the "Connect Server" interface to add MCP servers.
2.  Provide a name and the URL of the MCP server (e.g., `http://localhost:8000/sse`).
3.  The application will connect and discover available tools.

## Development

If you want to run the application locally without Docker:

### Backend

1.  Ensure you have **Python 3.11+** installed.

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Run the server:
    ```bash
    uvicorn backend.main:app --reload --port 3000
    ```

### Frontend

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Run the development server:
    ```bash
    npm run dev
    ```

## Project Structure

-   `backend/`: FastAPI backend application.
-   `frontend/`: React frontend application.
-   `data/`: Persistent storage for configuration and secrets.
-   `Dockerfile`: Multi-stage Docker build.
-   `docker-compose.yml`: Docker Compose configuration.
