from fastapi import FastAPI, WebSocket
import asyncio
from whisprx_sdk.ws_server import MessagePackConnection  # assuming exists

app = FastAPI(title="WhispRX API", version="0.1.0")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    conn = MessagePackConnection(websocket)
    await conn.handle()
