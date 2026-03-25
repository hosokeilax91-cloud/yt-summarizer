from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from summarizer import summarize_video
from notion_writer import save_to_notion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    url: str
    save_to_notion: bool = True

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    try:
        result = await summarize_video(req.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"要約エラー: {str(e)}")

    notion_url = None
    if req.save_to_notion:
        try:
            notion_url = await save_to_notion(result)
        except Exception as e:
            notion_url = None
            result["notion_error"] = str(e)

    result["notion_url"] = notion_url
    return result

# フロントエンドの静的ファイルを配信
STATIC_DIR = os.path.join(os.path.dirname(__file__), "../frontend/public")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
