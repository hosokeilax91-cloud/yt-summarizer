from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from summarizer import summarize_video, check_transcript_available
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

@app.get("/debug-env")
def debug_env():
    """環境変数の設定状況を確認（値は隠す）"""
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NOTION_TOKEN", "NOTION_DB_ID", "SUPADATA_API_KEY"]
    result = {}
    for k in keys:
        val = os.environ.get(k, "")
        if val:
            result[k] = f"SET (starts with: {val[:15]}..., length: {len(val)})"
        else:
            result[k] = "NOT SET"
    return result

@app.get("/debug-notion")
async def debug_notion():
    """Notion接続テスト"""
    import httpx
    token = os.environ.get("NOTION_TOKEN", "")
    db_id = os.environ.get("NOTION_DB_ID", "")

    if not token or not db_id:
        return {"status": "error", "message": "NOTION_TOKEN or NOTION_DB_ID not set"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # データベース情報を取得
        res = await client.get(
            f"https://api.notion.com/v1/databases/{db_id}",
            headers=headers,
        )
        if res.status_code == 200:
            data = res.json()
            props = list(data.get("properties", {}).keys())
            return {
                "status": "ok",
                "database_title": data.get("title", [{}])[0].get("plain_text", "不明"),
                "properties": props,
                "db_id": db_id,
            }
        else:
            return {
                "status": "error",
                "code": res.status_code,
                "message": res.text[:500],
                "db_id": db_id,
            }

class CheckRequest(BaseModel):
    url: str

@app.post("/check-transcript")
async def check_transcript(req: CheckRequest):
    """字幕の有無をチェック（フロントから確認用）"""
    try:
        result = await check_transcript_available(req.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
            print(f"[ERROR] Notion保存失敗: {e}")

    result["notion_url"] = notion_url
    return result

# フロントエンドの静的ファイルを配信
STATIC_DIR = os.path.join(os.path.dirname(__file__), "../frontend/public")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
