import os
import httpx
from datetime import datetime, timezone

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_DB_ID = os.environ.get("NOTION_DB_ID", "")
NOTION_VERSION = "2022-06-28"

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION,
}

def markdown_to_notion_blocks(md: str) -> list:
    """
    マークダウンテキストをNotionブロックに変換する。
    100ブロック制限に対応するため分割考慮済み。
    """
    blocks = []
    lines = md.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # H2見出し
        if line.startswith("## "):
            text = line[3:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                }
            })

        # H3見出し
        elif line.startswith("### "):
            text = line[4:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                }
            })

        # 箇条書き
        elif line.startswith("- "):
            text = line[2:].strip()
            # **bold**を処理
            rich_text = parse_inline_bold(text)
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": rich_text}
            })

        # 水平線
        elif line.strip() == "---":
            blocks.append({
                "object": "block",
                "type": "divider",
                "divider": {}
            })

        # 空行はスキップ
        elif line.strip() == "":
            pass

        # 通常テキスト
        else:
            rich_text = parse_inline_bold(line.strip())
            if rich_text:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": rich_text}
                })

        i += 1

    return blocks

def parse_inline_bold(text: str) -> list:
    """**bold**をNotionのrich_textに変換"""
    import re
    parts = re.split(r"(\*\*.*?\*\*)", text)
    rich_text = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            content = part[2:-2]
            rich_text.append({
                "type": "text",
                "text": {"content": content},
                "annotations": {"bold": True}
            })
        elif part:
            rich_text.append({
                "type": "text",
                "text": {"content": part}
            })
    return rich_text

async def create_notion_page(data: dict) -> str:
    """Notionに新規ページを作成してURLを返す"""

    blocks = markdown_to_notion_blocks(data["summary"])

    # Notionは1リクエスト100ブロックまでなので先頭99個だけ初回送信
    first_blocks = blocks[:99]
    remaining_blocks = blocks[99:]

    # タグをmulti_selectに
    tags_prop = [{"name": t} for t in data.get("tags", [])]

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "icon": {"emoji": "🎬"},
        "properties": {
            "タイトル": {
                "title": [{"text": {"content": data["title"]}}]
            },
            "URL": {
                "url": data["url"]
            },
            "著者/チャンネル": {
                "rich_text": [{"text": {"content": data["channel"]}}]
            },
        },
        "children": first_blocks
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.notion.com/v1/pages",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
        if res.status_code != 200:
            raise RuntimeError(f"Notion API エラー: {res.status_code} {res.text}")

        page = res.json()
        page_id = page["id"]
        page_url = page["url"]

        # 残りのブロックを追加（100ブロック超の場合）
        if remaining_blocks:
            for chunk_start in range(0, len(remaining_blocks), 99):
                chunk = remaining_blocks[chunk_start:chunk_start + 99]
                await client.patch(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=HEADERS,
                    json={"children": chunk},
                    timeout=30,
                )

    return page_url

async def save_to_notion(data: dict) -> str:
    if not NOTION_TOKEN or not NOTION_DB_ID:
        raise RuntimeError("NOTION_TOKEN または NOTION_DB_ID が設定されていません")
    return await create_notion_page(data)
