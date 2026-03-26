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


def safe_rich_text(text: str, max_len: int = 2000) -> list:
    """
    Notion rich_textの2000文字制限に対応。
    長いテキストは複数のrich_textに分割する。
    """
    if not text:
        return [{"type": "text", "text": {"content": ""}}]

    chunks = []
    for i in range(0, len(text), max_len):
        chunks.append({
            "type": "text",
            "text": {"content": text[i:i + max_len]}
        })
    return chunks


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
                    "rich_text": safe_rich_text(text)
                }
            })

        # H3見出し
        elif line.startswith("### "):
            text = line[4:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": safe_rich_text(text)
                }
            })

        # 箇条書き
        elif line.startswith("- "):
            text = line[2:].strip()
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
    """**bold**をNotionのrich_textに変換（2000文字制限対応）"""
    import re
    parts = re.split(r"(\*\*.*?\*\*)", text)
    rich_text = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            content = part[2:-2]
            # 2000文字を超える場合は分割
            for i in range(0, len(content), 2000):
                rich_text.append({
                    "type": "text",
                    "text": {"content": content[i:i + 2000]},
                    "annotations": {"bold": True}
                })
        elif part:
            for i in range(0, len(part), 2000):
                rich_text.append({
                    "type": "text",
                    "text": {"content": part[i:i + 2000]}
                })
    return rich_text


async def create_notion_page(data: dict) -> str:
    """Notionに新規ページを作成してURLを返す"""

    blocks = markdown_to_notion_blocks(data["summary"])

    # Notionは1リクエスト100ブロックまでなので先頭99個だけ初回送信
    first_blocks = blocks[:99]
    remaining_blocks = blocks[99:]

    # タイトルが長すぎる場合は切り詰め（Notionのtitleは2000文字制限）
    title = data.get("title", "タイトル不明")[:2000]
    channel = data.get("channel", "チャンネル不明")[:2000]

    # 3行サマリー（要点列用）
    three_line_summary = data.get("three_line_summary", "")[:2000]

    # 現在の日付（JST）
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "icon": {"emoji": "🎬"},
        "properties": {
            "タイトル": {
                "title": [{"text": {"content": title}}]
            },
            "URL": {
                "url": data["url"]
            },
            "著者/チャンネル": {
                "rich_text": [{"text": {"content": channel}}]
            },
            "要点": {
                "rich_text": [{"text": {"content": three_line_summary}}]
            },
            "読了/視聴日": {
                "date": {"start": now_date}
            },
            "種別": {
                "select": {"name": "動画"}
            },
        },
        "children": first_blocks
    }

    # タイムアウトを長めに設定（長い動画の要約は大量のブロックになるため）
    async with httpx.AsyncClient(timeout=120) as client:
        print(f"[INFO] Notion API: ページ作成中... (ブロック数: {len(blocks)})")
        res = await client.post(
            "https://api.notion.com/v1/pages",
            headers=HEADERS,
            json=payload,
            timeout=120,
        )
        if res.status_code != 200:
            error_detail = res.text[:500]
            print(f"[ERROR] Notion API: {res.status_code} {error_detail}")
            raise RuntimeError(f"Notion API エラー: {res.status_code} {error_detail}")

        page = res.json()
        page_id = page["id"]
        page_url = page["url"]
        print(f"[INFO] Notion ページ作成成功: {page_url}")

        # 残りのブロックを追加（100ブロック超の場合）
        if remaining_blocks:
            print(f"[INFO] Notion API: 残りブロック追加中... ({len(remaining_blocks)}ブロック)")
            for chunk_start in range(0, len(remaining_blocks), 99):
                chunk = remaining_blocks[chunk_start:chunk_start + 99]
                append_res = await client.patch(
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    headers=HEADERS,
                    json={"children": chunk},
                    timeout=120,
                )
                if append_res.status_code != 200:
                    print(f"[WARN] Notion追加ブロック失敗: {append_res.status_code}")

    return page_url


async def save_to_notion(data: dict) -> str:
    if not NOTION_TOKEN or not NOTION_DB_ID:
        raise RuntimeError("NOTION_TOKEN または NOTION_DB_ID が設定されていません")

    # DB IDのフォーマット確認（ハイフンなしの場合はそのまま使える）
    db_id = NOTION_DB_ID.strip()
    print(f"[INFO] Notion DB ID: {db_id[:8]}... (length: {len(db_id)})")

    return await create_notion_page(data)
