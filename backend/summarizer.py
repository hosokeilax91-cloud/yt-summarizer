import os
import re
import json
import html
import tempfile
import asyncio
import httpx
import anthropic

client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Invidious公開インスタンス（YouTube直接アクセスを回避）
INVIDIOUS_INSTANCES = [
    "https://vid.puffyan.us",
    "https://inv.tux.pizza",
    "https://invidious.fdn.fr",
    "https://invidious.privacyredirect.com",
    "https://yewtu.be",
]


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("YouTube URLからVideo IDを取得できませんでした")


def parse_xml_captions(xml_text: str) -> str:
    """YouTube字幕XML（timedtext形式）をプレーンテキストに変換"""
    texts = re.findall(r"<text[^>]*>([^<]+)</text>", xml_text)
    if texts:
        texts = [html.unescape(t).strip() for t in texts if t.strip()]
        return " ".join(texts)
    return ""


async def fetch_transcript_invidious(video_id: str) -> tuple:
    """
    Invidious API経由でYouTube字幕を取得。
    YouTubeに直接アクセスしないため、クラウドサーバーでもブロックされにくい。
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
        for instance in INVIDIOUS_INSTANCES:
            try:
                # 字幕一覧を取得
                resp = await http.get(f"{instance}/api/v1/captions/{video_id}")
                if resp.status_code != 200:
                    continue

                captions = resp.json().get("captions", [])

                # 日本語 → 英語の順で取得
                for lang in ["ja", "en"]:
                    for cap in captions:
                        cap_lang = cap.get("language_code", "")
                        if cap_lang.startswith(lang):
                            sub_url = cap.get("url", "")
                            if not sub_url.startswith("http"):
                                sub_url = f"{instance}{sub_url}"

                            sub_resp = await http.get(sub_url)
                            if sub_resp.status_code == 200:
                                text = parse_xml_captions(sub_resp.text)
                                if text:
                                    print(f"[INFO] Invidious ({instance}) で字幕取得成功: {lang}")
                                    return text, lang

            except Exception as e:
                print(f"[WARN] Invidious {instance} failed: {e}")
                continue

    return None, None


async def fetch_video_info_invidious(video_id: str) -> dict:
    """Invidious APIで動画情報を取得"""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
        for instance in INVIDIOUS_INSTANCES:
            try:
                resp = await http.get(f"{instance}/api/v1/videos/{video_id}?fields=title,author")
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "title": data.get("title", "タイトル不明"),
                        "channel": data.get("author", "チャンネル不明"),
                    }
            except Exception:
                continue

    # フォールバック: oEmbed API
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            resp = await http.get(oembed_url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "title": data.get("title", "タイトル不明"),
                    "channel": data.get("author_name", "チャンネル不明"),
                }
    except Exception:
        pass

    return {"title": "タイトル不明", "channel": "チャンネル不明"}


def fetch_transcript_legacy(video_id: str) -> tuple:
    """レガシー: youtube_transcript_api を使った字幕取得"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        for lang in ["ja", "en"]:
            try:
                t = transcript_list.find_transcript([lang])
                entries = t.fetch()
                text = " ".join(e["text"] for e in entries)
                return text, lang
            except Exception:
                continue

        t = transcript_list.find_generated_transcript(["ja", "en"])
        entries = t.fetch()
        text = " ".join(e["text"] for e in entries)
        return text, "auto"
    except Exception as e:
        print(f"[WARN] youtube_transcript_api failed: {e}")
        return None, None


async def transcribe_with_whisper(video_id: str) -> str:
    """yt-dlpで音声DL → OpenAI Whisper APIで文字起こし"""
    import yt_dlp
    import openai

    openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(f"音声ダウンロード失敗: {e}")

        audio_files = [f for f in os.listdir(tmpdir)
                       if f.endswith((".mp3", ".m4a", ".webm", ".opus"))]
        if not audio_files:
            raise RuntimeError("音声ファイルのダウンロードに失敗しました")

        audio_path = os.path.join(tmpdir, audio_files[0])

        file_size = os.path.getsize(audio_path)
        if file_size > 25 * 1024 * 1024:
            raise RuntimeError("音声ファイルが大きすぎます（25MB制限）。短い動画で試してください。")

        with open(audio_path, "rb") as f:
            transcript = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ja",
            )
        return transcript.text


SUMMARIZE_PROMPT = """あなたはYouTube動画の内容を分かりやすく要約する専門家です。
以下のトランスクリプトを読んで、日本語で構造化された要約を作成してください。

【出力形式】
必ず以下の形式で出力してください：

## 📌 3行サマリー
（動画の核心を3行で端的に）

## 🗂️ 章立て要約
（内容を3〜6個のトピックに分けて、それぞれ箇条書きで要点をまとめる）
### 1. [トピック名]
- ポイント1
- ポイント2

### 2. [トピック名]
...

## 💡 重要キーワード
- **キーワード**: 簡潔な説明（1行）

## 🔖 一言タグ
（この動画を表す短いタグを3〜5個、カンマ区切りで）

---
【トランスクリプト】
{transcript}
"""


async def summarize_video(url: str) -> dict:
    video_id = extract_video_id(url)

    # Step 1: Invidious APIで動画情報を取得
    info = await fetch_video_info_invidious(video_id)

    # Step 2: Invidious APIで字幕を取得（メイン）
    print(f"[INFO] Step 1: Invidious APIで字幕取得を試行...")
    transcript, lang = await fetch_transcript_invidious(video_id)

    # Step 3: youtube_transcript_api（フォールバック1）
    if not transcript:
        print(f"[INFO] Step 2: youtube_transcript_apiで字幕取得を試行...")
        transcript, lang = fetch_transcript_legacy(video_id)

    # Step 4: Whisper音声文字起こし（フォールバック2）
    whisper_used = False
    if not transcript:
        print(f"[INFO] Step 3: Whisperで音声文字起こしを試行...")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "この動画の字幕を取得できませんでした。"
                "OPENAI_API_KEYが設定されていないため、音声文字起こしもできません。"
            )
        try:
            transcript = await transcribe_with_whisper(video_id)
            whisper_used = True
        except Exception as e:
            raise RuntimeError(
                f"字幕取得・音声文字起こしのすべてに失敗しました: {e}"
            )

    # 長すぎる場合はトリミング
    max_chars = 80000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n...(長すぎるため以降省略)"

    # Claude APIで要約
    message = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": SUMMARIZE_PROMPT.format(transcript=transcript)
        }]
    )
    summary = message.content[0].text

    # タグ抽出
    tags = []
    for line in summary.splitlines():
        if line.startswith("## 🔖"):
            next_lines = summary.split("## 🔖")[1].strip().splitlines()
            for nl in next_lines:
                nl = nl.strip()
                if nl and not nl.startswith("#"):
                    tags = [t.strip() for t in nl.split(",") if t.strip()]
                    break
            break

    return {
        "video_id": video_id,
        "url": url,
        "title": info["title"],
        "channel": info["channel"],
        "summary": summary,
        "tags": tags,
        "whisper_used": whisper_used,
        "transcript_lang": lang if not whisper_used else "whisper",
    }
