import os
import re
import httpx
import tempfile
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import anthropic

client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("YouTube URLからVideo IDを取得できませんでした")

def get_video_info(video_id: str) -> dict:
    """YouTube oEmbed APIで動画タイトル・チャンネル名を取得"""
    try:
        import urllib.request, json
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=10) as r:
            data = json.loads(r.read())
        return {
            "title": data.get("title", "タイトル不明"),
            "channel": data.get("author_name", "チャンネル不明"),
        }
    except Exception:
        return {"title": "タイトル不明", "channel": "チャンネル不明"}

def fetch_transcript(video_id: str) -> tuple[str, str]:
    """
    字幕を取得。日本語→英語→自動生成の順で試みる。
    Returns: (transcript_text, source_language)
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # 日本語優先
        for lang in ["ja", "en"]:
            try:
                t = transcript_list.find_transcript([lang])
                entries = t.fetch()
                text = " ".join(e["text"] for e in entries)
                return text, lang
            except Exception:
                continue

        # 自動生成でも可
        t = transcript_list.find_generated_transcript(["ja", "en"])
        entries = t.fetch()
        text = " ".join(e["text"] for e in entries)
        return text, "auto"

    except (NoTranscriptFound, TranscriptsDisabled):
        return None, None
    except Exception as e:
        # 429 Too Many Requests などのエラー時もWhisperフォールバックへ
        print(f"[WARN] 字幕取得失敗（Whisperフォールバックへ）: {e}")
        return None, None

async def transcribe_with_whisper(video_id: str) -> str:
    """yt-dlpで音声DL → OpenAI Whisper APIで文字起こし"""
    import yt_dlp
    import openai

    openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "quiet": True,
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # mp3ファイルを探す
        mp3_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp3")]
        if not mp3_files:
            raise RuntimeError("音声ファイルのダウンロードに失敗しました")

        audio_path = os.path.join(tmpdir, mp3_files[0])
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
    info = get_video_info(video_id)

    # 字幕取得
    transcript, lang = fetch_transcript(video_id)

    whisper_used = False
    if not transcript:
        # Whisperフォールバック
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("この動画には字幕がなく、OPENAI_API_KEYも設定されていないため文字起こしできません")
        transcript = await transcribe_with_whisper(video_id)
        whisper_used = True

    # 長すぎる場合はトリミング（Claude APIの制限対策）
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
