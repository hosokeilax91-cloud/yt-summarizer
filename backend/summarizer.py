import os
import re
import json
import tempfile
import asyncio
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


def get_video_info_and_subtitles(video_id: str) -> dict:
    """
    yt-dlp で動画情報と字幕を一括取得。
    youtube_transcript_api よりクラウド環境での信頼性が高い。
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    result = {
        "title": "タイトル不明",
        "channel": "チャンネル不明",
        "transcript": None,
        "lang": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["ja", "en"],
            "subtitlesformat": "json3",
            "skip_download": True,
            "outtmpl": os.path.join(tmpdir, "video"),
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                result["title"] = info.get("title", "タイトル不明")
                result["channel"] = info.get("uploader", "チャンネル不明")

                # yt-dlp が字幕をファイルに書き出すため、再度ダウンロード
                ydl.download([url])
        except Exception as e:
            print(f"[WARN] yt-dlp extract_info failed: {e}")
            # タイトルだけ oEmbed で取得を試みる
            try:
                import urllib.request
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
                with urllib.request.urlopen(oembed_url, timeout=10) as r:
                    data = json.loads(r.read())
                result["title"] = data.get("title", "タイトル不明")
                result["channel"] = data.get("author_name", "チャンネル不明")
            except Exception:
                pass
            return result

        # 字幕ファイルを探して読み取る
        for lang in ["ja", "en"]:
            sub_file = os.path.join(tmpdir, f"video.{lang}.json3")
            if os.path.exists(sub_file):
                try:
                    with open(sub_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    events = data.get("events", [])
                    texts = []
                    for event in events:
                        for seg in event.get("segs", []):
                            t = seg.get("utf8", "").strip()
                            if t and t != "\n":
                                texts.append(t)
                    if texts:
                        result["transcript"] = " ".join(texts)
                        result["lang"] = lang
                        return result
                except Exception as e:
                    print(f"[WARN] json3 parse error for {lang}: {e}")
                    continue

        # json3 が無ければ vtt を試す
        for lang in ["ja", "en"]:
            sub_file = os.path.join(tmpdir, f"video.{lang}.vtt")
            if os.path.exists(sub_file):
                try:
                    with open(sub_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    lines = content.splitlines()
                    texts = []
                    for line in lines:
                        line = line.strip()
                        if (line and not line.startswith("WEBVTT")
                            and "-->" not in line
                            and not line.isdigit()
                            and not line.startswith("NOTE")):
                            # HTMLタグを除去
                            clean = re.sub(r"<[^>]+>", "", line)
                            if clean.strip():
                                texts.append(clean.strip())
                    if texts:
                        result["transcript"] = " ".join(texts)
                        result["lang"] = lang
                        return result
                except Exception as e:
                    print(f"[WARN] vtt parse error for {lang}: {e}")
                    continue

    return result


def fetch_transcript_legacy(video_id: str) -> tuple:
    """
    レガシー: youtube_transcript_api を使った字幕取得（フォールバック）
    """
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

        # mp3ファイルを探す
        audio_files = [f for f in os.listdir(tmpdir)
                       if f.endswith((".mp3", ".m4a", ".webm", ".opus"))]
        if not audio_files:
            raise RuntimeError("音声ファイルのダウンロードに失敗しました")

        audio_path = os.path.join(tmpdir, audio_files[0])

        # ファイルサイズチェック（Whisper APIの25MB制限）
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

    # Step 1: yt-dlp で動画情報＋字幕を一括取得
    info = get_video_info_and_subtitles(video_id)
    transcript = info["transcript"]
    lang = info["lang"]

    # Step 2: yt-dlp で字幕取得できなかった場合 → youtube_transcript_api
    if not transcript:
        print("[INFO] yt-dlp字幕取得失敗 → youtube_transcript_apiを試行")
        transcript, lang = fetch_transcript_legacy(video_id)

    # Step 3: それでも取得できない場合 → Whisperフォールバック
    whisper_used = False
    if not transcript:
        print("[INFO] 字幕取得失敗 → Whisperフォールバック")
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
                f"字幕取得・音声文字起こしの両方に失敗しました: {e}"
            )

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
