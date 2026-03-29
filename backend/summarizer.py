import os
import re
import json
import html
import tempfile
import asyncio
import httpx
import anthropic

client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Supadata API（YouTube字幕取得の専用サービス、無料枠あり）
SUPADATA_API_URL = "https://api.supadata.ai/v1/youtube/transcript"


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


async def fetch_transcript_supadata(video_id: str) -> tuple:
    """
    Supadata API経由でYouTube字幕を取得。
    専用の字幕取得サービスのため、クラウドサーバーでも確実に動作する。
    """
    api_key = os.environ.get("SUPADATA_API_KEY", "")
    if not api_key:
        print("[WARN] SUPADATA_API_KEY が設定されていません")
        return None, None

    async with httpx.AsyncClient(timeout=60) as http:
        for lang in ["ja", "en"]:
            try:
                resp = await http.get(
                    SUPADATA_API_URL,
                    params={"videoId": video_id, "lang": lang},
                    headers={"x-api-key": api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content")
                    if content and isinstance(content, list):
                        text = " ".join(
                            entry.get("text", "") for entry in content
                            if entry.get("text")
                        )
                        if text.strip():
                            print(f"[INFO] Supadata APIで字幕取得成功: {lang}")
                            return text, lang
                    elif content and isinstance(content, str):
                        if content.strip():
                            print(f"[INFO] Supadata APIで字幕取得成功: {lang}")
                            return content, lang
                else:
                    print(f"[WARN] Supadata API ({lang}): status {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                print(f"[WARN] Supadata API ({lang}) failed: {e}")
                continue

    return None, None


async def fetch_video_info(video_id: str) -> dict:
    """oEmbed APIで動画情報を取得（YouTube公式、ブロックされにくい）"""
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


async def download_audio_pytubefix(video_id: str, tmpdir: str) -> str:
    """pytubefix で音声をダウンロード（クラウドサーバー対応）"""
    from pytubefix import YouTube

    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).order_by("abr").first()
    if not stream:
        raise RuntimeError("音声ストリームが見つかりません")

    audio_path = stream.download(output_path=tmpdir, filename="audio")
    print(f"[INFO] pytubefix: 音声ダウンロード完了 ({os.path.getsize(audio_path) / 1024 / 1024:.1f}MB)")
    return audio_path


async def download_audio_ytdlp(video_id: str, tmpdir: str) -> str:
    """yt-dlp で音声をダウンロード（フォールバック）"""
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_files = [f for f in os.listdir(tmpdir)
                   if f.endswith((".mp3", ".m4a", ".webm", ".opus", ".mp4"))]
    if not audio_files:
        raise RuntimeError("音声ファイルのダウンロードに失敗しました")

    return os.path.join(tmpdir, audio_files[0])


async def transcribe_with_assemblyai(video_id: str) -> str:
    """音声DL → AssemblyAI APIで文字起こし（月100時間無料）"""
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY が設定されていません")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = None

        # 方法1: pytubefix（クラウドサーバーで動作しやすい）
        try:
            print("[INFO] pytubefix で音声ダウンロード試行...")
            audio_path = await download_audio_pytubefix(video_id, tmpdir)
        except Exception as e:
            print(f"[WARN] pytubefix 失敗: {e}")

        # 方法2: yt-dlp（フォールバック）
        if not audio_path:
            try:
                print("[INFO] yt-dlp で音声ダウンロード試行...")
                audio_path = await download_audio_ytdlp(video_id, tmpdir)
            except Exception as e:
                print(f"[WARN] yt-dlp 失敗: {e}")
                raise RuntimeError(
                    "この動画の音声をダウンロードできませんでした。"
                    "字幕がなく、音声取得もブロックされています。"
                )

        file_size = os.path.getsize(audio_path)
        print(f"[INFO] AssemblyAI で文字起こし中... ({file_size / 1024 / 1024:.1f}MB)")

        headers = {"authorization": api_key}

        async with httpx.AsyncClient(timeout=300) as http:
            # Step 1: 音声ファイルをアップロード
            print("[INFO] AssemblyAI: 音声アップロード中...")
            with open(audio_path, "rb") as f:
                upload_res = await http.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    content=f.read(),
                )
            if upload_res.status_code != 200:
                raise RuntimeError(f"AssemblyAI アップロードエラー: {upload_res.status_code}")
            audio_url = upload_res.json()["upload_url"]

            # Step 2: 文字起こしリクエスト（speech_models パラメータ必須）
            print("[INFO] AssemblyAI: 文字起こしリクエスト送信...")
            transcript_res = await http.post(
                "https://api.assemblyai.com/v2/transcript",
                headers={**headers, "content-type": "application/json"},
                json={
                    "audio_url": audio_url,
                    "speech_models": ["universal-2"],
                    "language_detection": True,
                },
            )
            if transcript_res.status_code != 200:
                raise RuntimeError(f"AssemblyAI リクエストエラー: {transcript_res.status_code} {transcript_res.text[:200]}")
            transcript_id = transcript_res.json()["id"]

            # Step 3: 結果をポーリング
            print("[INFO] AssemblyAI: 文字起こし処理中...")
            while True:
                poll_res = await http.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers,
                )
                poll_data = poll_res.json()
                status = poll_data["status"]

                if status == "completed":
                    text = poll_data.get("text") or ""
                    if not text.strip():
                        raise RuntimeError(
                            "音声の文字起こし結果が空でした。"
                            "この動画はBGMのみ、または音声が少ないため要約できません。"
                        )
                    print(f"[INFO] AssemblyAI: 文字起こし完了! ({len(text)}文字)")
                    return text
                elif status == "error":
                    raise RuntimeError(f"AssemblyAI エラー: {poll_data.get('error', '不明')}")

                await asyncio.sleep(3)


async def check_transcript_available(url: str) -> dict:
    """字幕があるかどうかチェック（フロントから確認用）"""
    video_id = extract_video_id(url)
    info = await fetch_video_info(video_id)

    # Supadata で字幕チェック
    transcript, lang = await fetch_transcript_supadata(video_id)
    if transcript:
        return {"has_transcript": True, "method": "subtitle", "title": info["title"], "channel": info["channel"]}

    # youtube_transcript_api でチェック
    transcript, lang = fetch_transcript_legacy(video_id)
    if transcript:
        return {"has_transcript": True, "method": "subtitle", "title": info["title"], "channel": info["channel"]}

    return {"has_transcript": False, "method": "assemblyai", "title": info["title"], "channel": info["channel"]}


SUMMARIZE_PROMPT = """あなたはYouTube動画の内容を分かりやすく要約する専門家です。
以下のトランスクリプトを読んで、日本語で構造化された要約を作成してください。

【重要な注意点】
- トランスクリプトは自動生成された字幕であり、話者の識別情報は含まれていません。
- 話者が誰であるか断定的に書かないでください。「動画では〜と説明されている」のように書いてください。
- 複数の話者がいる場合は「出演者が〜」「対談の中で〜」のように一般的に記述してください。
- 固有名詞や人名は、トランスクリプトに明確に含まれている場合のみ使用してください。

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

    # Step 1: 動画情報を取得（oEmbed API）
    info = await fetch_video_info(video_id)

    # Step 2: Supadata APIで字幕を取得（メイン - 最も確実）
    print(f"[INFO] Step 1: Supadata APIで字幕取得を試行...")
    transcript, lang = await fetch_transcript_supadata(video_id)

    # Step 3: youtube_transcript_api（フォールバック1）
    if not transcript:
        print(f"[INFO] Step 2: youtube_transcript_apiで字幕取得を試行...")
        transcript, lang = fetch_transcript_legacy(video_id)

    # Step 4: AssemblyAI音声文字起こし（フォールバック2 - 月100時間無料）
    whisper_used = False
    if not transcript:
        print(f"[INFO] Step 3: AssemblyAIで音声文字起こしを試行...")
        if not os.environ.get("ASSEMBLYAI_API_KEY"):
            raise RuntimeError(
                "この動画の字幕を取得できませんでした。"
                "音声文字起こし用のAPIキーが未設定です。"
            )
        try:
            transcript = await transcribe_with_assemblyai(video_id)
            whisper_used = True
        except Exception as e:
            raise RuntimeError(
                f"この動画は要約できませんでした。字幕がなく、音声の文字起こしにも失敗しました。"
                f"ミュージックビデオなど音声が主に音楽の動画は対応できません。（詳細: {e}）"
            )

    # 字幕が取得できたが内容が短すぎる場合
    if transcript and len(transcript.strip()) < 50:
        raise RuntimeError(
            "この動画の字幕は取得できましたが、内容が短すぎるため要約できません。"
            "ミュージックビデオや歌詞のみの動画は対応していません。"
        )

    # 長すぎる場合はトリミング（Claude APIのコンテキストに収まるように）
    max_chars = 100000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n...(長すぎるため以降省略)"

    # Claude APIで要約（タイムアウト対策で長めに設定）
    print(f"[INFO] Claude APIで要約中... (トランスクリプト: {len(transcript)}文字)")
    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
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

    # 3行サマリー抽出（「要点」列用）
    three_line_summary = ""
    if "## 📌 3行サマリー" in summary:
        section = summary.split("## 📌 3行サマリー")[1]
        # 次の見出し（##）までを取得
        if "## " in section[1:]:
            section = section[:section.index("## ", 1)]
        lines_list = [l.strip().lstrip("- ").strip() for l in section.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
        three_line_summary = "\n".join(lines_list)

    return {
        "video_id": video_id,
        "url": url,
        "title": info["title"],
        "channel": info["channel"],
        "summary": summary,
        "three_line_summary": three_line_summary,
        "tags": tags,
        "whisper_used": whisper_used,
        "transcript_lang": lang if not whisper_used else "whisper",
    }
