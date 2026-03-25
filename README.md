# 🎬 YT Summarizer

YouTube動画のURLを貼るだけで、AIが要約してNotionに自動保存するWebアプリ。

---

## 構成

```
yt-summarizer/
├── backend/
│   ├── main.py            # FastAPI サーバー
│   ├── summarizer.py      # 字幕取得 + Claude API 要約
│   ├── notion_writer.py   # Notion API 書き込み
│   └── requirements.txt
├── frontend/
│   └── public/
│       └── index.html     # シングルページUI
└── render.yaml            # Render デプロイ設定
```

---

## 必要な APIキー

| 変数名 | 用途 | 取得先 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude 要約 | console.anthropic.com |
| `NOTION_TOKEN` | Notion保存 | notion.so → Settings → Integrations |
| `NOTION_DB_ID` | 保存先DB | NotionのDBページURLから取得 |
| `OPENAI_API_KEY` | Whisper文字起こし（字幕なし動画用） | platform.openai.com |

---

## Notionデータベースの準備

以下のプロパティを持つデータベースを作成してください：

| プロパティ名 | タイプ |
|---|---|
| Name | タイトル |
| URL | URL |
| チャンネル | テキスト |
| タグ | マルチセレクト |
| 要約日 | 日付 |
| 文字起こし方法 | セレクト |

作成後、IntegrationをDBに接続（DBページ右上 → 接続 → 作成したIntegration）。

---

## Renderへのデプロイ手順

1. GitHubにこのリポジトリをpush
2. [render.com](https://render.com) でNew → Web Service
3. GitHubリポジトリを接続
4. 環境変数を設定（上記4つ）
5. Deploy → 完了後、発行されたURLからアクセス

---

## ローカルで動かす場合

```bash
cd backend
pip install -r requirements.txt

# .envファイルを作成
ANTHROPIC_API_KEY=sk-ant-...
NOTION_TOKEN=secret_...
NOTION_DB_ID=xxxxxxxx...
OPENAI_API_KEY=sk-...  # 任意

uvicorn main:app --reload --port 8000
```

ブラウザで http://localhost:8000 を開く。

---

## 機能

- ✅ YouTube字幕（日本語・英語）自動取得
- ✅ 字幕なし動画はWhisperで音声文字起こし
- ✅ Claude AIによる構造化要約（3行サマリー・章立て・キーワード・タグ）
- ✅ Notionに動画ごとに新規ページ作成
- ✅ Notionへの保存ON/OFFトグル
- ✅ スマホ・PCどちらでも動作
