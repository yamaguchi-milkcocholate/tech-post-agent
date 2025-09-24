import datetime as dt
import os
from typing import Any

import requests
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

GITHUB_API = "https://api.github.com/search/repositories"


@tool
def fetch_trend_repo(days: int, per_page: int, top_n: int) -> str:
    """Fetch trending repositories from GitHub."""

    return a

    def build_query(language: str, days: int, use_created: bool) -> str:
        """Search APIで“擬似トレンド”を作るためのクエリを生成"""
        since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        date_field = "created" if use_created else "pushed"
        # 例: language:Python + created:>=2025-09-02
        return f"language:{language} {date_field}:>={since}"

    @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def search_repos(token: str, query: str, per_page: int) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "trend-daily-script",
        }
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
        r = requests.get(GITHUB_API, headers=headers, params=params, timeout=30)
        # Rate limit 時はリトライ
        if r.status_code == 403 and "rate limit" in r.text.lower():
            raise RuntimeError("GitHub API rate limited")
        r.raise_for_status()
        return r.json()

    token = os.getenv("TREND_READ_GITHUB_TOKEN")

    query = build_query(language="Python", days=days, use_created=True)
    data = search_repos(token, query, per_page)
    items = data.get("items", [])

    lines = []
    for i, it in enumerate(items[:top_n], 1):
        lines.append(
            f"{i}. {it['full_name']}  ★{it['stargazers_count']}  {it['html_url']}"
        )
    contents = "\n".join(lines)

    return f"Fetched 'GitHub Trends' and content: {contents}"


a = """
1. Varietyz/Disciplined-AI-Software-Development ★282
   https://github.com/Varietyz/Disciplined-AI-Software-Development
   AIシステムとのソフトウェア開発における構造化されたアプローチを提供し、コードの膨張、アーキテクチャのドリフト、コンテキストの希薄化といった一般的な問題に対処します。
   :脳: AIとの協働を効率化し、品質を向上させるため。 / 難易度★3
   使いどころ: AI支援によるソフトウェア開発 ・コード品質の向上 ・開発プロセスの効率化
   - リポジトリをクローンします。
   - 必要な依存関係をインストールします。
   - 提供されているテンプレートやスクリプトをプロジェクトに統合します。
   This methodology provides a structured approach for collaborating with AI systems on software development projects. It addresses common issues like code bloat, architectural drift, and context dilution through systematic constraints and validation checkpoints.
2. X-Square-Robot/wall-x ★265
   https://github.com/X-Square-Robot/wall-x
   Wall-Xは、Pythonで開発された汎用ロボットの構築を目指すプロジェクトです。Embodied Foundation Modelを活用し、ロボットの知覚と行動を統合的に学習・制御することを目指しています。
   :脳: 汎用ロボット開発の最前線を体験できる。 / 難易度★3
   使いどころ: ロボット開発 ・AI研究 ・教育用ロボット
   - リポジトリをクローンする。
   - 必要な依存関係をインストールする。
   - モデルのトレーニングを開始する。
   Building General-Purpose Robots Based on Embodied Foundation Model
3. yoavf/absolutelyright ★258
   https://github.com/yoavf/absolutelyright
   このリポジトリは、Claude Codeがユーザーの生活選択をどれだけ正当化するかを追跡するシステムを提供します。ウェブサイトのバックエンドとして機能し、ユーザーの選択に対するClaudeの反応を記録します。
   :脳: ユーザーの選択に対するAIの反応を分析したい場合に有用です。 / 難易度★2
   使いどころ: AIによる意思決定分析 ・ユーザー行動の追跡 ・AIの信頼性評価
   - リポジトリをクローンします。
   - 必要な依存関係をインストールします。
   - ウェブサイトをローカルで起動します。
   Claude said I'm absolutely right!
4. BICLab/SpikingBrain-7B ★176
   https://github.com/BICLab/SpikingBrain-7B
   SpikingBrain-7Bは、脳のメカニズムに触発された大規模モデルで、効率的な注意機構、MoEモジュール、スパイクエンコーディングを統合しています。これにより、オープンソースのモデルエコシステムと互換性のある普遍的な変換パイプラインを提供し、2%未満の計算コストで継続的な事前学習を可能にします。
   :脳: 効率的な大規模モデルの事前学習を低コストで実現。 / 難易度★3
   使いどころ: 大規模言語モデルの事前学習 ・効率的な注意機構の開発 ・スパイキングニューラルネットワークの研究
   - リポジトリをクローンします。
   - 必要な依存関係をインストールします。
   - モデルを初期化し、データを準備します。
   - トレーニングを開始します。
5. vibevoice-community/VibeVoice ★160
   https://github.com/vibevoice-community/VibeVoice
   VibeVoiceは、テキストから長時間の会話型音声を生成するオープンソースのTTSモデルです。最大4人のスピーカーによる90分の音声合成が可能で、スケーラビリティやスピーカーの一貫性、自然なターンテイキングの課題に対応しています。
   :脳: 長時間・多人数の会話音声合成が可能。 / 難易度★3
   使いどころ: ポッドキャスト生成 ・オーディオブック制作 ・対話型AIシステムの音声合成
   - リポジトリをクローン: git clone https://github.com/vibevoice-community/VibeVoice.git
   - 依存関係をインストール: pip install -r requirements.txt
   - モデルをダウンロード: python download_model.py
   - 音声合成を実行: python generate_audio.py --text 'テキスト内容'
   Long-form conversational TTS | Community fork
"""
