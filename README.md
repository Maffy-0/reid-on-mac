# Person Re-identification System

YOLO + BoT-SORT + ReIDによる人物再識別システム

## 概要

このシステムは、macOS環境でリアルタイム人物検出・追跡・再識別を行います。ウェブカメラから取得した映像に対して以下の機能を提供します：

- **人物検出**: Ultralytics YOLOによる高精度な人物検出
- **追跡**: BoT-SORTトラッカーによるマルチオブジェクト追跡
- **再識別**: 特徴ベクトル比較による人物再識別
- **音楽再生**: 識別された人物に応じた個別音楽再生
- **ログ記録**: CSV形式での詳細なイベントログ

## システム要件

- macOS (Apple Silicon M1/M2推奨)
- Python 3.9以上
- uv (依存関係管理ツール)
- FFmpeg (音声処理用) ※オプション
- USB/Webカメラ

## インストール手順

### 1. FFmpegのインストール（オプション）

音声機能を使用する場合のみ必要：

```bash
# Homebrewを使用してFFmpegをインストール
brew install ffmpeg
```

### 2. 依存関係のインストール

```bash
# プロジェクトディレクトリに移動
cd /Users/toku/kenkyu/project_MEW

# uvを使用して依存関係をインストール
uv sync
```

### 3. 初期セットアップの実行

プロジェクトの初期化とディレクトリ作成を自動で行います：

```bash
# 自動セットアップ（推奨）
uv run python setup.py
```

手動でディレクトリを作成することも可能です：

```bash
mkdir -p models data audio logs templates
```

## 使用方法

### 基本実行

```bash
# 仮想環境を使用してシステムを起動
uv run python main.py
```

### 人物テンプレートの追加

新しい人物を登録する場合：

```bash
# 人物画像から特徴量テンプレートを作成
uv run python main.py --add-template person_1 /path/to/person1_photo.jpg
uv run python main.py --add-template person_2 /path/to/person2_photo.jpg
```

### 音楽ファイルの配置

各人物に対応する音楽ファイルを配置：

```bash
# audioディレクトリに人物IDと同名のMP3ファイルを配置
cp your_music1.mp3 audio/person_1.mp3
cp your_music2.mp3 audio/person_2.mp3
```

### テンプレート管理

```bash
# 登録されている人物一覧を表示
uv run python main.py --list-templates

# 特定の人物のテンプレート情報を表示
uv run python main.py --template-info person_1

# 人物のすべてのテンプレートを削除
uv run python main.py --remove-person person_1
```

## setup.py について

`setup.py`は初期セットアップ専用のスクリプトです。

### 目的
- 必要なディレクトリの自動作成
- サンプル音楽ファイルの生成
- プロジェクト環境の初期化

### 使用方法

```bash
# プロジェクト初回セットアップ時に実行
uv run python setup.py
```

### 実行される処理
1. **ディレクトリ作成**: `models/`, `data/`, `audio/`, `logs/`, `templates/`
2. **サンプルファイル生成**: `person_1.mp3`, `person_2.mp3`, `person_3.mp3`
3. **環境確認**: 必要な依存関係の確認

### いつ実行する？
- プロジェクトを初めて使用する時
- ディレクトリ構造をリセットしたい時
- サンプルファイルを再生成したい時

## 操作方法

システム実行中は以下のキーボード操作が可能です：

- **q**: システム終了
- **s**: 現在のフレームを保存
- **r**: システム状態をリセット

## 推奨ワークフロー

### 初回セットアップ
```bash
# 1. プロジェクトディレクトリに移動
cd /Users/toku/kenkyu/project_MEW

# 2. 依存関係のインストール
uv sync

# 3. 初期セットアップの実行
uv run python setup.py

# 4. システム要件の確認
uv run python main.py
```

### 人物登録の流れ
```bash
# 1. 人物の写真を準備（JPG/PNG形式）
# 2. テンプレートを追加
uv run python main.py --add-template tanaka_taro /path/to/tanaka_photo.jpg

# 3. 複数の角度・表情の写真も追加（精度向上のため）
uv run python main.py --add-template tanaka_taro /path/to/tanaka_photo2.jpg
uv run python main.py --add-template tanaka_taro /path/to/tanaka_photo3.jpg

# 4. 個人用音楽ファイルを配置
cp tanaka_theme.mp3 audio/tanaka_taro.mp3

# 5. 登録状況を確認
uv run python main.py --template-info tanaka_taro
```

### 日常的な使用
```bash
# システムを起動
uv run python main.py

# カメラの前で人物検出・追跡・再識別が自動実行
# ログは logs/ ディレクトリに自動保存
```

## ファイル構成

```
project_MEW/
├── main.py                 # メインシステム
├── setup.py               # 初期セットアップ
├── install.sh             # 自動インストールスクリプト
├── pyproject.toml         # プロジェクト設定
├── README.md              # このファイル
├── src/                   # コアモジュール
│   ├── __init__.py        # パッケージ初期化
│   ├── config.py          # 設定ファイル
│   ├── logger.py          # ログ管理
│   ├── person_tracker.py  # 人物検出・追跡
│   ├── person_reid.py     # 人物再識別
│   └── audio_manager.py   # 音楽再生管理
├── models/                # YOLOモデル格納
├── data/                  # フレーム画像保存
├── audio/                 # 人物別音楽ファイル
├── logs/                  # ログファイル
│   ├── system.log         # システムログ
│   └── person_log.csv     # 人物イベントログ
└── templates/             # 人物特徴量テンプレート
    └── person_database.pkl
```

## 設定のカスタマイズ

`src/config.py`ファイルで以下の設定を調整可能：

### 検出・追跡設定
```python
YOLO_CONF_THRESHOLD = 0.5    # 検出信頼度閾値
YOLO_IOU_THRESHOLD = 0.45    # IoU閾値
```

### 再識別設定
```python
REID_SIMILARITY_THRESHOLD = 0.7  # 類似度閾値
```

### カメラ設定
```python
CAMERA_INDEX = 0             # カメラインデックス
FRAME_WIDTH = 640            # フレーム幅
FRAME_HEIGHT = 480           # フレーム高さ
```

### 音声設定
```python
AUDIO_VOLUME = 0.8           # 音量レベル
AUDIO_FADE_DURATION = 1.0    # フェード時間
```

## ログファイル

### システムログ (logs/system.log)
システムの動作状況、エラー、警告などを記録

### 人物イベントログ (logs/person_log.csv)
```csv
timestamp,track_id,person_id,event_type,confidence
2025-06-28T10:30:15,1,person_1,entry,0.0
2025-06-28T10:30:16,1,person_1,identified,0.85
2025-06-28T10:30:45,1,person_1,exit,0.0
```

## トラブルシューティング

### カメラが認識されない場合
```bash
# カメラデバイスの確認
ls /dev/video*

# 別のカメラインデックスを試す
# src/config.pyのCAMERA_INDEXを変更
```

### MPS（GPU）が使用されない場合
```bash
# PyTorchのMPS対応確認
uv run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 追跡機能が動作しない場合
```bash
# BoT-SORT追跡に必要なlapパッケージの確認
uv run python -c "import lap; print('LAP available for tracking')"

# lapパッケージがない場合は自動的に検出のみモードに切り替わります
```
```bash
# macOSの音声システムを確認
system_profiler SPAudioDataType

# 音楽ファイルの形式確認（MP3形式である必要があります）
file audio/*.mp3

# 手動でのファイル再生テスト
afplay audio/person_1.mp3
```

### 依存関係の問題
```bash
# 仮想環境の再作成
rm -rf .venv
uv sync
```

## パフォーマンス最適化

### GPU（MPS）の活用
Apple Silicon MacではMetal Performance Shadersが自動的に使用されます。

### YOLOモデルサイズの調整
軽量化が必要な場合は`src/config.py`で：
```python
YOLO_MODEL = "YOLOv8n.pt"  # nano (最軽量)
# YOLO_MODEL = "yolov8s.pt"  # small
# YOLO_MODEL = "yolov8m.pt"  # medium
```

### フレームレートの調整
```python
FPS = 15  # フレームレートを下げて負荷軽減
```

## よくある質問（FAQ）

### Q: setup.pyは毎回実行する必要がありますか？
A: いいえ。初回セットアップ時のみ実行してください。ディレクトリやサンプルファイルが作成されます。

### Q: 同一人物で複数のテンプレートを登録できますか？
A: はい。同じperson_idで複数回`--add-template`を実行することで、複数のテンプレートを登録できます。これにより認識精度が向上します。

### Q: 音楽ファイルは必須ですか？
A: いいえ。音楽ファイルがなくても人物検出・追跡・再識別は動作します。音楽再生は追加機能で、macOSの内蔵音声プレイヤー（afplay）を使用します。

### Q: 音楽が再生されない場合は？
A: 以下を確認してください：
- MP3ファイルが正しく配置されているか
- ファイル名が人物IDと一致しているか  
- macOSの音量設定
- `afplay audio/person_1.mp3`で手動再生できるか

### Q: カメラが認識されない場合は？
A: `src/config.py`の`CAMERA_INDEX`を変更してみてください（0, 1, 2...）。また、他のアプリケーションがカメラを使用していないか確認してください。

### Q: システムが重い場合は？
A: `src/config.py`で以下を調整してください：
- `YOLO_MODEL = "yolov8n.pt"`（最軽量モデル）
- `FPS = 15`（フレームレート低下）
- `FRAME_WIDTH = 320, FRAME_HEIGHT = 240`（解像度低下）

### Q: ログファイルの場所は？
A: `logs/system.log`（システムログ）と`logs/person_log.csv`（人物イベントログ）に保存されます。

## セキュリティとプライバシー

- 映像データはローカルのみで処理
- 外部ネットワークへの送信なし
- ログファイルにはタイムスタンプと識別IDのみ記録

## ライセンス

このプロジェクトは教育・研究目的で作成されています。商用利用の際は各ライブラリのライセンスを確認してください。

## サポート

問題が発生した場合は、`logs/system.log`を確認してエラーの詳細を把握してください。
