# Cloud GPU Notebook 実行手順

## 0. 前提条件
- Hugging Face アカウントと `unsloth/gpt-oss-20b` へアクセスできるトークンを用意する
- 80GB 以上のディスク (モデルとデータキャッシュ用)、Linux (Ubuntu 22.04 推奨)
- CUDA 対応 GPU (A100 40GB 以上を推奨、24GB 以下では 4bit でもメモリ不足になりやすい)
- ローカルにある `data/*.csv` をクラウド環境へ同期できる手段 (例: `scp`, `rsync`, S3 等)

## 1. GPU インスタンスの準備
- クラウド事業者のコンソールで GPU ノードを作成し、SSH 鍵を登録する
- OS 起動後、GPU ドライバがプリインストールされていない場合は最新版の NVIDIA Driver + CUDA Toolkit を導入する
- SSH で接続できることと `nvidia-smi` が正常に動くことを確認する

## 2. 基本ツールの導入
```bash
sudo apt update
sudo apt install -y git tmux build-essential python3 python3-venv python3-pip unzip
python3 -m pip install --upgrade pip
```

## 3. プロジェクト取得とデータ配置
```bash
git clone https://github.com/takumi0211/notebook_grpo.git
cd notebook_grpo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```
- ローカルの `data/*.csv` をクラウド側の `notebook_grpo/data/` へコピーする  
  `scp -r data/ user@remote:/path/to/notebook_grpo/data` などで同期する
- Harmony 形式へ変換が必要な CSV がある場合は以下を実行
  ```bash
  python convert_to_harmony.py data --overwrite
  ```

## 4. Python 依存関係のセットアップ
- GPU 版 PyTorch (CUDA 12.1 の例)
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- GRPO / Unsloth で使うライブラリ
  ```bash
  pip install \
    bitsandbytes \
    accelerate \
    transformers==4.56.2 \
    tokenizers \
    trl==0.22.2 \
    datasets \
    pandas \
    unsloth \
    unsloth_zoo \
    uv \
    jupyterlab
  ```
- Hugging Face へのログイン
  ```bash
  huggingface-cli login
  ```
- 単一 GPU の場合でも `accelerate` 設定を作成しておくと便利
  ```bash
  accelerate config default
  ```

## 5. Notebook サーバを起動
```bash
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```
- 表示された URL もしくはトークンをローカルブラウザで入力し、`gpt_oss_Fine_tuning.ipynb` を開く
- 必要に応じて `tmux` や `screen` を使い、切断時もサーバが維持されるようにする

## 6. Notebook 実行の流れ
1. 先頭セルで `!nvidia-smi` や `torch.cuda.is_available()` を実行して GPU が見えているか確認  
   `CUDA_VISIBLE_DEVICES` を制御したい場合はこのセルで設定する
2. 依存パッケージをノートブック内で追加するセル (`pip install --upgrade -qqq uv` など) を実行  
   既に仮想環境で導入済みでも再実行して問題ない
3. `train_grpo.py` のロジックに合わせて以下の値を確認・必要なら変更
   - `MODEL_ID` (デフォルト: `unsloth/gpt-oss-20b`)
   - 保存先 `runs/grpo_gptoss20b_lora4_tes`
   - 学習ステップ数・サンプル数 (`TOTAL_STEPS`, `PROMPTS_PER_STEP`, `NUM_GENERATIONS`)
   - `data/` 配下の CSV が期待通り読み込めているか
4. データプレビュー用セルで `load_prompt_dataset()` がエラーにならないことを確認  
   Harmony 形式でない場合は事前に変換すべきファイルを洗い出す
5. 学習用セルを実行 (`run_training()` など)  
   - 実行中は GPU メモリを監視 (`!watch -n 10 nvidia-smi` を別ターミナルで実行)
   - 中断する際は `trainer.trainer.accelerator.state.deepspeed_plugin` 等の状態が壊れないように `KeyboardInterrupt` 後に `Runtime`→`Restart` を選ぶ
6. 学習完了後 `runs/grpo_gptoss20b_lora4_tes/` に LoRA アダプタとトークナイザが保存されることを確認  
   必要なら `tar czf runs.tar.gz runs/` 等で成果物をまとめてダウンロードする

## 7. 追加のベストプラクティス
- 長時間実行前に `ulimit -n 4096` を設定し、ファイルディスクリプタ枯渇を防ぐ
- `HF_HOME` や `TORCH_HOME` を NVMe 領域へ変更するとキャッシュ I/O が高速になる
- 共有 GPU ノードでは `wandb` を使ったログ送信や `accelerate launch` での再現性確保を検討する
- 新しいプロンプトを追加する際は `convert_to_harmony.py` を利用して一括変換→`data/` に配置→ノートブックで再読込
