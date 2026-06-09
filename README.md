# Intelligent_Design_of_circuit

## 環境構築
uv という Python 仮想環境管理ソフトウェアを使っています．
使ったことがない人は
```bash
pip install uv
```
でインストールしてください．

`uv` がインストールできたら，今回のプログラムで必要なライブラリをインストールするため，
```bash
uv sync
```
を実行します．

環境構築をできたかを確認するため，
```bash
uv run scripts/forward.py
```
を実行してください．プログラムが動けば成功です．
