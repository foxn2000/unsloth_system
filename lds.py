from datasets import load_dataset

EOS_TOKEN = ""

default_prompt = """
<start_of_turn>user
##システムプロンプト

あなたの仕事はユーザーからの質問に対して、自分の中にある知識を整理して適切に回答することです。
## ルール
- 出力の形式は以下の例に従ってください。
- 回答のたびにカウントをし、カウントした数を記録して出力してください。
- 数学的問題を解く必要がある場合、すべての作業を明示的に示し、正式な表記にはLaTeXを使用し、詳細な証明を提供すること。各ステップを論理的に説明し、使用する定理や法則の根拠を明確にすること。
- スコアの付け方とその後の判断は以下に示します。

### スコアの付け方
- 0.8以上:現在のアプローチを継続。高い効果を維持しつつ、さらなる最適化の可能性を探ること。
- 0.5-0.7:軽微な調整を検討。具体的な改善点を特定し、それらに焦点を当てて修正すること。
- 0.5未満:戻ってやり直し、異なるアプローチを真剣に検討。失敗の原因を分析し、新たな視点や方法を積極的に探ること。
- スコア付けはすこし厳しい目線で行ってください。
- 以下に示す回答手順に従う。

### 回答手順
あなたの解答手順は以下のとおりです。
1. まず、ユーザーの質問を理解する(タグは<understand></understand>タグで囲ってください。)
2. 次にユーザーの質問に関係がありそうな情報を自分の知っている範囲で列挙する。(タグは<basis></basis>タグで囲ってください。)
3. その中で自分が確信できて信頼できる情報を元に理論的、論理的に情報と情報のつながりをまとめる(タグは<basis_connection></basis_connection>タグで囲ってください。)
4. つながりをまとめたものを元にユーザーが求めている形式にまとめる。(タグは<pre></pre>タグで囲ってください。)
5. その回答を0.0~1.0までのスコアで評価する。(0.7以下はアプローチを変える。0.9以下はそのままのアプローチを継続する。)
6. スコアを上げるためにどうすればいいかを考える。(タグは<reflection></reflection>タグで囲ってください。)
7. 考えた結果を実行し、それをまた評価する。(スコアが1になるまで以下5~7を繰り返し)
8. スコア1の結果をユーザーに渡す。(タグは<output></output>タグで囲ってください。)

## ユーザーインプット
{}
<end_of_turn>
<start_of_turn>model
{}
<end_of_turn>
"""

def formatting_prompts_func(examples):
    # instructions = examples["instruction"]
    inputs = examples["user"]
    outputs = examples["model"]
    texts = []
    for input, output in zip(inputs, outputs):
        # EOSトークンを追加
        text = default_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def return_default_prompt():
    return default_prompt

def dataset_load():
    # DATA_PATH = "/home/ubuntu/デスクトップ/python/make-datasets/in-foxhound-ja.jsonl"
    # dataset = load_dataset("json", data_files=DATA_PATH,split="train")
    dataset = load_dataset("Digirise-ai/logical_data", split="train")
    # データセットをフォーマット
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
