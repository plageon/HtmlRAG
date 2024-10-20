# <div align="center">HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieval Results in RAG Systems</div>
We propose HtmlRAG, which uses HTML instead of plain text as the format of external knowledge in RAG systems. To tackle the long context brought by HTML, we propose **Lossless HTML Cleaning** and **Two-Step Block-Tree-Based HTML Pruning**.

- **Lossless HTML Cleaning**: This cleaning process just removes totally irrelevant contents and compress redundant structures, retaining all semantic information in the original HTML. The compressed HTML of lossless HTML cleaning is suitable for RAG systems that have long-context LLMs and are not willing to loss any information before generation.

- **Two-Step Block-Tree-Based HTML Pruning**: The block-tree-based HTML pruning consists of two steps, both of which are conducted on the block tree structure. The first pruning step uses a embedding model to calculate scores for blocks, while the second step uses a path generative model. The first step processes the result of lossless HTML cleaning, while the second step processes the result of the first pruning step.

---

## 🔌 Apply HtmlRAG in your own RAG systems

We provide a simple tookit to apply HtmlRAG in your own RAG systems.

### 📦 Installation

```bash
cd toolkit/
pip install -e .
```

Please refer to the [user guide](toolkit/README.md) for more details.

If you are interested in reproducing the results in the paper, please follow the instructions below.

---

## 🔧 Dependencies

You can directly import a conda environment by importing the yml file.
```bash
conda env create -f environment.yml
conda activate htmlrag
```
Or you can intsall the dependencies by yourself.
```bash
conda create -n htmlrag python=3.9 -y
conda activate htmlrag
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge faiss-cpu
pip install scikit-learn transformers transformers[deepspeed] rouge_score evaluate dataset gpustat anytree json5 tensorboardX accelerate bitsandbytes markdownify bs4 sentencepiece loguru tiktoken matplotlib langchain lxml vllm notebook trl spacy rank_bm25 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📂 Data Preparation

### Download datasets used in the paper

1. We randomly sample 400 questions from the test set (if any) or validation set in the original datasets for our evaluation. The processed data is stored in the [html_data](./html_data) folder.


2. We apply query rewriting to extract sub-queries and Bing search to retrieve relevant URLs for each querys, and then we scrap static HTML documents through URLs in returned search results. 
Original webpages are stored in the [html_data](./html_data) folder. 
Due to git file size limitation, we only provide a small subset of test data in this repository. The full processed data will be released soon.

|  Dataset   |                       ASQA                        |                             HotpotQA                             |                     NQ                      |                             TriviaQA                             |                          MuSiQue                           |                       ELI5                        |
|:----------:|:-------------------------------------------------:|:----------------------------------------------------------------:|:-------------------------------------------:|:----------------------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
| Query Data |               [asqa-test.jsonl](html_data/asqa/asqa-test.jsonl)                | [hotpot-qa-test.jsonl](html_data/hotpot-qa/hotpot-qa-test.jsonl) | [nq-test.jsonl](html_data/nq/nq-test.jsonl) | [trivia-qa-test.jsonl](html_data/trivia-qa/trivia-qa-test.jsonl) | [musique-test.jsonl](html_data/musique/musique-test.jsonl) | [eli5-test.jsonl](html_data/eli5/eli5-test.jsonl) |
| HTML Data  | [html-sample](html_data/asqa/bing/binghtml-slimplmqr-asqa-test-sample.jsonl) | [html-sample](html_data/hotpot-qa/bing/binghtml-slimplmqr-hotpot-qa-test-sample.jsonl) | [html-sample](html_data/nq/bing/binghtml-slimplmqr-nq-test-sample.jsonl) | [html-sample](html_data/trivia-qa/bing/binghtml-slimplmqr-trivia-qa-test-sample.jsonl) | [html-sample](html_data/musique/bing/binghtml-slimplmqr-musique-test-sample.jsonl) | [html-sample](html_data/eli5/bing/binghtml-slimplmqr-eli5-test-sample.jsonl) |

   

### Use your own data

You can use your own data by following the format of the datasets in the [html_data](./html_data) folder.

1. Prepare your query file in `.jsonl` format. Each line is a json object with the following fields:

```json
{
  "id": "unique_id",
  "question": "query_text",
  "answers": ["answer_text_1", "answer_text_2"]
}
```
2. Conduct a optional pre-retrieval process to get sub-queries from the original user's question. The processed sub-queries should be stored in a `{rewrite_method}_result` key in the json object.

```json
{
  "id": "unique_id",
  "question": "query_text",
  "answers": ["answer_text_1", "answer_text_2"],
  "your_method_rewrite": {
    "questions": [
      {
        "question": "sub_query_text_1"
      },
      {
        "question": "sub_query_text_2"
      }
    ]
  }
}
```

3. Conduct web search using bing

```bash
./scripts/search_pipeline_apply.sh
```

---

## 🧹 HTML Cleaning

```bash
bash ./scripts/simplify_html.sh
```

## ✂️ Block-Tree-Based HTML Pruning

### Step 1: HTML Pruning with Text Embedding

```bash
bash ./scripts/tree_rerank.sh
bash ./scripts/trim_html_tree_rerank.sh
```

### Step 2: Generative HTML Pruning

```bash
bash ./scripts/tree_rerank_tree_gen.sh
```
---

## 📊 Evaluation

### Baselines

We provide the following baselines for comparison:

- **BM25**: A widely used sparse rerank model. 
```bash
export rerank_model="bm25"
./scripts/rerank.sh
./scripts/trim_html_fill_chunk.sh
```

- **[BGE](https://huggingface.co/BAAI/bge-large-en)**: An embedding model, BGE-Large-EN with encoder-only structure. Our scripts requires instantiation of an embedding model with [TEI](https://github.com/huggingface/text-embeddings-inference).
```bash
export rerank_model="bge"
./scripts/rerank.sh
./scripts/trim_html_fill_chunk.sh
```
- **[E5-Mistral](https://huggingface.co/intfloat/e5-mistral-7b-instruct)**: A embedding model based on an LLM, Mistral-7B.  Our scripts requires instantiation of an embedding model with [TEI](https://github.com/huggingface/text-embeddings-inference).
```bash
export rerank_model="e5-mistral"
./scripts/rerank.sh
./scripts/trim_html_fill_chunk.sh
```
- **LongLLMLingua**: An abstractive model using Llama7B to select useful context.
```bash
./scripts/longlongllmlingua.sh
```
- **[JinaAI Reader](https://huggingface.co/jinaai/reader-lm-1.5b)**: An end-to-end light-weight LLM with 1.5B parameters fine-tuned on an HTML to Markdown converting task dataset. 
```bash
./scripts/jinaai_reader.sh
```
### Evaluation Scripts

1. Instantiaize a LLM inference model with [VLLM](https://github.com/vllm-project/vllm/). We recommend using [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) or [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct).
2. Run the chatting inference using the following command:
```bash
./scripts/chat_inference.sh
```
2. Follow the evaluation scripts in [eval_scrips.ipynb](./jupyter/eval_scrips.ipynb)

### Results

- **Results for [HTML-Pruner-Phi-3.8B]() with Llama-3.1-70B-Instruct as chat model**
  (The fine-tuned pruner model's checkpoint will be released soon).

| Dataset       | ASQA  | ASQA  | HotpotQA | NQ    | NQ    | TriviaQA | TriviaQA | MuSiQue | ELI5    | ELI5 |
|---------------|-------|-------|----------|-------|-------|----------|----------|---------|---------|------|
| Metrics       | EM    | Hit@1 | EM       | EM    | Hit@1 | EM       | Hit@1    | EM      | ROUGE-L | BLEU |
| BM25          | 49.50 | 21.95 | 38.25    | 47.00 | 35.56 | 88.00    | 25.63    | 9.50    | 16.15   | 6.99 |
| BGE           | 68.00 | 30.57 | 41.75    | 59.50 | 45.05 | 93.00    | 27.04    | 12.50   | 16.20   | 6.64 |
| E5-Mistral    | 63.00 | 28.75 | 36.75    | 59.50 | 44.07 | 90.75    | 26.27    | 11.00   | 16.17   | 6.72 |
| LongLLMLingua | 62.50 | 27.74 | 45.00    | 56.75 | 42.89 | 92.50    | 27.23    | 10.25   | 15.84   | 6.39 |
| JinaAI Reader | 55.25 | 23.73 | 34.25    | 48.25 | 35.40 | 90.00    | 25.35    | 9.25    | 16.06   | 6.41 |
| HtmlRAG       | 68.50 | 30.53 | 46.25    | 60.50 | 45.26 | 93.50    | 27.03    | 13.25   | 16.33   | 6.77 |

---

## 🚀 Training

### 1. Download Pretrained Models

```bash
mkdir ../../huggingface
huggingface-cli download --resume-download --local-dir-use-symlinks False microsoft/Phi-3.5-mini-instruct --path ../../huggingface/Phi-3.5-mini-instruct/
```

### 2. Configure training data

```bash
cd sft/
python dataset.py
```

### 3. Train the model

You can follow our settings if you are training on A800 clusters.

```bash
bash ./scripts/train_longctx.sh
```



