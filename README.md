# playstore-review-chatbot

Create Chatbot to Answer questions based on playstore review data

## Configure

### Prerequisite
This project uses:
* Python 3.11
* Mistral:7b on Ollama

#### How to install Ollama
Please open this [link](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) to see how to install Ollama on your platform

#### Install mistral:7b on Ollama
```
ollama pull mistral
```

### Installation

Install dependencies required for this project by execute:
```
make install
```

### Run

Please follow these steps below

#### Run Data Sampler

We use data from [spotify review dataset](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews/) then sample it to only ~200 because computation limit. Download the data from above url then do:

```
python sampler.py
```

#### Run Data Ingestion
This command below to ingest data to faiss vectordb
```
python ingest.py
```

#### Run the Chatbot

Execute the command below to run the chatbot
```
make run
```

### Notes

#### TODO:
1. The solution above has limitation on prompt context length so it decrease the accuracy. For large data, we need to try another approach such as using agents or function calling so we can process the whole dataset
2. We use mistral:7b which is free and open source llm so the accuracy is not so well. We need to try using gpt3.5 or gpt4 (but i have no money :sadpepe)
3. Need to try to improve the prompt for better accuracy