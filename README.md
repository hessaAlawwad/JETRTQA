# Multimodal Retriever-Generator Model for Visual Question Answering

This repository implements a multimodal retrieval and generation model aimed at visual question answering (VQA). The model utilizes advanced techniques in machine learning and deep learning, including the use of state-of-the-art vision-language models and retrieval-based training.

## Used Models

### 1. **Llama-3.2-11B-Vision-Instruct (Fine-tuned with LoRA)**
The core of the model utilizes **Llama-3.2-11B-Vision-Instruct**, fine-tuned using the **unsloth** package. This fine-tuning leverages **LoRA (Low-Rank Adaptation)** to efficiently adjust the pre-trained Llama model to perform well in the multimodal setting with images and text. 

For more details on the fine-tuning and the LoRA adapter used, visit:
- [unsloth GitHub repository](https://github.com/unslothai/unsloth)

### 2. **ImageBind**
**ImageBind** is a novel cross-modal model developed by Facebook Research that is utilized to create embeddings for both image and text data in a shared space. It enables us to map both image and text inputs into a common feature space for downstream retrieval and generation tasks.

For more details about ImageBind, refer to:
- [ImageBind GitHub repository](https://github.com/facebookresearch/ImageBind)

## Project Overview

This project implements a **retriever-generator** architecture. It consists of the following main components:

1. **Retrieval Model**: A neural network that computes a retrieval score between a multimodal query (image + text) and candidate documents, which can either be images or texts. The retrieval model is trained to rank documents based on their relevance to the query.
   
2. **Generator Model**: A model that generates answers based on the retrieved documents and the query. This model is based on the fine-tuned Llama-3.2-11B-Vision-Instruct.

3. **Image and Text Embeddings**: The embeddings are generated using the **ImageBind** model for both image and text inputs. These embeddings are used for computing the retrieval scores and for feeding into the generator model.

## Requirements

To run this code, you'll need to have the following dependencies installed:

- `torch`
- `transformers`
- `peft`
- `imagebind`
- `PIL`
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib`
- `bitsandbytes`
- `IPython`

## Training
To train the model, you need to:

Download or prepare your dataset. 
The script expects the dataset to be in CSV format with specific columns such as question_text, question_image, answer_choices, label, and documents.
Adjust the configuration in the script (such as paths to models, dataset location, and training settings).
Run the training script, which will handle loading the data, preprocessing, and training the retrieval and generation models.

## Model Output
The model saves the trained retriever and generator models to a specified output directory. The output models can then be loaded for inference or further fine-tuning.

