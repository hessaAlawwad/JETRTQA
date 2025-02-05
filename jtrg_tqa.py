import os
import io
import time
import logging
import json
import pickle
import ast
import pathlib
import textwrap
import warnings
from getpass import getpass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, Markdown
from IPython.core.debugger import Pdb

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import List, Optional, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForImageTextToText

from peft import get_peft_model, LoraConfig, TaskType

import bitsandbytes as bnb
from unsloth import FastVisionModel
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

 # Configure logging
logging.basicConfig(
        #filename="training_log.txt",
        filename="/home/habdulazizalawwad/training_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
     
 
def calculate_logits(question, question_image, answer_choices, label, result_type, result, predicted_answer=None, logits=None):

        #calculate logins from using the model llama3.2 vision instruct

        return results

def find_full_path(filename):
    main_path = "/home/habdulazizalawwad/dataset/"
    images_root = main_path + "tqa_train_val_test/"
    subfolders = ["test", "train", "val"]
    for folder in subfolders:
        path = os.path.join(images_root, folder, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image {filename} not found in test, train, or val folders.")

       
def getEmbeddingVector(inputs):
     with torch.no_grad():
         embedding = imagebind(inputs)
     for key, value in embedding.items():
         vec = value.reshape(-1)
         return(vec)

# dataToEmbedding: Takes in the image file path and returns an embedded vector
def imageToEmbedding(path, device):
     inputs = { ModalityType.VISION: data.load_and_transform_vision_data(path, device)  }
     vec = getEmbeddingVector(inputs)
     return(vec)

# dataToEmbedding: Takes in the text file path and returns an embedded vector
def textToEmbedding(txt, device):
     text_list = [txt]
     inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device)}
     vec = getEmbeddingVector(inputs)
     return(vec)
           
class RetrievalModel(nn.Module):
    def __init__(self, input_dim=1024*3, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)  # Output Ŝqi,dk; Output dimension is 1; just the rank


class RetrieverTrainer:
    def __init__(self, generator_model, processor,  input_dim=1024 * 3, train_generator=False, use_precompute=False): # Change according to imagebind output dimensions
        # times 3 because we are embedding text and image for the query along with the retrieved context.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_generator = train_generator
        self.use_precompute = use_precompute
        self.retrieval_model = RetrievalModel(input_dim)  # Input is concatenation of query and doc embeddings

        if self.train_generator:
            self.generator_model = generator_model # fA
            self.processor = processor
            # self.generator_model.to(device)
            # self.processor.to(device)


        #self.imagebind_model = imagebind_model # this is needed since we dont have pre-computed embeddings
        self.retrieval_model.to(device)
        # self.imagebind_model.to(device)

        # Optimizers
        self.retrieval_optimizer = torch.optim.AdamW(self.retrieval_model.parameters(), lr=3e-3)

        if self.train_generator:
            self.generator_optimizer = torch.optim.AdamW(self.generator_model.parameters(), lr=1e-5)


    def encode_diagram_query(self, question_text, question_image):
        """
        Encodes the multimodal query using ImageBind
        """
        query_text_emb = textToEmbedding(question_text, device)
       
        full_path = find_full_path(question_image)
        query_image_emb = imageToEmbedding([full_path], device)

        query_text_emb = query_text_emb.to(device)
        query_image_emb = query_image_emb.to(device)

        return torch.cat([query_text_emb, query_image_emb], dim=-1)
       
    def encode_textual_query(self, question_text):
        """
        Encodes the textual query using ImageBind and ensures a 3072-dimensional embedding.
        """
        query_text_emb = textToEmbedding(question_text, device)  # Get text embedding
   
        # Ensure the embedding is on the correct device
        query_text_emb = query_text_emb.to(device)
   
        # Check embedding shape
        expected_dim = 2*1024
        actual_dim = query_text_emb.shape[-1]  # Get the last dimension (features)
   
        if actual_dim != expected_dim:
            print(f"WARNING: Reshaping text-only query embedding from {actual_dim} to {expected_dim}")
   
            # If embedding is smaller, pad it with zeros
            if actual_dim < expected_dim:
                padding_size = expected_dim - actual_dim
                query_text_emb = F.pad(query_text_emb, (0, padding_size), "constant", 0)
            # If embedding is larger, trim it
            elif actual_dim > expected_dim:
                query_text_emb = query_text_emb[:, :expected_dim]
   
        return query_text_emb


    def encode_retrieved_document(self, doc, doc_type):
        """
        Encodes the document using ImageBind
        """        
        if doc_type == "image":
            doc_image = doc['context_image']
            doc_image = str(doc_image)
            # print("Type of image path: ", type(doc_image))
            # Check if the image path is None or empty
            if not doc_image or doc_image == "None":
                raise ValueError(f"Invalid image path: {doc_image}")

            full_path = find_full_path(doc_image)
            encoded_doc = imageToEmbedding([full_path], device)
        else:
            doc_text = doc['text_content']
            encoded_doc = textToEmbedding(doc_text, device)

        encoded_doc = encoded_doc.to(device)
        return encoded_doc

    def get_retrieval_score(self, query_emb, doc_emb):
        """
        Calculates Ŝqi,dk (retriever's score) between query and document
        This uses the retrieval model to score the ranking
        (part of fR(Q; ϕ))

        Here,
            ipdb>  query_emb.size()
            torch.Size([2, 1, 1024]) -- Account for the question's parts

            ipdb>  doc_emb.size()
            torch.Size([1, 1024])
        """
        # return self.retrieval_model(combined_emb)
        query_emb_flat = query_emb.view(-1)  # Flatten query embedding
        doc_emb_flat = doc_emb.view(-1)  # Flatten document embedding

        # # Ensure both embeddings are on the correct device
        query_emb_flat = query_emb_flat.to(device)
        doc_emb_flat = doc_emb_flat.to(device)

        # Concatenate the embeddings and pass through the retrieval model
        combined_emb = torch.cat([query_emb_flat, doc_emb_flat], dim=0)
        combined_emb = combined_emb.unsqueeze(0)  # Add batch dimension

        return self.retrieval_model(combined_emb)

    def get_option_content(self, options_string, option_letter):
        # Extract option letters from the options string
        options = options_string.split(", ")
        available_letters = [option.split(".")[0].strip() for option in options]

        # Check if the provided option letter is in the list of available letters
        if option_letter not in available_letters:
            return f"Option '{option_letter}' is not available in the options string."

        # Iterate through options and find the matching letter
        for option in options:
            if option.startswith(f"{option_letter}."):
                # Return the content after the letter and period
                return option.split(".")[1].strip()

        # If the option is not found (shouldn't happen), return None
        return None

    def compute_similarity_loss(self, document_type, document, answer_choices, predicted_char, lambda_weight=0.7):
        """
        document_type: Image or text context
        document: str | image - provided context
        answer: str - generated answer
        lambda_weight: float - weight for similarity loss
        """
        # Compute embeddings
        document_embedding = self.encode_retrieved_document(document, document_type)
        predicted_answer = self.get_option_content(answer_choices, predicted_char)
        answer_embedding = textToEmbedding(predicted_answer, device)

        cos_sim = F.cosine_similarity(document_embedding.unsqueeze(0), answer_embedding.unsqueeze(0))

        # Loss is 1 - cosine similarity
        similarity_loss = lambda_weight * (1 - cos_sim.clone())
        return similarity_loss

    def compute_single_cross_entropy(self, logits_list, label, valid_options):
        valid_options = [option for option in valid_options if any(logit_entry['letter'] == option for logit_entry in logits_list)]
        if not valid_options:
            raise ValueError("No valid options found in logits_list.")

        option_to_index = {option: idx for idx, option in enumerate(valid_options)}
        if label not in option_to_index:
            raise ValueError(f"Invalid label: {label}")

        sorted_logits = torch.zeros(len(valid_options))
        for idx, option in enumerate(valid_options):
            for logit_entry in logits_list:
                if logit_entry['letter'] == option:
                    sorted_logits[idx] = logit_entry['logit']
                    break

        label_index = option_to_index[label]

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(sorted_logits.unsqueeze(0), torch.tensor([label_index]))
        return loss
       
    def _save_analysis(self, analysis_data, epoch, batch):
        os.makedirs("analysis", exist_ok=True)
        analysis_file = os.path.join("analysis", f"analysis_epoch_{epoch}_batch_{batch}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=4)
           
    def train_step(self, batch_data, current_epoch=1, current_batch=0):
        """
        Processes a batch of questions and updates weights once per batch
        """
        self.retrieval_model.train()
           
        # Initialize accumulators
        batch_total_loss = 0.0
        batch_rank_loss = 0.0
        batch_gen_loss = 0.0
        analysis_data = []
               
        self.retrieval_optimizer.zero_grad()
        if self.train_generator:
            self.generator_optimizer.zero_grad()
   
        # Process each question in the batch
        for q_idx in range(len(batch_data['question_text'])):
            # Extract single question data
            question_text = batch_data['question_text'][q_idx]
            question_image = batch_data['question_image'][q_idx]
            answer_choices = batch_data['answer_choices'][q_idx]
            label = batch_data['label'][q_idx]
            documents = batch_data['documents'][q_idx]
   
            # Initialize per-question tracking
            q_analysis = {
                "doc_losses": [],
                "gen_losses": [],
                "Ci_values": [],
                "pairwise_comparisons": [],
                "query_text": question_text,
                "query_image": question_image
            }
   
            # Encode query
            if pd.notna(question_image) and question_image.strip():  # Ensure it's not empty
                query_emb = self.encode_diagram_query(question_text, question_image)
            else:
                query_emb = self.encode_textual_query(question_text)
           
            if torch.isnan(query_emb).any():
                print("ERROR: NaN found in query embedding!")
             
            #Normalize Embeddings Before Retrieval Score Calculation    
            query_emb = F.normalize(query_emb, p=2, dim=-1)
           
            # First pass: Document processing
            doc_losses = []
            gen_losses = []
            for doc in documents:
                #print("Document structure:", documents[0])
                # Debugging: Print the structure of doc
                #print("Inspecting doc structure:")
                #for index, element in enumerate(doc, start=1):
                    #print(f"{index}: {element}")
       
                doc_type = doc['result_type']
                result_image_path = doc['context_image']
                content = doc['text_content']
                cosin_sim_score =  doc['cosin_sim_score']      
                logits = doc['logits']
                             
                # Compute generator losses
                if self.use_precompute:
                    predicted_char = logits[0]['predicted_token']
                else:
                    predicted_char, _ = calculate_logits(question_text, question_image, answer_choices, doc, label)
               
                logits = logits[0]['option_logits']
                cross_entropy_loss = self.compute_single_cross_entropy(logits, label, answer_choices)
                similarity_loss = self.compute_similarity_loss(doc_type, doc, answer_choices, predicted_char, 0.7)
                gen_loss = cross_entropy_loss + similarity_loss
   
                doc_losses.append(cross_entropy_loss)
                gen_losses.append(gen_loss)
                q_analysis["doc_losses"].append(cross_entropy_loss.item())
                q_analysis["gen_losses"].append(gen_loss.item())
   
            # Second pass: Ranking loss
            total_rank_loss = 0.0
            pair_count = 0
            for i, doc_i in enumerate(documents):
                doc_emb_i = self.encode_retrieved_document(doc_i, doc_i['result_type'])
                doc_emb_i = F.normalize(doc_emb_i, p=2, dim=-1)
                if torch.isnan(doc_emb_i).any():
                    print("ERROR: NaN found in document doc_emb_i embedding!")

                score_i = self.get_retrieval_score(query_emb, doc_emb_i)
               
                for j, doc_j in enumerate(documents[i+1:], start=i+1):
                    Li = 1 if doc_losses[i] != doc_losses[j] else 0

                    if Li != 0:
                        doc_emb_j = self.encode_retrieved_document(doc_j, doc_j['result_type'])
                        doc_emb_j = F.normalize(doc_emb_j, p=2, dim=-1)
                        if torch.isnan(doc_emb_j).any():
                            print("ERROR: NaN found in document doc_emb_j embedding!")
                           
                        score_j = self.get_retrieval_score(query_emb, doc_emb_j)
   
                        #Ci = (torch.sigmoid(score_i) - torch.sigmoid(score_j)).abs()
                        #Fi = torch.log(Ci) if doc_losses[i] < doc_losses[j] else torch.log(1 - Ci)
                        eps = 1e-8  # Small value to prevent log(0)
                        Ci = (torch.sigmoid(score_i) - torch.sigmoid(score_j)).abs() + eps  # Avoid zero
                        Fi = torch.log(Ci) if doc_losses[i] < doc_losses[j] else torch.log(1 - Ci + eps)
                        pair_rank_loss = -Fi
   
                        total_rank_loss += pair_rank_loss
                        pair_count += 1
   
                        q_analysis["Ci_values"].append(Ci.item())
                        q_analysis["pairwise_comparisons"].append({
                            "doc_i": i,
                            "doc_j": j,
                            "score_i": score_i.item(),
                            "score_j": score_j.item(),
                            "Ci": Ci.item(),
                            "Fi": Fi.item(),
                        })
   
            # Calculate per-question losses
            avg_rank_loss = total_rank_loss / pair_count if pair_count > 0 else torch.tensor(0.0)
            gen_loss = torch.stack(gen_losses).mean() if gen_losses else torch.tensor(0.0)
            q_total_loss = avg_rank_loss + gen_loss
   
            # Accumulate batch losses
            batch_total_loss += q_total_loss
            batch_rank_loss += avg_rank_loss
            batch_gen_loss += gen_loss
            analysis_data.append(q_analysis)
   
        # Average losses over batch
        batch_size = len(batch_data['question_text'])
        avg_total_loss = batch_total_loss / batch_size
        avg_rank_loss = batch_rank_loss / batch_size
        avg_gen_loss = batch_gen_loss / batch_size
   
        # Backpropagate once per batch
        avg_total_loss.backward()
       
        # Clip Gradients to Prevent Exploding Gradients
        torch.nn.utils.clip_grad_norm_(self.retrieval_model.parameters(), max_norm=1.0)

        # Update weights once per batch
        self.retrieval_optimizer.step()
        if self.train_generator:
            self.generator_optimizer.step()
   
        # Save analysis (one file per batch)
        self._save_analysis(analysis_data, current_epoch, current_batch)
   
        return (
            avg_total_loss.item(),
            avg_rank_loss.item(),
            avg_gen_loss.item()
        )

class TQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    # Keep lists of dictionaries separate per sample
    question_texts = [item['question_text'] for item in batch]
    question_images = [item['question_image'] for item in batch]
    answer_choices = [item['answer_choices'] for item in batch]
    labels = [item['label'] for item in batch]
    documents = [item['documents'] for item in batch]  # Keep as a list of lists

    return {
        'question_text': question_texts,
        'question_image': question_images,
        'answer_choices': answer_choices,
        'label': labels,
        'documents': documents
    }
 
if __name__ == "__main__":

    # Log the start of the script
    logging.info("Script execution started.")
    # Set the device based on availability
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load and instantiate the ImageBind model
    imagebind = imagebind_model.imagebind_huge(pretrained=False)
    logging.info("ImageBind model instantiated.")

    # Load the model's state dict from the checkpoint
    checkpoint_path = "/home/habdulazizalawwad/models/ImageBind/.checkpoints/imagebind_huge.pth"
    imagebind.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    logging.info(f"ImageBind model weights loaded from: {checkpoint_path}")
   
    # Set the model to evaluation mode and move it to the device
    imagebind.eval()
    imagebind.to(device)
    logging.info(f"ImageBind model set to evaluation mode and moved to {device}.")

    # Make sure you're in offline mode
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Load the generator model and tokenizer from the local path
    local_model_path = "/home/habdulazizalawwad/models/lora_adapter"
    # Load the model and tokenizer locally
    #generator_model = AutoModelForImageTextToText.from_pretrained(local_model_path, torch_dtype=torch.bfloat16 )
    #tokenizer = AutoTokenizer.from_pretrained(local_model_path)
   
    generator_model, tokenizer = FastVisionModel.from_pretrained(  
                                  model_name=local_model_path,
                                  dtype=torch.bfloat16,
                                  load_in_4bit=False,
                                  )
   
    logging.info(f"Generator model loaded from {local_model_path}.")
    logging.info(f"Trainable Parameters Prints by me: {generator_model.print_trainable_parameters()}.")
    logging.info(f"Type of generator_model: {type(generator_model)}")
   
   
    # Verify that the generator model is loaded and moved to the device
    generator_model.to(device)
    logging.info(f"Generator model moved to {device}.")
   
    # Load the processor
    local_processor_path = "/home/habdulazizalawwad/models/lora_adapter"
    processor = AutoProcessor.from_pretrained(local_processor_path)
    logging.info(f"Processor loaded from {local_processor_path}.")

    # Check if tokenizer and processor are working as expected
    logging.info(f"Tokenizer and Processor are ready.")

   
    trainer = RetrieverTrainer(generator_model, processor, train_generator=False, use_precompute=True)    

    number_of_epochs = 1
    batch_size = 8
    num_gpus = torch.cuda.device_count()

    # Load and preprocess the data
    file_path = "/home/habdulazizalawwad/dataset/filtered_dataset.csv"
    df = pd.read_csv(file_path)
    #df = df.head(10)
   
    dataset = {}
   
    for i in tqdm(range(len(df))):
        ques_id = df.loc[i, 'questionID']
        if ques_id in list(dataset.keys()):
            dict_ = dataset[ques_id]
            dict_['documents'].append(
                {
                    "result_type": df.loc[i, 'result_type'],
                    "context_image": df.loc[i, 'context_image'],
                    "text_content": df.loc[i, 'text_content'],
                    "cosin_sim_score": df.loc[i, 'cosin_sim_score'],
                    "logits": ast.literal_eval(df.loc[i, 'logits']),
                }
            )
            dataset[ques_id] = dict_
        else:
            dict_ = {}
            dict_['question_text'] = df.loc[i, 'question_text']
            dict_['question_image'] = df.loc[i, 'question_image']
            dict_['answer_choices'] = df.loc[i, 'answer_choices']
            dict_['label'] = df.loc[i, 'label']
            dict_['documents'] = [
                {
                    "result_type": df.loc[i, 'result_type'],
                    "context_image": df.loc[i, 'context_image'],
                    "text_content": df.loc[i, 'text_content'],
                    "cosin_sim_score": df.loc[i, 'cosin_sim_score'],
                    "logits": ast.literal_eval(df.loc[i, 'logits']),
                }
            ]
            dataset[ques_id] = dict_
   
    final_data = [v for k,v in dataset.items()]
    dataset_tqa = TQADataset(final_data)
    data_loader = DataLoader(dataset_tqa, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
   
    batch_times = []
   
    # Pre-training message
    logging.info(f"==((====))==  Num GPUs = {num_gpus}")
    logging.info(f"   \\   /|    Num of questions = {len(dataset)} | Num Epochs = {number_of_epochs}")
    logging.info(f" / \\_/ \\    Batch size per device = {batch_size}")
   
    # Start training
    for epoch_idx in range(number_of_epochs):
        epoch_loss = 0
        epoch_start_time = time.time()
   
        for batch_idx, batch_data in tqdm(enumerate(data_loader), desc=f"Epoch {epoch_idx+1}"):
            batch_start_time = time.time()

            # Forward pass and loss calculation for the ENTIRE BATCH
            loss, rank_loss, gen_loss = trainer.train_step(
                batch_data=batch_data,
                current_epoch=epoch_idx,
                current_batch=batch_idx,
                )

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
   
            epoch_loss += loss
   
            logging.info(
                f"Batch {batch_idx + 1}, Loss: {loss:.4f}, Rank Loss: {rank_loss:.4f}, "
                f"Gen Loss: {gen_loss:.4f}, Time: {batch_time:.4f} seconds"
            )
   
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
   
        logging.info(f"Epoch {epoch_idx + 1} completed in {epoch_time:.2f} seconds.")
        logging.info(f"Epoch {epoch_idx + 1}, Average Loss: {epoch_loss / len(data_loader):.4f}")
   
    # Calculate and log time statistics
    avg_time_per_question = sum(batch_times) / len(batch_times)
    total_time = avg_time_per_question * len(dataset_tqa) * number_of_epochs
    total_questions = len(dataset_tqa)  # Number of questions
    avg_loss_per_question = epoch_loss / total_questions

    logging.info(f"Average time per questionID: {avg_time_per_question:.4f} seconds")
    logging.info(f"Epoch {epoch_idx + 1}, Average Loss per Question: {avg_loss_per_question:.4f}")
    logging.info(f"Estimated total training time for {number_of_epochs} epochs: {total_time:.2f} seconds")
   
    # Specify the directory to save the models
    save_directory = "/home/habdulazizalawwad/output_models/"
   
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)
   
    # Saving the retriever model
    retriever_model_path = os.path.join(save_directory, 'retriever_model.pth')
    torch.save(trainer.retrieval_model.state_dict(), retriever_model_path)
    print(f"Retriever model saved at: {retriever_model_path}")
   
    # Saving the generator model (if it's trained and being used)
    if trainer.train_generator:
        trainer.generator_model.save_pretrained(save_directory)
       
        #generator_model_path = os.path.join(save_directory, 'generator_model.pth')
        #torch.save(trainer.generator_model.state_dict(), generator_model_path)
        print(f"Generator model saved at: {save_directory}")