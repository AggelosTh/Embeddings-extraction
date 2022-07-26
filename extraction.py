from transformers import BertModel, AutoModel, BertConfig, AutoConfig, BertTokenizer, AutoTokenizer, AutoModelWithLMHead
import unicodedata
import pickle
import torch
import numpy as np
from collections import Counter

# Loading the pre-trained BERT model

###################################
# Embeddings will be derived from
# the outputs of this model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1',
           output_hidden_states = True,).to(device)
# Setting up the tokenizer
###################################

# For Greek Bert to use it is necessary the folloing function
# which removes accents and lowercase the text

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()
 

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

def token(text):
  text = text.strip()
  text = text.split(' ')
  return [sen for sen in text]  


EMBEDDING_DIM = 768


def bert_text_preparation(text, tokenizer, token):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    # cleaned_text = strip_accents_and_lowercase(text)
    marked_text = "[CLS] " + text.strip() + " [SEP]"
    # tokenized_text = tokenizer.tokenize(marked_text)
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    #print(tokens_tensor.size())
    segments_tensors = torch.tensor([segments_ids]).cuda()
    #print(segments_tensors.size())

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, bert_model):
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        # print(len(outputs))
        # print(len(outputs[0]))
        # print(len(outputs[0][0]))
        # print(len(outputs[0][0][0]))
        # print(len(outputs[0][0][0][0]))
        # input()
        hidden_states = outputs[2][:1]
        #encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # Getting embeddings from the sum of the last 4 embedding layers of BERT
    summed_last_4_layers = torch.stack(hidden_states[-4:]).sum(0)
    #token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    #token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    token_embeddings = []
    list_token_embeddings = list()
    for tensor in summed_last_4_layers[0]: 
      token_embeddings.append(tensor.tolist())
      '''
      for value in tensor.tolist():
        token_embeddings.append(value)  
    list_token_embeddings.append(token_embeddings)
    '''
    return(token_embeddings)


    # This cell takes about 4-5 minutes to execute

# This function calls the BERT tokenizer and tokenizes my preprocessed texts
# And then it produces a contextual embedding for each title

def tokenize_text_get_emb(text_processed):
  all_embeddings = []
  tokenized_texts = []
  for text in text_processed:
    # text = strip_accents_and_lowercase(text)
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer, token)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
    tokenized_text.pop() # removing the [SEP] token
    tokenized_text.pop(0) # removing the [CLS] token
    list_token_embeddings.pop() # removing the [SEP] embedding
    list_token_embeddings.pop(0) # removing the [CLS] embedding
    tokenized_texts.append(tokenized_text)
    all_embeddings.append(list_token_embeddings)
  return(tokenized_texts, all_embeddings)

def flatten(list):
  return [x for xs in list for x in xs]

def toString(list):
  return ' '.join(str(v) for v in list)


with open('corpus_accom_gr.txt', encoding='utf-8') as corpus:
  text = corpus.readlines()

text_processed = text
# text_processed = [strip_accents_and_lowercase(sen) for sen in text]

tokenized_texts, all_embeddings = tokenize_text_get_emb(text_processed) 
# all_embeddings is a list of lists. Each sublist represents a title. 
# The elements of each sublist are the word and subword embeddings of that sentence.
# so we have successfully produced contextual embeddings for each title, which we will then average to get the avg sentence embedding
tokenized_texts = flatten(tokenized_texts)
all_embeddings = flatten(all_embeddings)

# print(tokenized_texts)
# input()
dictionary = dict(zip(tokenized_texts, all_embeddings))
# print(len(all_embeddings[0][0]))

with open('greek_bert_accom.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('greek_bert_accom.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)

# with open('bert_emb_test', 'w') as text:
#   for token, vector in b.items():
#     text.write(token + ' ')
#     text.write(toString(vector))
#     text.write('\n')


voc=Counter()
voc.update(tokenized_texts)
token_index = {t[0]: i for i,t in enumerate(voc.items())}
MAX_NB_WORDS = len(voc.keys())
print(MAX_NB_WORDS)
print(len(embeddings.keys()))
input()



with open('bert_emb_accom.txt', 'w', encoding='utf-8') as file:
  for token, i in token_index.items():#words of dataset
      if i >= MAX_NB_WORDS:
          continue
      try:
          embedding_vector = embeddings[token][0:768]
          token = token.strip()
          file.write(token + ' ')
          for num in embedding_vector:
            file.write(str(num) + ' ')
          file.write('\n')  
          #embedding_vector = model[word][0:WV_DIM]
          # words not found in embedding index will be all-zeros.
      except:
          pass   
