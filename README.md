# BERT_Pipeline
A generic ML pipeline for text classification using BERT (NLP)

Deep Learning Architecture: Using Google's hugging face PyTorch port.

Similar to building other custom PyTorch architectures, I am initializing the BERT pre-trained model and adding additional layers to act as a customer classifier.

## Initializing and Preprocessing:

Similar to any TensorFlow or PyTorch models, I have defined the architecture. I have coded in PyTorch, so in the init and forward functions of the class, I have initialized the model parameters and defined model.

Once we created the forward pass, we need to find a way to optimize the weights to reduce the loss. In the case of BERT, we need to tokenize the strings in the sentence, convert into id's and map them to BERT vocabulary.

The pre-trained model we are using here is 'bert-base-uncased' and the hugging face's port helps to initialize the tokenizer by giving pre-trained model as an input. Further, we can get the tokenized text from the sentence using the initialed tokenizer.

Once we get tokens, we can convert into ID's so we can map to BERT vocabulary. A list of tokens can be converted to a list of IDS by convert_tokens_to_ids function.

## Training and Evaluation: 

I am using the standard way to calculate the target label by adding softmax at the end. Similar to other pre-trained models, I have two learning rates lr_pre and lr_regular. The idea is that there is no need to apply aggressive learning rates on the pre-trained part of the network as the parts of network are randomly initialzied but others are pre trained. Decreasing the learning rate for every epoch or for a group of epochs is a good practice. Using Adam optimizer on cross-entropy loss.
