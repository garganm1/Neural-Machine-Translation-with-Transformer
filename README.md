# Neural-Machine-Translation-with-Transformer

Transformers are the craze right now (especially in the NLP domain) and so it is natural (or even required) for any Deep Learning enthusiast to have undergone the underlying concepts and/or apply them in their own work. In this project, the aim is to use Transformers in the context of translation (from English to French). The dataset has been taken from http://www.manythings.org/bilingual/ and it is quite similar to one of my other projects (https://github.com/garganm1/Neural-Machine-Translation-with-Bahdanau-Attention) which incorporates an LSTM based Encoder-Decoder system to learn and translate (applied with Bahdanau Attention)

Since the Transformers were uncovered (in Dec. 2017 precisely through https://arxiv.org/pdf/1706.03762.pdf), they have overthrown the earlier conventional SOTA Attention based models and currently sit on the throne. Many pre-trained and variant architectures such as BERT, GPT have now entered the market. They are so popular that the company HuggingFace have their own library on it (which I assume is open-sourced). They are much much faster than the previous RNN based attention models and deliver better results.

This project will undergo all of the concepts behind the Transformer, move through the formulation of concepts on Python for building the model, train and test(validate) simultaneously and finally use inferences algorithms (Greedy and Beam Search) to generate results.

In this notebook, we need to have tensorflow (2.4) installed in order to run it. Some knowledge on Attention Based Models and Object Oriented Programming employed in Deep Learning Models is assumed.

Source Language:- English <br>
Target Language:- French

The notebook is segmented into following sections:

Section 1: Data Processing <br>
Section 2: Data Tokenization <br>
Section 3: Understanding the Transformer <br>
Section 4: Transformer Model Peripherals <br>
Section 5: Defining the Model <br>
Section 6: Defining Model Parameters <br>
Section 7: Training the Model <br>
Section 8: Model Inference (Greedy & Beam Search)

## Section 1: Data Processing

The data consists of english phrases/sentences and their french translations.

1. All special characters are removed
2. Sentence-ending symbols (. , ? !) are given spaces
3. Add 'BOS' at the start of the sentence and 'EOS' at the end of the sentence (to signal the model the start and end of any phrase/sentence)

We end up with data that contains information like this-

| ENG_processed | FRA_processed |
| ------------- | ------------- |
| <start> tom lied to all of us . <end> | <start> tom nous a a tous menti . <end> |
| <start> tom lied to everybody . <end> | <start> tom a menti a tout le monde . <end> |
| <start> tom liked what he saw . <end> | <start> tom a aime ce qu il a vu . <end> |

and so on...

## Section 2: Data Tokenization

The tensorflow's tokenizer and padding functionalities will tokenize the processed text.

This means that for two texts in the corpus -

S1:- I am here <br>
S2:- He is here right now

The tokenized form would be -

S1:- [1, 2, 3, 0, 0] <br>
S2:- [4, 5, 3, 6, 7]

Basically, the tokenized form would replace the word with a unique number that would represent that word.

- Notice the padding done at the end of sentence 1 (two 0's added). This will be done based on the maximum length of a sentence in a particular language
- Notice the repetiton of 3 in both tokenized forms which represent the word 'here' being repeated

Two separate tokenizers for each language will fit onto the each language's corpus to be tokenized. The tokenized text will be pickle stored finally to train the model with

There are other ways to tokenize such as using BERT Tokenizer that also takes sub-words into consideration apart from the whole words. For this notebook, we will stick with just the basic tokenization processes.


## Section 3: Understanding the Transformer

While it is highly recommended to understand the functioning of Transformer, I would suggest to look into the Notebook file wherein I have given a comprehensive explanation to the architecture (ofc with the help of some internet articles, posts and the original paper). It would not have looked good to include the lengthy discussion on this readme. So, check out the notebook and dash forward to this section inside.



## Section 4: Transformer Model Peripherals

Now that the model is understood, it is time to formulate the architecture on Python.

Firstly, functions for creating positional encodings are coded and tested. Then, functions for creating padding and look-ahead masks are coded and tested. These are imperative and serve as peripheral functions to the model. The functions are quite flexible in the input that they are given and Tensorflow plays a major part in it.


## Section 5: Defining the Model

The Model has Encoder and Decoder at its root and they consist of Encoding layers and Decoding layers respectively. Each of the Encoding/Decoding layer works with atleast one Scaled-Dot-Product Attention (or Self-Attention) layer with multiple heads and a Feed-Forward Network layer. Each of the Self-Attention and FFN layer has an Add & Normalization layer after it.

Thus, the way of building the model is from its smallest components to bigger ones. The function for Scaled-Dot-Product Attention is built first that would create Self-Attended output. Then, a Multi-Head Attention class is built which would perform self-attention incorporating the number of heads as well. A function for Feed-Forward Network layer is coded afterwards.

As the components of Encoding layer and Decoding layer were made, we move on to the Encoding layer and Decoding layer themselves that would have the Multi-Head Attention and Feed-Forward Network layer as per the architecture described in Section-3.

These individual layers would form up the Encoder and Decoder respectively and so this is the next step wherein the number of Encoding/Decoding layers is given as an additional input to the class. The Encoder and Decoder would also have the embedding layer and hold the positional encodings as well that get added to the embeddings. These are then passed on to the Encoding/Decoding layers.

Finally, compiling everything together is the Transformer Class that will take the batch of source and target sequences, pass them to the Encoder and Decoder and learn how to translate.



## Section 6: Defining Model Parameters

There are several model parameters that need to be defined along with some helper functions.

The optimizer Adam is used to optimize but it has a varied learning rate which first increases linearly up until the warmup steps (4000) and then decreases proportionally to the inverse square root of the step number. This is implemented using the LearningRateSchedule module provided by TF.

A simple Categorical Cross-Entropy is provided which compares the predicted word (and the probabilities of words) with what the actual word is.

Some helper functions built are:-

1. Shuffler - meant to shuffle the training dataset
2. Generator - meant to generate batches of source and target data to provide to the model for training/validation
3. Create Masks - Using the batch of source and target data, the padding and look-ahead masks for Encoder and Decoder will be created through this

Other note-worthy parameters are - 

num_layers :- These are the number of layers provided to Encoder and Decoder (i.e. number of Encoding and Decoding layers)
d_model :- This is the dimensionality of model i.e. the embedding size of token for both source and target sequences
num_heads :- The number of heads inside each Encoding and Decoding layer that would create Self-Attention over the sequences

As the project is meant to be descriptive, the parameters taken here are scaled.



## Section 7: Training the Model

As stated earlier, an Object-oriented approach is applied as the Tensorflow-Keras libaries don't have predefined layers that can incorporate this architecture.

Once the classes are formulated and model has been built (along with loss calculation and optimizer defined), the tf.GradientTape() function will be implemented to train the model on each batch and update the gradients of the trainable parameters. The model is trained for 20 epochs (with 2 patience), with shuffling of training data in each epoch. After each epoch, testing(validation) is performed on unseen data to keep track of how the model is learning.



## Section 8: Model Inference

**1. Greedy Search**

Greedy Search is the most basic inference algorithm. It takes the word with the highest probability at each output from the decoder input. This word is then fed to the next time step of the decoder to predict the next word until we hit the 'end' signal

Some outputs from Greedy Search -

<ins>OUTPUT-1</ins> <br>
Input:         : this is coming good <br>
Prediction     : ca vient bien <br>
Ground truth   : ça vient bien <br>

<ins>OUTPUT-2</ins> <br>
Input:         : You may speak <br>
Prediction     : vous pouvez parler <br>
Ground truth   : Vous pouvez parler <br>

<ins>OUTPUT-3</ins> <br>
Input:         : it is very cold here <br>
Prediction     : il fait tres froid ici <br>
Ground truth   : il fait vraiment froid ici <br>


**2. Beam Search**

Beam Search is slightly complicated. It produces K (which is user-defined) number of translations based on highest conditional probabilities of the words

The algorithm is explained in one of my other projects (link - https://github.com/garganm1/Neural-Machine-Translation-with-Bahdanau-Attention). Please see to understand how the algorithm works.

Some outputs from Beam Search -

Some outputs from Beam Search -

- evaluate_with_beam('it is very cold here', 5) :-

  - Translated Sentence 1 : ce tres froid; Associated Neg Log Probability: [4.7579575] 

  - Translated Sentence 2 : il tres froid ici; Associated Neg Log Probability: [0.33498693] 

  - Translated Sentence 3 : c tres rhume la; Associated Neg Log Probability: [4.0017323] 

  - Translated Sentence 4 : on tres froid ici; Associated Neg Log Probability: [4.830948] 

  - Translated Sentence 5 : ca a froid ici; Associated Neg Log Probability: [3.0651] 

- evaluate_with_beam('You may speak', 5)

  - Translated Sentence 1 : vous parler; Associated Neg Log Probability: [1.0982362] 

  - Translated Sentence 2 : il parler; Associated Neg Log Probability: [1.2011795] 

  - Translated Sentence 3 : tu vous parler; Associated Neg Log Probability: [2.6814516] 

  - Translated Sentence 4 : ca peut parler; Associated Neg Log Probability: [3.2999525] 

  - Translated Sentence 5 : on vous peut parler; Associated Neg Log Probability: [4.1590004] 


- evaluate_with_beam('this is very good', 5)

  - Translated Sentence 1 : c tres bon; Associated Neg Log Probability: [0.6709021] 

  - Translated Sentence 2 : il fort bien; Associated Neg Log Probability: [2.0666888] 

  - Translated Sentence 3 : ca vraiment beau; Associated Neg Log Probability: [2.3271205] 

  - Translated Sentence 4 : ce agit bonne; Associated Neg Log Probability: [4.0318303] 

  - Translated Sentence 5 : voila tres bien; Associated Neg Log Probability: [2.9427016] 


This concludes the implementation of Transformer architecture on the application of language translation. Transformers are a revolutionary 'technology' especially in the field of NLP. Recently, some progress has been made on its application in Computer Vision problems as well such as - https://arxiv.org/pdf/2010.11929.pdf. Nevertheless, they have been a great step in improving and elongating the research depth in Deep Learning.
