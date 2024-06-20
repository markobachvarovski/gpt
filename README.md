# What it is
A basic implementation of a generative pre-trained transformer (also known as GPT) trained on an extrapolation of Shakespeare's play, "The tempest". The GPT aims to generate and output that is similar to Shakespeare's play. <br/> <br/>
To initiate the transformer, all you need to do is run ```main.py``` - follow [this](https://www.geeksforgeeks.org/how-to-run-a-python-script/) guide if you are unfamiliar with how to do this.<br/>
From here, the transformer will print an output before and after the training is complete, as well as an estimate for the loss at every 500 steps (more details on this below). **Note that the training will take a long time! (Close to 12 hours on my Intel i5 CPU)**

<br/>
Sample output is provided in the sections below, but first, it's important to understand how the transformer works

# How it works

## Tokenizer
A vocabulary is created which consists of the set of all characters in the training data. We use a character-level tokenizer, meaning that every individual character in the vocabulary has their own token (as opposed to multi-character tokenizers). The training data is tokenized and stored into a PyTorch tensor.

## Loading data
The data is broken up into chunks of size ```chunk_size```, and this is the maximum context length that the transformer can process. To speed up the loading, we can process ```batch_size``` number of chunks in parallel. The logic to get a single batch is as follows:<br/><br/>
Generate batch_size random numbers between ```0``` and ```data length - chunk_size``` (we choose this ending index to ensure that any random number chosen has at least ```chunk_size``` tokens to be processed). <br/> Let's call the random numbers selected ```i_n```, where n is in the interval ```(0, batch_size)```
.<br/>For every chunk, we will select chunk size independent tokens, beginning from ```data[i_n]```  and ending at ```data[i_n + chunk_size]```. We will also select the same amount of dependent tokens, offset by 1, i.e beginning at ```data[i_n + 1]``` and ending at ```data[i_n + chunk_size + 1]```. <br/>
The purpose of the dependent and independent tokens are to train the model which token should follow in the context of any given independent token.
### Example:
Let's make the simplest assumption that we have a batch of size 1, a chunk of size 8, and a data input of size 10,000.<br/> We generate a random index ```i_1``` from 1 to 10,000 - 8, let's take ```i_1 = 2783```. The independent variables we select will be ```data[2783:2791]```, and the dependent variables will be from ```data[2784:2792]```. We want to train the model to learn the following sequence:<br/>
In the context of ```data[2783]```, ```data[2784]``` must follow.<br/>
In the context of ```data[2783:2784]```, ```data[2785]``` must follow.<br/>
In the context of ```data[2783:2785]```, ```data[2786]``` must follow etc.

## Neural network - The Bigram Language Model
In this implementation, we use the Bigram Language Model. A bigram is any sequence of 2 words in a given dataset. In the sentence "I love apples and Nutella", the bigrams are: "I love", "love apples", "apples and", "and Nutella".<br/><br/>
Every BigramLanguageModel has an embedding table of size ```vocab_size``` by ```vocab_size```.<br/> We store the independent variables as embeddings. The output of the embedding table has the shape ```(batch_size, chunk_size, vocab_size)``` and what it stores are the scores of the tokens that could follow the current token.<br/><br/>
At the earliest stage, the embedding table does not consider any context, rather it looks at a single token in isolation and outputs an array of vocab_size scores for every following token. Since we have ```batch_size``` batches of ```chunk_size``` chunks, that explains why the shape is ```(batch_size, chunk_size, vocab_size)```. We aim to refine these scores by training the model, and to quantify the correctness of the scores, we must use a loss function

## Loss function
The loss function chosen is the negative log-likelihood loss function, more commonly known as the cross-entropy. The loss function is as follows:<br/><br/>
$` loss(token) = -ln(1/vocab\_size) `$,<br/><br/>
and the loss depends on the size of the vocabulary. In the data provided, ```vocab_size = 65```, so the loss expected is $`-ln(1/65) = 4.174`$

## New token generation
The new tokens are generated using the following algorithm:
1. Get the features from the embedding table with the current tokens available
2. Convert the features into probabilities using ```torch.softmax```
3. Sample one token from the probability distribution using ```torch.multinomial```
4. Append the new token to the array of current tokens
5. Repeat process ```max_new_tokens``` times

## Training the model - optimization step:
We use the ```AdamW``` optimizer from ```PyTorch```. We set the epoches to $`10,000`$ and the learning rate to $`1*10^-3`$ as default and adjust if needed.<br/><br/>
The training loop is as follows:
 1. For each step, evaluate the loss of the current data
 2. Reset the optimizer gradients to 0 from the previous step
 3. Get the gradients of every parameter from the loss (using ```loss.backwards()```)
 4. Update the parameters using those gradients (using ```optimizer.step()```)

## Optimization methods and further refining the output

### Self-attention
Self-attention is implemented as the dot product of the matrix of independent inputs x and the weights.<br/>
The weights matrix has size $`chunk_size^2`$, and is filled with the weights of every token processed so far in the chunk.<br/>
It is initialized as follows: 
 1. For every token that's processed, create an array with ```1```s on the interval ```[0:n]``` where ```n``` is the index of the current token
 2. Set the interval ```[n+1:chunk_size-1]``` to be ```0```
 3. Normalize the array so that all weights sum to 1
 4. Insert array into weights matrix at ```weights[n-1]```

**Example:**
Assume the first token is being processed, then for ```n = 1```, ```chunk_size = 4```, the array would be ```[1, 0, 0, 0]``` and this would be inserted at ```weights[0]```<br/>
When the second token is being processed, ```n = 2```, the array would be ```[0.5, 0.5, 0, 0]``` after normalization, inserted into the weights matrix at ```weights[1]```<br/>
For ```n = 3```, we have ```[0.33, 0.33, 0.33, 0]```, inserted into ```weights[2]```<br/>
Finally, for ```n = 4```, we have ```[0.25, 0.25, 0.25, 0.25]``` inserted into ```weights[3]```<br/><br/>
The initialized weights matrix for a chunk of ```chunk_size = 4```, would be:<br/>
```weights = [[  1,     0,    0,    0],```<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;```[0.5,   0.5,    0,    0],```<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;```[0.33, 0.33, 0.33,    0],```<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;```[0.25, 0.25, 0.25, 0.25]]```<br/><br/>
You might be wondering what the logic is behind this initialization. At the time of processing the ```n```-th token, all the following tokens are future tokens, and they are what we are trying to predict, so they carry no weight to our calculation and are set to 0. <br/>
The tokens from ```0``` to ```n``` all carry uniform weights since we don't have anything to compare them to, and this is exactly **the problem that self-attention tries to solve**, i.e. **how do we adjust the weights of the tokens based on previous context?**
<br/><br/>
We accomplish that by using 2 vectors, called the ```query``` and ```key``` vectors. <br/>
The query vector asks questions about the input data, and the key vector provides answers for them. The weights are then the dot product between the query vector and the transpose of the key vector.<br/>
At initialization, the weights of both vectors are sampled randomly from a uniform distribution, but as we train the model during the optimization step, the weights of both vectors are updated independently to minimize loss.<br/>
The initial weights are multiplied by $`1/sqrt(head\_size)`$ to control the variance and make sure softmax doesn't converge or diverge if one value differs too much from the rest. <br/><br/>
Finally, we also have a value vector. Instead of the weights being applied to the raw training data, we apply them to the value vector which functions as a wrapper for the raw data.<br/>
It converts the token data into a vector to be used for the final weighted output of the attention algorithm.<br/>
The query and key vectors are crucial to determining the weights, but the value vector actually contains the information from every token.

### Multi-headed attention:
Using one head of self-attention yields an improved although insufficient result. To refine further, we use multiple heads and concatenate the outputs of all heads.

### Feed-forward layers
There is a simple feed-forward layer added right after the self-attention step with the purpose of allowing the tokens to think on the data independently, after self-attending.<br/>
This layer consists of a several transformations in the following order:
1. A linear transformation is applied to every individual token's embedding (either the original or refined from previous training steps) which transforms the original features into a higher dimension
2. A non-linear activation function (specifically the Rectifier Activation Function - ```ReLU()```) is applied to the output of the linear transform, to capture non-linear relationships that a simple linear layer can not represent
3. A second linear transformation is applied to the non-linear function output to convert features into the original embedding dimension for compatibility with the other training steps
4. A dropout layer is added (explained in detail in the [Droupout](https://github.com/markobachvarovski/gpt/README.md#dropout) section)

### Residual connections
The current feed-forward layer transforms the input features given and maps them to an entirely new transformation at every training step. But this isn't exactly accurate - ideally we'd like to learn from past input to improve the transformation, rather than transforming it anew at every step.<br/>
The solution to this is simple: instead of outputting the transformation itself, output the sum of the transformation and the input features.<br/> 
This is known as a residual connection. In large neural networks, gradients can easily diminish to very small numbers or grow to very large ones the more layers there are to the net.<br/>
By summing the input at every step, we are keeping the original gradients present and are enabling the network to incrementally refine the features, instead of remapping them every time.
<br/>This allows the original gradients to propagate deeper into the network, and ensuring that we are training effectively at deeper levels. Let's give a specific example for visualization purposes:

Let's assume the original features/embeddings are as follows:<br/>
```x = [0.1, 0.2, 0.3]```

Applying the feed-forward layer, we end up with a transformation like so:<br/>
```x_ffwd = [0.0498, 0.0703, 0.0722]```

Without using residual connections, the output is simply the output of the feed-forward layer, so:<br/>
```output = x_ffwd = [0.0498, 0.0703, 0.0722]``` <br/><br/>
With residual connections, the output is the input features + the output of the feed-forward layer, so:<br/>
```output = x + x_ffwd = [0.1, 0.2, 0.3] + [0.0498, 0.0703, 0.0722] = [0.1498, 0.2703, 0.3722]```<br/><br/>

It's easy to see that the residual connection method yields an output closer to the input, and the input features weigh heavily into the output, thus allowing us to incrementally refine the output.<br/>
This is only one training step, so at every following step, the input becomes the current output, and it's easily noticeable that after several steps, using only the feed-forward output can sway in unexpected and incorrect directions, so using the residual connections method grounds the output to make sure we never get unexpected output.

### LayerNorm
Before being passed into the self-attention and feed-forward layers, the input features are normalized using LayerNorm, which helps stabilize the training data in the further layers.

### Dropout
Dropout is another method of optimizing the output, by dropping out certain features in the network to 0.<br/>
What this effectively does is it allows the network to train on a smaller (and different) sub-network at every training step.<br/>
A dropout rate controls the number of dropped neurons and every neuron has an equal chance to be dropped randomly.<br/><br/>

This particular method is quite effective at preventing overfitting, ensuring that the neural network doesn't rely on any particular set of neurons too heavily in comparison to others.
<br/>The dropout is added at three layers: as the last layer in the feed-forward layer, at the end of the multi-headed self-attention and right after calculating the weights in the forward pass at the Head level.

# Sample output
I generated a 100 token output before and after optimization <br/> <br/>
**Output before optimization:**

fgUEHSaEIIh?&IjbYvySm SjRiCei$UtC
vTgPddKeTEaE3 Y:o zuH X?:Wk?.$&i;Cjq!W-IPXblzlt!EvV&hRRiyawhZX'ZBD

The output before any optimization is fairly non-sensical and does not contain words or sentences.<br/>
The loss at the very first step for the training and actual data sets is 4.3379 and 4.3336, respectively. <br/>
After training for 5000 steps, the loss is evaluated as follows: <br/><br/>

| Step # | Training data loss | Actual data loss |
|--------| ------------------ | ---------------- |
| 0      | 4.3379             | 4.3336           |
| 500    | 2.4674             | 2.4779           |
| 1000   | 2.3165             | 2.3385           |
| 1500   | 2.1444             | 2.1867           |
| 2000   | 2.0170             | 2.0844           |
| 2500   | 1.9184             | 2.0151           |
| 3000   | 1.8404             | 1.9642           |
| 3500   | 1.7750             | 1.9108           |
| 4000   | 1.7324             | 1.8844           |
| 4500   | 1.6933             | 1.8493           |

<br/>
After 5000 steps, we've managed to get the loss down to 1.6933 for the training data and 1.8493 for the actual data, **resulting in a reduction in loss by ~57%!**
<br/><br/>
**Output after optimization:**<br/>

Jon vagen! Nuth Righald her mird.

JULIET:
But you will:
I sprjicke and to but adly sis.

WARWICHI:

<br/>
The output now resembles the training data more closely. We notice the structure is similar to the training data provided, and words and sentences can be extrapolated. Longer words are prone to more syntactical errors, but shorter words and names that come up often are generated correctly, such as "Juliet", "will, "you" etc. With a larger training set with a higher repetition of words, we can expect the syntactical mistakes to be decreased. The generation has no semantical meaning, although this is also something that can be improved by using larger training sets

# References:
["Let's build GPT:from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
