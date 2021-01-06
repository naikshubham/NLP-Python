
## Natural Language Generation
- Represent the data sequentially and use neural network architecture to model your text data.Train a recurrent network to generate new text, character by character.
- Generation of texts in a certain style
- Machine translation
- Sentence or word auto completion
- Generation of textual summaries
- Automated chatbots

### Handling sequential data
- Sequential data is any kind of data where the order matters. Ex. Text data, Time Series data, DNA sequences.
- Names dataset - The sequence of characters follows some probability distribution which is not known to us. Our goal is to guess this distribution from the existing names and generate new names that are similar to the names in this dataset.
- ***Our goal is to train a model that will predict a new character given a set of characters as input.*** So the model must understand when a name starts and ends.
- ***World delimiters*** : We can use special characters that are not used in any name in the dataset to mark the start and the end. These are called the start and the end token respectively. `\t \n`

```python
# insert start & end token
data['name'] = data['name'].apply(lambda x:'\t' + x)
data['target'] = data['target'].apply(lambda x:x[1:len(n)] + '\n')
```

#### Vocabulary for names dataset
- ML models deals with numbers, so we need to convert these sequences of characters into suitable integer representations.
- ***Vocabulary*** : set of all unique characters used in the dataset

```python
def get_vocabulary(names):
    # define vocab as a set and include start and end token
    vocabulary = set(['\t', '\n'])
    # iterate over all names and all chars of each names
    for name in names:
        for c in name:
            if c not in all_chars:
                vocabulary.add(c)
    # return vocabulary
    return vocabulary
```

#### Character to integer mapping
- Sort the vocabulary and assign numbers in order.
- \t :0 , \n:1, a:2, b:3s and so on.

```python
ctoi = {char:idx for idx, char in enumerate(sorted(vocabulary))}
itoc = {idx:char for char,idx in enumerate(sorted(vocabulary))}
```

### Reccurent Neural Network
- RNN for baby name generator : generate the next character given the current character and the history as input.
- Suppose we want to generate the name John. In the first time step, we need to input the tab character which should generate `j` as output. In the 2nd timestep, we need to input `j` to the network which should return `o`, and the state will keep track that the charcters tab and j are already encountered. This will continue until every character of the name is processed.
- The inputs, outputs and states are represented by vectors. At each time-step, the network transforms the input vector into the output vector and the state vector is updated to reflect characters already encountered.

#### Encoding of the characters
- Each character can be represented by a vector of length equal to the vocabulary size. The vector will have a 1 at the index which is the mapping of that character. All other positions will have zeros. This is called one-hot encoding.

#### Number of time-steps
- Number of time-steps will be the length of the name.As the names have different lengths, the time-step can be made equal to the length of the longest name with the shorter names padded with zero after the newline.
- `get_max_len` function gets the length of the longest names, saving the lengths of the names in a list and finding out the maximum.

```python
def get_max_len(names):
    length_list=[]
    for l in names:
        length_list.append(len(l))
    max_len = np.max(length_list)
    return max_len
    
max_len = get_max_len(names)
```

#### Input and target vectors
- Input and the target vectors are 3-dimensional. The first dim is the number of names in the dataset, the second is the no. of time steps which is the length of the longest name. The third dim is the size of each one-hot encoded vector which is the vocab size.

#### Initialize the Input vector
- First define the input vector as a 3-dimensional zero vector. The first dim is the no. of names in the dataset, the second dim is the length of the longest name which defines our step size and the third dim is the size of the vocab.
- To fill this vector with data, we need to convert each character of each name to its one-hot encoded vector.

```python
input_data = np.zeros((len(names_df.input), max_len + 1, len(vocabulary)), dtype='float32')

# fill the vector data
for n_idx, name in enumerate(names.name):
    for c_idx, char in enumerate(name):
        input_data[n_idx, c_idx, char_to_idx[char]] = 1
```

#### Initialize the target vector

```python
target_data = np.zeros((len(names_df.names), max_len + 1, len(vocabulary)), dtype='float32')

for n_idx, name in enumerate(names.target):
    for c_idx, char in enumerate(name):
        target_data[n_idx, c_idx, char_to_idx[char]] = 1
```

#### Build and compile RNN
- Create a sequential model, then adding an RNN layer of 50 units. `return_sequences=True` makes sure that the RNN layer outputs a sequence and not just a single vector.
- This output sequence is then passed to a dense layer with softmax activation to generate the output. The softmax activation predicts prob values for each char in the vocab.
- The **TimeDistributed wrapper layer** is used to make sure the dense layers can handle three-dim input.
- Model is compiled using categorical cross entropy and adam optimizer. Categorical cross-entropy loss is used when we have more than two labels.
- Here the output will be a character from the vocabulary and so, the number of labels is the size of the vocabulary.

```python
model = Sequential()
model.add(SimpleRNN(50, input_shape=(max_len +1, len(vocabulary)), return_sequences=True)
model.add(TimeDistributed(Dense(len(vocabulary), activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

### Inference using RNN
- Train the model and get predictions from the trained model.

#### Train RNN
- We can use the keras fit function to train the model.
- Batch size: number of samples after which the parameters are adjusted.
- Epochs : specifies the number of times the full dataset will be iterated.

```python
model.fit(input_data, target_data, batch_size=128, epochs=15)
```

#### Predict first character
- We trained the model in such a way that it'll produce the next char given the current char as input. And, the first char is the tab char which is the start token. We can feed the tab char to the network and get the most probable next char as output.
- We can create a 3-d zero vector for the output sequence and initialize it to contain the tab character.
- We can use the `predict_proba` method to get the probability distribution for the next char in the sequence.
- As we want to generate the first char after tab, we need to slice the prob distribution list to get the prob distribution for the first char.
- Now we can find the next char by sampling the vocabulary randomly using this prob distribution.

```python
output_seq = np.zeros((1, max_len + 1, len(vocabulary)))
output_seq[0, 0, char_to_idx['\t']] = 1  # output_seq[0, 0, 0] = 1
# prob distribution for the next char
probs = model.predict_proba(output_seq, verbose=0)[:,1,:]
# sample the vocab using the prob distribution
first_char = np.random.choice(sorted(list(vocabulary)), replace=False,p=probs.reshape(28)) 
```

#### Predict second character using the first
- We can use the generated first character to predict the second char in the sequence. The same process can be used to predict the most probable second char given the tab and the first char
- We can keep on generating characters in this manner until the end token or the newline is encountered. We can also put a constraint on the max length of the names and stop when the number of generated characters reaches this maximum.

```python
# insert first char in the sequence
output_seq[0, 1, char_to_idx[first_char]] = 1

# sample from prob distribution
probs = model.predict_proba(output_seq, verbose=0)[:,2,:]
second_char = np.random.choice(sorted(list(vocabulary)), replace=False,p=probs.reshape(28))
```


### Limitation of RNN
- Simple recurrent neural networks suffer from the problem of vanishing gradients where the gradients of the weights become smaller and smaller and eventually become zero as we move backward from the last time-step towards the first time-step.
- Recurrent neural networks also suffer from the exploding gradient problem where the gradient values of the weights become bigger and bigger as we move back-propagate towards the first time-step. 












