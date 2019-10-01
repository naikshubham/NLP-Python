# Feature Engineering for NLP in Python

## NLP Feature Engineering

#### One-hot encoding with pandas using get_dummies

```python
import pandas as pd

df = pd.get_dummies(df, columns=['sex'])
```

- Not mentioning columns will lead pandas to encode non-numerical features

#### Text pre-processing
- Converting to lowercase
- Converting to base-form

#### Vectorization 
- After pre-processing the text is converted into a set of numerical training features through a process known as vectorization.

#### Basic features
- Number of words
- Number of characters
- Average length of words
- Number of hashtags used in tweets

#### POS tagging
- Some NLP algo may require you to extract features for individual words.
- For instance, we may want to do part of speech tagging to know the different parts of speech present in the text
Word , POS
I -> Pronoun
have -> Verb
a -> Article
dog -> Noun

#### Named Entity Recognition
- Does a noun refer to person, organization or country?
- e.g Brian works at DataCamp.
- here there are two nouns, Brian and DataCamp

Noun, NER
Brian -> Person
DataCamp -> Organization

### Readability tests
- Used to determine the readability of any passage. In other words, at what educational level a person should be inorder to comprehend a particular piece of text.
- Readability test is done using a mathematical formula utilizing the word, syllable and sentence count.
- Used in fake news and opinion spam detection.

#### Readability test examples
- Flesch reading ease
- Gunning fog index
- Simple Measure of Gobbledygook(SMOG)
- Dale-Chall score

#### Flesch reading ease
- It is the oldest and widely used readability test. The score is dependent on two factors:
- **Greater the average sentence length, harder the text is to read.**
- **Greater the average number of syllables in a word, harder the text is to read.**
- Higher the Flesch reading ease score, the greater the readability.
- Therefore, the higher score indicates that the text is easier to understand.

#### Gunning fog index
- Score is dependent on the avarage sentence length.
- It uses the percentage of complex words(in place of average syllables) to compute its score.
- **Greater the percentage of complex words, harder the text is to read**
- Here complex words refer to those words who have 3 or more syllables.
- **Higher the score, lesser the readability** (Higher the score the more difficult is the passage is to understand)

### The textatistic library
- We can conduct these readability tests in python using the **textatistic** library.

```python
from textatistic import Textatistic

# create a Textatistic object
readability_scores = Textatistic(text).scores

# generate scores
print(readability_scores['flesch_score'])
print(readability_scores['gunningfog_score'])
```

### Text Preprocessing : Tokenization and Lemmatization
- Converting words into lowercase
- Removing leading and trailing whitespaces
- Removing punctuations
- Removing stopwords
- Expanding contractions
- Removing special characters (numbers, emojis, etc)

#### Tokenization using spaCy
- Splitting a string

```python
import spacy

# load the english model en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

string = "Hello! I don't know what I'm doing here"

# create a Doc object
doc = nlp(string)

# doc object contains required tokens and many other things
# Generate list of tokens
tokens = [token.text for token in doc]
print(tokens)
```
### Lemmatization using spacy
- Convert into its base form
- reducing, reduces, reduced, reduction -> reduce
- am, are, is -> be
- Lemmatization converts words with apostrophe into full -forms.
- n't -> not
- 've -> have
- Similar to extracting token using spacy

```python
lemmas = [token.lemma_ for token in doc]
```

### Part-of-speech tagging
- Used in Word-sense disambiguation to identify the sense of the word in a sentence.
- For instance consider the sentences : "The bear is a majestic animal" and "Please bear with me".
- Both sentences use the word **bear** , but means different things.
- POS tagging helps in identifying this distinction by identifying one bear as a **noun** and other as a **verb**.
- Also used in **sentiment analysis**, **question answering**, **fake news and opinion spam detection**

#### POS tagging
- It is a process of assigning every word or piece of text, its corresponding part of speech.
- e.g "Jane is an amazing guitatist"
- POS Tagging
Jane -> proper noun

is -> verb

an -> determiner

amazing -> adjective

guitatist -> noun

### POS tagging using Spacy
- POS tagging is easy to do using spacy models and performing it is almost identical to generating tokens or lemmas.

```python
import spacy

nlp = spacy.load('en_core_web_sm')

string = "Jane is an amazing guitarist"

doc = nlp(string)

# generate the list of tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)
```
- Spacy inferes the POS tagging based on the predictions given by its pre-trained model
- Accuracy of POS tagging depends on the data the model has been trained on.

#### POS annnotations in spacy
- PROPN -> proper noun
- DET -> determinant
- https://spacy.io/api/annotation















