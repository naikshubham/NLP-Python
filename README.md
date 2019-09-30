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

### Basic feature extraction

#### Number of Characters
