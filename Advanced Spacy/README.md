
### The nlp object

```python
# import the english language class
from spacy.lang.en import English

# create the nlp object
nlp = English()
```

- we can use the nlp object like a function to analyze text. Contains the preprcoessing pipeline. It also includes language specific rules used for tokenizing the text into words and punctuation.

#### The Doc object
- When we process a text with the nlp object, spacy creates a Doc object - short for "document". 

```python
# created by processing a string of text with the nlp object
doc = nlp("Hello world!")

# iterate over tokens in a doc
for token in doc:
    print(token.text)
    
token = doc[1]

print(token.text)
```

#### The Span object
- A span object is a slice of the document consisting of one or more tokens. It's only a view of the doc and doesn't contain any data itself. To create a span we can use python slicing

```python
# a slice from the Doc is a span object
span = doc[1:4]

print(span.text)
```

#### Lexical attributes

```python
doc = nlp("It costs $5.")

print("Index: ", [token.i for token in doc])
print("Text: ", [token.text for token in doc])
print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])
```

### Statistical models
- Enables spacy to predict linguistic attributes in context e.g Part-of-speech tags, syntatic dependencies, named entities trained on labeled example texts.
- `en_core_web_sm` package is a small English model that supports all core capabilities and is trained on web text.

```python
import spacy

nlp = spacy.load('en_core_web_sm')
```

#### Predicting Part-of-speech tags

```python
import spacy

# load the small English model
nlp = spacy.load('en_core_web_sm')

# process the text
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)
```

- In spacy attributes that returns string ends with an underscore, attributes without the underscore return an ID.

#### Predicting the Syntactic Dependencies
- In addition to the part-of-the-speech tags, we can also predict how the words are related. For example, whether a word is the subject of the sentence or an object.
- Spacy uses a standardized labels scheme.
- The "dep underscore" attribute returns the predicted dependency label.

<img src="images/syntactic_dependencies.JPG" width="350" title="syntatic_dependencies">

- The pronoun she is a nominal subject attached to the verb - in this case, to "ate". The noun "pizza" is a direct object attached to the verb "ate". It is eaten by the subject "she".
- The determiner "the", also known as an article, is attached to the noun "pizza".

#### Predicting Named Entities

```python
# process a text
doc = nlp(u"Apple is looking at buying U.K startup for $1 billion")

#iterate over the predicted entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

#### The explain method
- To get definitions for the most common tags and labels.

```python
spacy.explain("GPE")
spacy.explain("NER")
spacy.explain("dobj")
```

### Rule based matching

#### Why not regular expressions
- Spacy's matcher let's us write rules to find words and phrases in text. Compared to regular expressions, the matcher workds with DOc and token objects instead of only strings.
- It's also more flexibe : we can search for texts but also other lexical attributes.
- We can even write rules that use the model's predictions. For example, find the word "duck" only if it's a verb, not a noun.

#### Match patterns
- Match patterns are lists of dictionaries. Each dictionary describes one token. The keys are the names of the token attributes, mapped to their expected values.
- E.g `[{'ORTH':'iPhone'}, {'ORTH':'X'}]` . In this example, we're looking for two tokens with the text "iPhone" and "X". We can also match on other token attributes.
- E.g `[{'LOWER':'iphone'}, {'LOWER':'x'}]`. Here, we're looking for two tokens whose lowercase forms equal "iphone" and "x".
- We can even write patterns using attributes predicted by the model. `[{'LEMMA':'buy'}, {'POS':'NOUN'}]` . Here we are matching a token with the lemma "buy", plus a noun. The lemma is the base form, so the pattern would match phrases like "buying milk" or "bought flowers".

#### Using the Matcher

```python
import spacy

# import the matcher
from spacy.matcher import Matcher

# load the model and create the nlp object
nlp = spacy.load('en_core_web_sm')

# initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# add pattern to the matcher
pattern = [{'ORTH':'iPhone'}, {'ORTH':'X'}]

# the first argument is a unique ID to identify which pattern was matched,
# second argument is an optional callback, we don't need here so we set it to None
# third argument is the pattern
matcher.add('IPHONE_PATTERN', None, pattern)

# process some text
doc = nlp("New iPhone X release date leaked")

# call the matcher on the doc, it returns a list of tuples 
matches = matcher(doc)

# each tuple consists of 3 values : the match ID, the start index and the end index of the matched span
# match_id : hash value of the pattern name
# start : start index of matched span
# end : end index of matched span
# iterate over the matches
for match_id, start, end in matches:
    # get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
```

#### Matching lexical attributes
- 5 tokens consisting of only digits. Three case-insensitive tokens for 'fifa', 'world' and 'cup' and a token that consists of punctuation. The pattern matches the token `"2018 FIFA World Cup:"`

```python
pattern = [{"IS_DIGIT":True}, {"LOWER":'fifa'}, {"LOWER":"world"}, {"LOWER":"cup"},
            {"IS_PUNCT":True}]
doc = nlp("2018 FIFA World Cup : France won!")
```

#### Matching other token attributes

```python
pattern = [{'LEMMA':'love', 'POS':'VERB'}, {'POS':'NOUN'}]

doc = nlp("I loved dogs but the now I love cats more")
```

- Here we have 2 tokens. A verb with the lemma "love", followed by a noun. This pattern matches "loved dogs" and "love cats".

#### Using operators and quantifiers

```python
pattern = [{'LEMMA':'buy'},
           {'POS':'DET', 'OP':'?'}, # optional : match 0 or 1 times
           {'POS':'NOUN'}]
doc = nlp("I bought a smartphone. Now I'm buying apps.")
```

- Operators and quantifiers lets us define how often a token should be matched. They can be added using the 'OP' key. Here the "?" operator makes the determiner token optional, so it will match a token with the lemma "buy", an optional article and a noun.
- "OP" can have one of the four values:

<img src="images/operators.JPG" width="350" title="operators">


























































































          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          








    
    


























```python

```
