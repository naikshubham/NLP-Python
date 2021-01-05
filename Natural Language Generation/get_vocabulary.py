# Get vocabulary of Names dataset
def get_vocabulary(names):  
    # Define vocabulary to be set
    all_chars=set()
    
    # Add the start and end token to the vocabulary
    all_chars.add('\t')
    all_chars.add('\n')  
    
    # Iterate for each name
    for name in names:

        # Iterate for each character of the name
        for c in name:

            if c not in all_chars:
            # If the character is not in vocabulary, add it
                all_chars.add(c)

    # Return the vocabulary
    return all_chars