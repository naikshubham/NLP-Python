import numpy as np

def get_max_len(names):
    """
    Function to return length of the longest name.
    Input: list of names
    Output: length of the longest name
    """

    # create a list to contain all the name lengths
    length_list=[]

    # Iterate over all names and save the name length in the list.]
    for l in names:
        length_list.append(len(l))

    # Find the maximum length
    max_len = np.max(length_list)

    # return maximum length
    return max_len
