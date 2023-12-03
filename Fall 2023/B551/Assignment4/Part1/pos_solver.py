###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids:
# Frangil Ramirez Koteich (fraramir)
# Nilambar Halder Tonmoy (nhalder)
# Paul Coen (pcoen)



import random
import math
import statistics

# Adding numpy for an easy way of storing the transition probabilities between a sequence.
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # Adding init function to setup the needed variables to use within the class.
    def __init__(self):
        # In this we want to learn the "prior" values for each word and what tags they commonly are (what is their probability of occurrence).
        self.word_counts = {}

        # I wanted to make this dynamic to store the transition probabilities for arbitrary classes, but for now this will work for Part 1
        # The first index is the prior value (rows)
        # The second index is the next value (columns)
        # The 13th value is the initial transition probability (from the first word in a sentence)
        self.transition_probabilities = np.zeros((13, 12), dtype=int)
        # Might be needed later, adding for now.
        # This is just the number of boxes in the above matrix.
        #self.transition_total = 144

        # Using this to index the numpy array.
        # This gives an idea of how row and column values are given too (each unique number is a unique part-of-speech)
        self.label_index = {'det': 0, 'noun': 1, 'adj': 2, 'verb': 3, 'adp': 4, '.': 5, 'adv': 6, 'conj': 7, 'prt': 8, 'pron': 9, 'num': 10, 'x': 11}

        # The same as the above, except you can use numbers to index the label values
        self.label_index_reverse = {0:'det', 1:'noun', 2:'adj', 3:'verb', 4:'adp', 5:'.', 6:'adv', 7:'conj', 8:'prt', 9:'pron', 10:'num', 11:'x'}

        # Used for prior probability for all words
        self.label_counts = {'det': 0, 'noun': 0, 'adj': 0, 'verb': 0, 'adp': 0, '.': 0, 'adv': 0, 'conj': 0, 'prt': 0, 'pron': 0, 'num': 0, 'x': 0}

        # Used for marginal probability for all words
        self.total_words_counted = 0


    # Calculate the log of the posterior probability of a given sentence
    def posterior(self, model, sentence, label):
        #print(model)

        if model == "Simple":
            # Keep a total for output
            bayes_result_total = 0
            # For each word, find it's result (if it is in the training data) and add it to the log sum
            for word, label_val in zip(sentence, label):
                if word not in self.word_counts.keys():
                    # Handle the case of a word not being in the training dataset, don't add anything
                    bayes_result_total += 0
                else:
                    # If it is in the training data, calculate the bayes result and (as long as it is more than 0) calculate the log and add it to the log sum.
                    temp_bayes = ((self.word_counts[word][label_val] / self.label_counts[label_val]) * (self.label_counts[label_val] / sum(self.label_counts.values()))) / (self.word_counts[word]["total_words"] / self.total_words_counted)
                    bayes_result_total += temp_bayes

            if temp_bayes <= 0:
                return -math.inf
            else:
                return math.log(bayes_result_total)

        elif model == "HMM":
            HMM_total = 0

            # Keep track of the prior label to account for transition probabilities and their influence.
            prior_label = ("", 1)

            for word, label_val in zip(sentence, label):
                if word not in self.word_counts.keys():
                    # Handle the case of a word not being in the training dataset
                    # Don't add any value to the log sum (this value is undefined normally), but ensure the transition probabilities pass through completely.
                    HMM_total += 0
                    temp_bayes = 1
                else:
                    # Calculate the bayes result
                    temp_bayes = ((self.word_counts[word][label_val] / self.label_counts[label_val]) * (self.label_counts[label_val] / sum(self.label_counts.values()))) / (self.word_counts[word]["total_words"] / self.total_words_counted)
                    
                    # Determine if this is the first word or not.
                    if (prior_label[0] == ""):
                        # If it is, use first-word transition probabilities
                        transition_prob = self.transition_probabilities[max(self.label_index.values()) + 1, self.label_index[label_val]] / sum(self.transition_probabilities[max(self.label_index.values()) + 1, :])
                    else:
                        # If not, use prior word to current word transition probabilities
                        transition_prob = self.transition_probabilities[self.label_index[prior_label[0]], self.label_index[label_val]] / sum(self.transition_probabilities[self.label_index[prior_label[0]], :])

                    # Multiply the result with the original bayes value
                    temp_bayes *= transition_prob * prior_label[1]

                    # Same as the simplified model, take the log as long as it is more than 0, otherwise add nothing.
                    HMM_total += temp_bayes

                # Set prior label for the next word
                prior_label = (label_val, temp_bayes)

            if (temp_bayes <= 0):
                return -math.inf
            else:
                return math.log(HMM_total)
            
        else:
            print("Unknown algo!")


    # Do the training!
    def train(self, data):
        # For more details on the variables, see the __init__ function.
        for (sentence, labels) in data:
            # Set prior_label to none at the start of each sentence to calculate start word probabilities.
            prior_label = None
            for word, label in zip(sentence, labels):
                # If a word isn't already in the dictionary, add it and include the value it has for the label and total word count of the individual word.
                if word not in self.word_counts.keys():
                    self.word_counts[word] = {'det':  0, 'noun': 0, 'adj': 0, 'verb': 0, 'adp': 0, '.': 0, 'adv': 0, 'conj': 0, 'prt': 0, 'pron': 0, 'num': 0, 'x': 0, 'total_words': 0}
                self.word_counts[word][label] += 1
                self.word_counts[word]['total_words'] += 1

                # Capture the transition probabilities
                # The row indexes the prior word, the column indexes the current word. [row, column]/[prior_word, current_word] for indexing
                if (prior_label != None):
                    self.transition_probabilities[self.label_index[prior_label], self.label_index[label]] += 1
                else:
                    self.transition_probabilities[max(self.label_index.values()) + 1, self.label_index[label]] += 1
                
                prior_label = label

                # Capture the total sum of a given label
                self.label_counts[label] += 1

                # Capture the total sum of all words seen in the data
                self.total_words_counted += 1

        #print(self.word_counts)
        #print(self.transition_probabilities)

        # The reason we are storing information in the above format is that it allows for storing the likelihood of a given word as the integer values.
        # This allows for greater precision in how we store the probabilities and avoids the initial issues with floating point rounding for small values.
        # The simplified model can directly look up words with this method and determine a label.
        # NOTE: A note should be that if a label isn't in the training data, we likely should just assign it a (deterministic) random label for the simplified model.


    # Functions for each algorithm.
    # Return the most likely part-of-speech for each word.
    def simplified(self, sentence):
        return_sequence = []
        # This should just be a maximum likelihood estimate per-word.
        for word in sentence:
            if word not in self.word_counts.keys():
                #print(f'Word not found: {word}')
                # If a word isn't in the training dataset, return the most common part-of-speech (in this training data it was 'noun')
                return_sequence.append('noun')
            else:
                #print(f'{word}')
                # Determine the max label
                max_label = ("", 0)
                for label in self.label_index.keys():
                    # This is P(S | W) = (P(W | S) * P(S)) / P(W) where S is the part-of-speech tag and W is the word
                    bayes_result = ((self.word_counts[word][label] / self.label_counts[label]) * (self.label_counts[label] / sum(self.label_counts.values()))) / (self.word_counts[word]["total_words"] / self.total_words_counted)
                    #print(f'{word} - {temp_val}')
                    if (max_label[1] < bayes_result):
                        max_label = (label, bayes_result)
                
                # Set the label to the max result found
                return_sequence.append(max_label[0])
        
        return return_sequence


    # Functions for each algorithm.
    # Return the most likely part-of-speech for each word.
    def hmm_viterbi(self, sentence):
        # This stores the sequence of labels to output
        return_sequence = []
        
        # Keep track of prior labels
        # It is initially empty at the start of a sentence to account for initial word probabilities stored in the transition matrix
        prior_labels = []
        
        # These are used to reconstruct the best sequence found.
        # At the bottom of this function are print statements to show what this looks like.
        max_values_at_level = np.zeros((len(sentence), 12))
        max_value_label_prior_indexes = np.zeros((len(sentence), 12), dtype=int)

        # Keep track of the max transition probabilities for each label (class) value
        max_labels_prob = np.zeros(12)

        for word_index, word in enumerate(sentence):
            for label in self.label_index.keys():
                # Calculate the result for the individual value
                if word not in self.word_counts.keys():
                    bayes_result = 1 # Consider all values for a word not in the training dataset (I.E. focus on the other components like the transition value and the previous probability we calculated).
                else:
                    # Same as line 169:
                    # This is P(S | W) = (P(W | S) * P(S)) / P(W) where S is the part-of-speech tag and W is the word
                    bayes_result = ((self.word_counts[word][label] / self.label_counts[label]) * (self.label_counts[label] / sum(self.label_counts.values()))) / (self.word_counts[word]["total_words"] / self.total_words_counted)

                # Multiply it by the transition probabilities to determine the correct possible label.
                bayes_results_v = []
                if len(prior_labels) == 0:
                    # This is the initial transition probabilities when we don't have a previous label value
                    transition_value = self.transition_probabilities[max(self.label_index.values()) + 1, self.label_index[label]] / sum(self.transition_probabilities[max(self.label_index.values()) + 1, :])
                    bayes_results_v.append(transition_value * bayes_result)
                else:
                    possible_results = []
                    # These are the transition probabilities from the prior max for each label value
                    for row_label, prob_val in prior_labels:
                        transition_value = self.transition_probabilities[self.label_index[row_label], self.label_index[label]] / sum(self.transition_probabilities[self.label_index[row_label], :])
                        #print(transition_value)
                        possible_results.append(transition_value * bayes_result * prob_val)

                    # Only look at the max value from the above transitions
                    bayes_result_v_indexes = np.argmax(possible_results)
                    #print(bayes_result_v_indexes.shape)

                    # If there are multiple, include all of them (only the top first value will be selected)
                    if bayes_result_v_indexes.shape == ():
                        bayes_results_v = [possible_results[bayes_result_v_indexes]]
                        max_value_label_prior_indexes[word_index, self.label_index[label]] = self.label_index[prior_labels[bayes_result_v_indexes][0]]
                    else:
                        for possible_results_index in bayes_result_v_indexes:
                            bayes_results_v.append(possible_results[possible_results_index])

                        max_value_label_prior_indexes[word_index, self.label_index[label]] = self.label_index[prior_labels[bayes_result_v_indexes[0]][0]]

                # Include the max as the new prior probability for this label
                max_labels_prob[self.label_index[label]] = max(bayes_results_v)
                max_values_at_level[word_index, self.label_index[label]] = max(bayes_results_v)
                

            # Set all the new max prior probabilities for the next word to use
            prior_labels = []
            for i, label_prob in enumerate(max_labels_prob):
                prior_labels.append((self.label_index_reverse[i], label_prob))
            
            # Output the max value, for ties, take the first occurrence as given below
            max_pred_label_index = np.argmax(max_labels_prob)

            # The return value is the one that has the highest probability of occurrence, for identical probabilities, it takes the first occurrence.
            if max_pred_label_index.shape == ():
                return_sequence.append(self.label_index_reverse[max_pred_label_index])
            else:
                return_sequence.append(self.label_index_reverse[max_pred_label_index[0]])

        #calc_prior = 0
        #for row_index, (row, word) in enumerate(zip(max_values_at_level, sentence)):
        #    print(f'{row_index} | {word} | {row}')
        #    calc_prior += max(row)
        #print(f'Posterior: {math.log(calc_prior)}')
        #print(f'{max_value_label_prior_indexes}')


        # Build the actual return sequence in reverse:
        # This is needed as the first return is independent of the sequence (determined by the init values)
        first_return = return_sequence[0]
        last_return = return_sequence[-1]
        return_sequence = []
        last_index = self.label_index[last_return]
        #print(last_index)
        
        # Reconstruct the path in reverse
        for max_path in np.flip(max_value_label_prior_indexes[:], axis=0):
            #print(f'{last_index} | {max_path}')
            return_sequence.append(self.label_index_reverse[last_index])
            last_index = max_path[last_index]
        
        # Reverse the reconstructed path to be in the correct order.
        return_sequence.reverse()

        #print(return_sequence)
        return return_sequence
        

    # No change needed in this part
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

