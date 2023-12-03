#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# Frangil Ramirez Koteich (fraramir)
# Nilambar Halder Tonmoy (nhalder)
# Paul Coen (pcoen)

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


class Solver:
    # Adding init function to setup the needed variables to use within the class.
    def __init__(self):
        # Pixel dictionary below will store the ground-truth (noise-free) pixel distribution of each character.
        self.pixel_counts = {}
        # Available characters
        self.truth = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        )
        # Bounding-box size for each char
        self.WIDTH = 14
        self.HEIGHT = 25

        # TODO: Define what this is actually storing, what it said before is wrong.
        self.char_pixels = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25,
            "a": 26,
            "b": 27,
            "c": 28,
            "d": 29,
            "e": 30,
            "f": 31,
            "g": 32,
            "h": 33,
            "i": 34,
            "j": 35,
            "k": 36,
            "l": 37,
            "m": 38,
            "n": 39,
            "o": 40,
            "p": 41,
            "q": 42,
            "r": 43,
            "s": 44,
            "t": 45,
            "u": 46,
            "v": 47,
            "w": 48,
            "x": 49,
            "y": 50,
            "z": 51,
            "0": 52,
            "1": 53,
            "2": 54,
            "3": 55,
            "4": 56,
            "5": 57,
            "6": 58,
            "7": 59,
            "8": 60,
            "9": 61,
            "(": 62,
            ")": 63,
            ",": 64,
            ".": 65,
            "-": 66,
            "!": 67,
            "?": 68,
            '"': 69,
            "'": 70,
            " ": 71,
        }

        # Assigning an index (i.e. label) to each character. Will be used in HMM model.
        self.char_index = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25,
            "a": 26,
            "b": 27,
            "c": 28,
            "d": 29,
            "e": 30,
            "f": 31,
            "g": 32,
            "h": 33,
            "i": 34,
            "j": 35,
            "k": 36,
            "l": 37,
            "m": 38,
            "n": 39,
            "o": 40,
            "p": 41,
            "q": 42,
            "r": 43,
            "s": 44,
            "t": 45,
            "u": 46,
            "v": 47,
            "w": 48,
            "x": 49,
            "y": 50,
            "z": 51,
            "0": 52,
            "1": 53,
            "2": 54,
            "3": 55,
            "4": 56,
            "5": 57,
            "6": 58,
            "7": 59,
            "8": 60,
            "9": 61,
            "(": 62,
            ")": 63,
            ",": 64,
            ".": 65,
            "-": 66,
            "!": 67,
            "?": 68,
            '"': 69,
            "'": 70,
            " ": 71,
        }

        # Initializing counter for each char
        self.char_counts = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
            "E": 0,
            "F": 0,
            "G": 0,
            "H": 0,
            "I": 0,
            "J": 0,
            "K": 0,
            "L": 0,
            "M": 0,
            "N": 0,
            "O": 0,
            "P": 0,
            "Q": 0,
            "R": 0,
            "S": 0,
            "T": 0,
            "U": 0,
            "V": 0,
            "W": 0,
            "X": 0,
            "Y": 0,
            "Z": 0,
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0,
            "f": 0,
            "g": 0,
            "h": 0,
            "i": 0,
            "j": 0,
            "k": 0,
            "l": 0,
            "m": 0,
            "n": 0,
            "o": 0,
            "p": 0,
            "q": 0,
            "r": 0,
            "s": 0,
            "t": 0,
            "u": 0,
            "v": 0,
            "w": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0,
            "8": 0,
            "9": 0,
            "(": 0,
            ")": 0,
            ",": 0,
            ".": 0,
            "-": 0,
            "!": 0,
            "?": 0,
            '"': 0,
            "'": 0,
            " ": 0,
        }

        # Initializing counter for total number of char
        self.total_char_counted = 0

        self.transition_probabilities = np.zeros(
            (len(self.truth) + 1, len(self.truth)), dtype=int
        )

    # train on text file provided in Part 1 (ignoring part-of-speech tags)
    def train(self, data):
        # for each sentence
        # print(len(data))
        for i in range(len(data)):
            # a sentence of n char typically contains n - 1 blank spaces ' '
            # self.char_counts[' '] += len(data[i]) - 1
            # self.total_char_counted += len(data[i]) - 1

            # print(data[i])

            prior_char = ""
            for character in data[i]:
                # print(character)

                # if self.char_counts[character] != len(data):
                self.char_counts[character] += 1
                self.total_char_counted += 1

                if prior_char == "":
                    self.transition_probabilities[
                        len(self.truth), self.char_index[character]
                    ] += 1
                else:
                    self.transition_probabilities[
                        self.char_index[prior_char], self.char_index[character]
                    ] += 1

                prior_char = character

        # print(self.transition_probabilities)

        # number of blank spaces is extremely large, which causes P(' ') to also be very large
        # thus, the model outputs ' ' too often, even if the pixels of the input image do not have a high similarity score
        # decrement counter to compensate
        self.char_counts[" "] /= 3

    # compute similarity between input pixels and ground-truth pixels
    def compute_similarity(self, goal, observation):
        similarity = 0

        # image is 2D array of pixels
        # (actually it is a 2D array of chars due to pre-processing, but that does not matter)
        for i in range(len(goal)):
            for k in range(len(goal[i])):
                # bounding-box for chars is too large, so too many white pixels surrounding each char
                # thus, similarity scores for all chars were too high
                # we choose to only consider pixels that are black in at least 1 of the 2 images
                if (goal[i][k] == "*") or (observation[i][k] == "*"):
                    if goal[i][k] == observation[i][k]:
                        similarity += 1
                    else:
                        # similarity scores were still too large,
                        # so we choose to penalize the network more heavily for distinct pixels
                        similarity -= 1
        # normalize similarity score
        if similarity > 0:
            return similarity / (self.HEIGHT * self.WIDTH)
        # if negative just return very small positive similarity
        elif similarity < 0:
            return np.abs(1 / (similarity * self.HEIGHT * self.WIDTH))
        else:
            return 0

    def compute_simple_prob(self, observation):
        max = ("", -math.inf)  # negative inf
        counter = 0
        for i in range(len(self.truth)):
            x = self.compute_similarity(self.char_pixels[self.truth[i]], observation)
            y = self.char_counts[self.truth[i]]
            # divisor can be set to 1 / 73 since there are 73 distinct characters
            # however, it would be a constant factor which does not affect the result
            # so, we chose to remove it
            divisor = 1

            #
            # np.where code snippet below adapted from
            # https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
            # (just making sure we do not divide by 0)
            #
            # using log prob for improved accuracy and speed
            x = np.where(x > 0, x, -math.inf)  # negative inf
            y = np.where(y > 0, y, -math.inf)  # negative inf
            temp_bayes = (
                np.log2(x, out=x, where=x > 0)
                + (np.log2(y, out=y, where=y > 0) - np.log2(self.total_char_counted))
            ) - np.log2(1)
            #
            # end of adapted code
            # NOTE: adapted is only the np.log and np.where code, but all else is ours
            #

            if temp_bayes > max[1]:
                max = (self.truth[i], temp_bayes)

        return max

    def compute_HMM(self, test_letters):
        # This stores the sequence of labels to output
        return_sequence = []

        # Keep track of prior labels
        # It is initially empty at the start of a sentence to account for initial word probabilities stored in the transition matrix
        prior_labels = []

        # These are used to reconstruct the best sequence found.
        # At the bottom of this function are print statements to show what this looks like.
        max_values_at_level = np.zeros((len(test_letters), len(self.truth)))
        max_char_value_prior_indexes = np.zeros(
            (len(test_letters), len(self.truth)), dtype=int
        )

        # Keep track of the max transition probabilities for each label (class) value
        max_labels_prob = np.zeros(len(self.truth))

        for letter_index, letter in enumerate(test_letters):
            for i in range(len(self.truth)):
                x = self.compute_similarity(self.char_pixels[self.truth[i]], letter)
                y = self.char_counts[self.truth[i]]
                # divisor can be set to 1 / 73 since there are 73 distinct characters
                # however, it would be a constant factor which does not affect the result
                # so, we chose to remove it
                divisor = 1 / len(self.truth)

                #
                # np.where code snippet below adapted from
                # https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
                # (just making sure we do not divide by 0)
                #
                # using log prob for improved accuracy and speed
                x = np.where(x > 0, x, -math.inf)  # negative inf
                y = np.where(y > 0, y, -math.inf)  # negative inf
                bayes_result = (
                    np.log2(x, out=x, where=x > 0)
                    + (
                        np.log2(y, out=y, where=y > 0)
                        - np.log2(self.total_char_counted)
                    )
                ) - np.log2(divisor)
                #
                # end of adapted code
                # NOTE: adapted is only the np.log and np.where code, but all else is ours
                #
                # print(bayes_result)

                # bayes_result = math.exp(bayes_result)

                bayes_results_v = []
                if len(prior_labels) == 0:
                    # This is the initial transition probabilities when we don't have a previous label value
                    if (
                        sum(
                            self.transition_probabilities[
                                max(self.char_index.values()) + 1, :
                            ]
                        )
                        == 0
                    ):
                        # Make the value very small but not zero to avoid math errors
                        transition_value = 0
                    else:
                        transition_value = self.transition_probabilities[
                            max(self.char_index.values()) + 1, i
                        ] / sum(
                            self.transition_probabilities[
                                max(self.char_index.values()) + 1, :
                            ]
                        )

                    if transition_value == 0:
                        bayes_results_v.append(-math.inf)
                    else:
                        bayes_results_v.append(np.log2(transition_value) + bayes_result)
                else:
                    possible_results = []
                    # These are the transition probabilities from the prior max for each label value
                    for row_label, prob_val in prior_labels:
                        if (
                            sum(
                                self.transition_probabilities[
                                    self.char_index[row_label], :
                                ]
                            )
                            == 0
                        ):
                            # Make the value very small but not zero to avoid math errors
                            transition_value = 0
                        else:
                            transition_value = self.transition_probabilities[
                                self.char_index[row_label], i
                            ] / sum(
                                self.transition_probabilities[
                                    self.char_index[row_label], :
                                ]
                            )

                        # print(transition_value)
                        if transition_value == 0:
                            possible_results.append(-math.inf)
                        else:
                            possible_results.append(
                                np.log2(transition_value) + bayes_result + prob_val
                            )

                    # Only look at the max value from the above transitions
                    bayes_result_v_indexes = np.argmax(possible_results)
                    # print(bayes_result_v_indexes.shape)

                    # If there are multiple, include all of them (only the top first value will be selected)
                    if bayes_result_v_indexes.shape == ():
                        bayes_results_v = [possible_results[bayes_result_v_indexes]]
                        max_char_value_prior_indexes[letter_index, i] = self.char_index[
                            prior_labels[bayes_result_v_indexes][0]
                        ]
                    else:
                        for possible_results_index in bayes_result_v_indexes:
                            bayes_results_v.append(
                                possible_results[possible_results_index]
                            )

                        max_char_value_prior_indexes[letter_index, i] = self.char_index[
                            prior_labels[bayes_result_v_indexes[0]][0]
                        ]

                # Include the max as the new prior probability for this label
                max_labels_prob[i] = max(bayes_results_v)
                max_values_at_level[letter_index, i] = max(bayes_results_v)

            # Set all the new max prior probabilities for the next word to use
            prior_labels = []
            for label_index, label_prob in enumerate(max_labels_prob):
                prior_labels.append((self.truth[label_index], label_prob))

            # Output the max value, for ties, take the first occurrence as given below
            max_pred_label_index = np.argmax(max_labels_prob)

            # The return value is the one that has the highest probability of occurrence, for identical probabilities, it takes the first occurrence.
            if max_pred_label_index.shape == ():
                return_sequence.append(self.truth[max_pred_label_index])
            else:
                return_sequence.append(self.truth[max_pred_label_index[0]])

        # print(max_values_at_level)
        # print(max_char_value_prior_indexes)
        # print(max_labels_prob)
        # print(max_pred_label_index)

        last_return = return_sequence[-1]
        return_sequence = []
        last_index = self.char_index[last_return]
        # print(last_index)

        # Reconstruct the path in reverse
        for max_path in np.flip(max_char_value_prior_indexes[:], axis=0):
            # print(f'{last_index} | {max_path}')
            return_sequence.append(self.truth[last_index])
            last_index = max_path[last_index]

        # Reverse the reconstructed path to be in the correct order.
        return_sequence.reverse()

        # print(return_sequence)
        return "".join(return_sequence)

    def load_training_letters(self, fname):
        letter_images = load_letters(fname)
        for i in range(0, len(self.truth)):
            self.char_pixels[self.truth[i]] = letter_images[i]


# NOTE: ADDING NOISE
#   for j in range(0, len(TRAIN_LETTERS)):
#       x = random.randint(0, 100)
#      if x == 5:
#          y = random.randint(0, len(letter_images[j]))
#         if letter_images[j][y] == ' ':
#             letter_images[j][y] = '*'
#         else:
#              letter_images[j][y] = ' '
#           break;


def read_data(fname):
    exemplars = []
    file = open(fname, "r")
    for line in file:
        data = tuple([w for w in line.split()])
        possible_tokens = data[0::2]
        sentence = ""
        prior_character = ""
        for token in possible_tokens:
            if prior_character != "":
                if token[0] not in "(),.-!?\"' ":
                    sentence += " "

            for character in token:
                if (
                    character
                    in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
                ):
                    sentence += character
                    prior_character = character
                # Handle the case of forward ticks and convert them into single quotes (a character actually in the dataset)
                if character in "`":
                    sentence += "'"
                    prior_character = "'"

        # print(sentence)
        exemplars.append(sentence)
    return exemplars


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(
        0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH
    ):
        result += [
            [
                "".join(
                    [
                        "*" if px[x, y] < 1 else " "
                        for x in range(x_beg, x_beg + CHARACTER_WIDTH)
                    ]
                )
                for y in range(0, CHARACTER_HEIGHT)
            ],
        ]
    return result


# if input image is mostly white pixels just return a blank space
# this reduces number of false positives
def detect_spaces(goal):
    similarity = 0
    for i in range(len(goal)):
        for k in range(len(goal[i])):
            if goal[i][k] == " ":
                similarity += 1
    return similarity / (CHARACTER_HEIGHT * CHARACTER_HEIGHT)


#####
# main program
if len(sys.argv) != 4:
    raise Exception(
        "Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png"
    )

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

test_letters = load_letters(test_img_fname)
data = read_data(train_txt_fname)
solver = Solver()
solver.load_training_letters(train_img_fname)
solver.train(data)

# print(solver.char_counts)

print("Simple: ", end="")
for c in range(len(test_letters)):
    if detect_spaces(test_letters[c]) > 0.55:
        print(" ", end="")
    else:
        print(solver.compute_simple_prob(test_letters[c])[0], end="")
print()
print("   HMM: ", end="")
token_to_send = []
for c in range(len(test_letters)):
    if detect_spaces(test_letters[c]) > 0.55:
        if len(token_to_send) != 0:
            print(solver.compute_HMM(token_to_send), end="")
            token_to_send = []
        print(" ", end="")
    else:
        token_to_send.append(test_letters[c])

if len(token_to_send) != 0:
    print(solver.compute_HMM(token_to_send), end="")


# print("   HMM: " + solver.compute_HMM(test_letters))
print()

# print(detect_spaces(test_letters[len(test_letters)-7]))
## Below is just some sample code to show you how the functions above work.
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in test_letters[4] ]))
# print("\n".join([ r for r in solver.char_pixels[' ']]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print(test_letters[2])


# The final two lines of your output should look something like this:
# print("Simple: " + "Sample s1mple resu1t")
# print("   HMM: " + "Sample simple result")
