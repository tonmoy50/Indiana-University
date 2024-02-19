import numpy as np
from PIL import Image
import sys


def load_images(path):
    image = Image.open(path)
    return image


def load_answers(path):
    with open(path, "r") as f:
        answers = [line.strip() for line in f.readlines()]
    return answers


def get_next_bin_val(char):
    """Get the next binary value of the character"""
    encoded_data = int("".join([str(answer_choices[i]) for i in char]))
    for i in bin(encoded_data)[2:].zfill(16):
        yield i


def encode_answers(img, answers, offset_x=50, offset_y=50):
    """Encoding the answers using Steganography Algorithm"""
    img = np.array(img)
    for i in range(len(answers)):
        correct_answer_bin_str = get_next_bin_val(answers[i].split(" ")[1])
        for j in range(16):
            pixel = img[offset_x + i, offset_y + j]
            bin_val = next(correct_answer_bin_str)
            if bin_val == "0":
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
            else:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
            img[offset_x + i, offset_y + j] = pixel

    return Image.fromarray(img)


if __name__ == "__main__":
    blank_img_path = sys.argv[1]
    answer_path = sys.argv[2]
    output_img_path = sys.argv[3]

    answer_choices = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    decode_answer_choices = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    blank_img = load_images(blank_img_path)
    answers = load_answers(answer_path)

    assert (
        blank_img.size[0] >= len(answers) and blank_img.size[1] >= 16
    ), "Image is too small to contain all the answers"

    img = encode_answers(blank_img, answers)
    img.save(output_img_path)
