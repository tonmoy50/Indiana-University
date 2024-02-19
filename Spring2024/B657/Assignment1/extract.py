import numpy as np
from PIL import Image
import sys


def load_images(path):
    image = Image.open(path)
    return image


def decode_answers(img, total_questions=85, offset_x=50, offset_y=50):
    """Decoding the answers using Steganography Algorithm"""
    img = np.array(img)
    answers = list()
    for i in range(total_questions):
        answer = list()
        for j in range(16):
            pixel = img[offset_x + i, offset_y + j]
            if pixel[0] > 200 or pixel[1] > 200 or pixel[2] > 200:
                answer.append("1")
            else:
                answer.append("0")
        answers.append(str(int("".join(answer), 2)))

    for i, answer in enumerate(answers):
        answers[i] = "".join(str(decode_answer_choices[x]) for x in answer)

    return answers


if __name__ == "__main__":
    img = load_images(sys.argv[1])
    output_txt_path = sys.argv[2]

    decode_answer_choices = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    answers = decode_answers(img)
    with open(output_txt_path, "w") as f:
        for i, answer in enumerate(answers):
            f.write(f"""{i+1} {answer}""" + "\n")
