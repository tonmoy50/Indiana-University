import subprocess
import os
from grade import *


def test_grading(test_ground_truth_image_path, test_ground_truth_file_path):
    command = ["python", "./grade.py"] + [test_ground_truth_image_path, "output.txt"]
    subprocess.run(command)

    with open(test_ground_truth_file_path, "r") as f:
        answers = [line.strip() for line in f.readlines()]
    with open("output.txt", "r") as f:
        detected_answers = [line.strip() for line in f.readlines()]
    assert all(
        [answer == detected for answer, detected in zip(answers, detected_answers)]
    )


def test_extracting(test_ground_truth_image_path):
    command = ["python", "./grade.py"] + [test_ground_truth_image_path, "output.txt"]
    subprocess.run(command)

    with open("output.txt", "r") as f:
        detected_answers = [line.strip() for line in f.readlines()]

    print(detected_answers)


def test_injection_extraction(form_path):
    command = ["python", "./inject.py"] + [
        form_path,
        "test-images/a-3_groundtruth.txt",
        "injected.jpg",
    ]
    subprocess.run(command)

    command = ["python", "./extract.py"] + ["injected.jpg", "output.txt"]
    subprocess.run(command)

    with open("test-images/a-3_groundtruth.txt", "r") as f:
        answers = [line.strip() for line in f.readlines()]
    with open("output.txt", "r") as f:
        detected_answers = [line.strip() for line in f.readlines()]
    assert all(
        [answer == detected for answer, detected in zip(answers, detected_answers)]
    )


test_grading("test-images/a-3.jpg", "test-images/a-3_groundtruth.txt")
test_grading("test-images/a-27.jpg", "test-images/a-27_groundtruth.txt")

test_extracting("test-images/a-30.jpg")
test_extracting("test-images/a-48.jpg")
test_extracting("test-images/b-13.jpg")
test_extracting("test-images/b-27.jpg")
test_extracting("test-images/c-18.jpg")
test_extracting("test-images/c-33.jpg")

test_injection_extraction("test-images/blank_form.jpg")
