from PIL import Image, ImageDraw
import sys
import numpy as np

from tqdm import tqdm
import math


def load_image(path):
    image = Image.open(path)
    return image


def draw_rectangle(img, x0, y0, x1, y1, color="red"):
    # width, height = 335, 40
    new_img = ImageDraw.Draw(img)
    new_img.rectangle([(x0, y0), (x1, y1)], outline=color)


def find_bounding_boxes(binary):
    y_start = int(binary.size[1] * 0.3)
    # bbox_width, bbox_height = 335, 40
    bbox_width, bbox_height = int(math.ceil(binary.size[0] * 0.197)), int(
        math.ceil(binary.size[1] * 0.0181)
    )
    y_jumper = bbox_height
    x_shift = int(math.ceil(binary.size[0] * 0.047))
    box_space = int(math.ceil(binary.size[0] * 0.00588))
    x_limmiter = 0
    box_count = 0
    pixel_recorder = list()
    x_start = binary.size[0] - 1
    y = y_start
    for part in range(3):
        x_limmiter = 0
        y = y_start
        while y < binary.size[1]:
            for x in range(x_start, x_limmiter, -1):
                if binary.getpixel((x, y)):
                    if (
                        np.array(
                            [
                                binary.getpixel((x - i, y)) != 0
                                for i in range(bbox_width)
                            ]
                        ).sum()
                        > bbox_width * 0.3
                    ):
                        modifier = int(math.ceil(binary.size[1] * 0.002272))  # 5
                        pixel_recorder.append(
                            (
                                x - bbox_width + modifier,
                                y - modifier,
                                x,
                                y + bbox_height,
                            )
                        )
                        y += y_jumper
                        x_limmiter = x - bbox_width - box_space
                        box_count += 1
                        break
            y += 1
        x_start = pixel_recorder[-1][0] - x_shift

    return pixel_recorder


def order_answers(student_answers):
    question_no = 59
    student_answers_mapped = dict()
    for i, answer in enumerate(student_answers):
        if i == 27:
            question_no = 30
        elif i == 56:
            question_no = 1
        student_answers_mapped[str(question_no)] = answer
        question_no += 1

    student_answers_ordered = list()
    for i in range(85):
        student_answers_ordered.append(student_answers_mapped[str(i + 1)])

    return student_answers_ordered


def check_if_written_answer(binary, pixel_record, show=False):
    x_shift = int(math.ceil(binary.size[0] * 0.047))
    cropped = binary.crop(
        (
            pixel_record[0] - x_shift,
            pixel_record[1],
            pixel_record[0],
            pixel_record[1] + int(math.ceil(binary.size[1] * 0.0159)),  # 35
        )
    )
    if show:
        cropped.show()
    return np.array(
        [
            cropped.getpixel((i, j)) != 0
            for i in range(cropped.size[0])
            for j in range(cropped.size[1])
        ]
    ).sum()


def find_answers_from_boxes(binary, pixel_recorder):

    q_count = 0
    identity_map = {0: "#", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    student_answers = list()
    for pixel_record in pixel_recorder:
        cropped = binary.crop(
            (
                pixel_record[0],
                pixel_record[1],
                pixel_record[2],
                pixel_record[3],
            )
        )

        # print(
        #     check_if_written_answer(
        #         binary, pixel_record, show=True if q_count == 11 else False
        #     )
        # )
        # q_count += 1
        parts = 6
        x0_portion = 0
        x1_portion = cropped.size[0] // 6
        student_answer = list()
        for i in range(parts):
            section = cropped.crop((x0_portion, 0, x1_portion, cropped.size[1]))
            if (
                np.array(
                    [
                        section.getpixel((i, j)) != 0
                        for i in range(section.size[0])
                        for j in range(section.size[1])
                    ]
                ).sum()
                > 500
            ):
                student_answer.append(identity_map[i]) if i != 0 else None

            x0_portion = x1_portion
            x1_portion = x1_portion + (cropped.size[0] // 6)

        if check_if_written_answer(binary, pixel_record) > 100:
            student_answer.append(" x")
        student_answers.append("".join(ans for ans in student_answer))

    return order_answers(student_answers)


def draw_bounding_box(img, height=40, width=335, offset_x=0, offset_y=0):

    pixel_recorder = find_bounding_boxes(img)

    binary = img.convert("L").point(lambda x: 255 if x < 128 else 0, "1")

    for box in tqdm(pixel_recorder):
        # print(box)
        draw_rectangle(img, box[0], box[1], box[2], box[3])
        bounded_region = binary.crop(box)
        # if np.array(
        #     [
        #         bounded_region.getpixel((i, j)) != 0
        #         for i in range(bounded_region.size[0])
        #         for j in range(bounded_region.size[1])
        #     ]
        # ).sum() > ((bounded_region.size[0] * bounded_region.size[1]) / 7):
        #     draw_rectangle(img, box[0], box[1], box[2], box[3])
    # img.show()
    student_answers = find_answers_from_boxes(binary, pixel_recorder)

    return img, find_answers_from_boxes(binary, pixel_recorder)


def write_answers_to_file(answers, output_filepath):
    with open(output_filepath, "w") as f:
        for i, answer in enumerate(answers):
            f.write(f"""{i+1} {answer}""" + "\n")


def get_student_answers(img):
    img = img.convert("RGB")
    # img.show()
    threshold = 128
    gray = img.convert("L")
    binary = gray.point(lambda x: 255 if x < threshold else 0, "1")
    student_answers = find_answers_from_boxes(binary, find_bounding_boxes(binary))
    return student_answers


def create_scored_image(img, answers):
    img = img.convert("RGB")

    threshold = 128
    gray = img.convert("L")
    binary = gray.point(lambda x: 255 if x < threshold else 0, "1")

    pixel_recorder = find_bounding_boxes(binary)

    for pixel_record, answer in zip(pixel_recorder, answers):
        cropped = binary.crop(
            (
                pixel_record[0],
                pixel_record[1],
                pixel_record[2],
                pixel_record[3],
            )
        )

        parts = 6
        x0_portion = 0
        x1_portion = cropped.size[0] // 6
        for i in range(parts):
            section = cropped.crop((x0_portion, 0, x1_portion, cropped.size[1]))

            if (
                np.array(
                    [
                        section.getpixel((i, j)) != 0
                        for i in range(section.size[0])
                        for j in range(section.size[1])
                    ]
                ).sum()
                > 500
            ):
                option_labels = ["#", "A", "B", "C", "D", "E"]

                option = option_labels[i]

                first_white_x = np.argmax(np.any(section == 255, axis=0))
                first_white_y = np.argmax(np.any(section == 255, axis=None))
                draw_rectangle(
                    img,
                    pixel_record[0] + x0_portion + first_white_x + 10,
                    pixel_record[1] + first_white_y + 10,
                    pixel_record[0] + x0_portion + first_white_x + 50,
                    pixel_record[1] + first_white_y + 40,
                    color="green",
                )
            x0_portion = x1_portion
            x1_portion = x1_portion + (cropped.size[0] // 6)
    return img


if __name__ == "__main__":

    print("Recognizing form.jpg...")
    img = load_image(sys.argv[1])
    output_filepath = sys.argv[2]
    student_answers = get_student_answers(img)
    scored_image = create_scored_image(img, student_answers)
    scored_image.save("scored.jpg")
    write_answers_to_file(get_student_answers(img), output_filepath)
