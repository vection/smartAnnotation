from PIL import Image
from boundbox import BoundBox
import pytesseract
import re
from copy import deepcopy
# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract'


# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

class AnnotationTool:
    def __init__(self, ocr_path=None):
        if ocr_path:
            pytesseract.pytesseract.tesseract_cmd = ocr_path
        self.y_threshold = 5
        self.x_threshold = 30

    def ocr(self, image, merge=False):
        data = pytesseract.image_to_data(Image.open(path), output_type=pytesseract.Output.DICT)

        bounds = BoundBox.pytesseract_boxes(data)

        cleaned_bounds = self.clean_ocr(bounds)
        if merge:
            cleaned_bounds = BoundBox.merge_box(cleaned_bounds, dx=0, vx=0)
        return cleaned_bounds

    def check_valid(self, text,return_text=False):
        pattern = re.compile("[a-zA-Z0-9]+")
        res = pattern.search(text)
        if return_text:
            return res[0].lower()
        if res:
            if len(res[0]) < 1:
                return False
        return True

    def clean_ocr(self, bounds):
        new_bounds = []
        for b in bounds:
            if len(b.text_value.strip()) < 1:
                continue
            new_bounds.append(b)
        return new_bounds

    def find_alternative_key(self, bounds, value_bound):
        options = []
        copy_bounds = deepcopy(bounds)
        # check for left keys
        for bound in bounds:
            if bound.text_value == value_bound.text_value or bound.p4.y > value_bound.p4.y + self.y_threshold or bound.p1.y < value_bound.p1.y + self.y_threshold:
                continue

            if self.check_valid(bound.text_value) is False:
                continue

            options.append(bound)
        if len(options) == 0:
            for bound in copy_bounds:
                if bound.text_value == value_bound.text_value or bound.p4.y > value_bound.p4.y + self.y_threshold:
                    continue

                if self.check_valid(bound.text_value) is False:
                    continue
                candidate_center = int((bound.p1.x + bound.p2.x) / 2)
                if abs(bound.p2.x - value_bound.p1.x) <= self.x_threshold or abs(
                        bound.p1.x - value_bound.p2.x) <= self.x_threshold or \
                        (value_bound.p1.x < candidate_center < value_bound.p2.x):
                    options.append(bound)

        for opt in options:
            opt.text_value = self.check_valid(opt.text_value,return_text=True)
        return options


path = r'E:\Aviv\ocr_project\1_Images\1_Images\20210428_151140.jpg'
anno = AnnotationTool()
a_bounds = anno.ocr(path)
for i, b in enumerate(a_bounds):
    print(i, b)

keys_options = anno.find_alternative_key(a_bounds, a_bounds[50])
print(keys_options)
