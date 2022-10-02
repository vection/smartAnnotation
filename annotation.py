from PIL import Image
from boundbox import BoundBox
import pytesseract
import re
from copy import deepcopy
import json
import PIL

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract'


# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

class AnnotationTool:
    def __init__(self, ocr_path=None, vocab=None, key_find=False):
        if ocr_path:
            pytesseract.pytesseract.tesseract_cmd = ocr_path
        if vocab:
            f = open(vocab)
            self.vocab = json.load(f)
        self.data_types = {'date': 'number', 'r_number': 'text', 'client_name': 'text', 'total_amount': 'number',
                           'salesperson': 'text'}
        self.y_threshold = 5
        self.x_threshold = 50
        self.x_search_threshold = 10
        self.y_search_threshold = 20
        self.key_find = key_find

    def ocr(self, image, merge=False):
        if isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        else:
            data = pytesseract.image_to_data(Image.open(image), output_type=pytesseract.Output.DICT)

        temp_bounds = BoundBox.pytesseract_boxes(data)
        cleaned_bounds = self.clean_ocr(temp_bounds)
        if merge:
            cleaned_bounds = BoundBox.merge_box(cleaned_bounds, dx=0.5, merge_box=False)
        return cleaned_bounds

    def search_by_vocab(self, bounds, name):
        founded = []
        for b in bounds:
            if any([word.lower() in b.text_value.lower() for word in self.vocab[name]]):
                founded.append(b)
        return founded

    def check_valid(self, text, return_text=False, value_type=None):
        if value_type == 'number':
            text = text.replace(".", "").replace(",", "").replace(":", "").replace("/", "")
            pattern = re.compile("[0-9]+")
            res = pattern.search(text)
            if return_text:
                if res:
                    return text
                else:
                    return False
            if res:
                if len(res[0]) < 1:
                    return False
        elif value_type == 'text':
            pattern = re.compile("[a-zA-Z]+")
            res = pattern.search(text)
            if return_text:
                if res:
                    return text
                else:
                    return False
            if res:
                if len(res[0]) < 1:
                    return False
            return True
        else:
            pattern = re.compile("[a-zA-Z0-9]+")
            res = pattern.search(text)
            if return_text:
                if res:
                    return res[0].lower()
                else:
                    return False
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

    def search_bound(self, bounds, x, y, value_type=None):
        bounds = BoundBox.merge_box(bounds, dx=0.5, merge_box=False)
        if value_type and value_type in self.data_types:
            value_type = self.data_types[value_type]
        for b in bounds:
            if (b.p1.x - self.x_search_threshold <= x <= b.p2.x + self.x_search_threshold) and (
                    b.p1.y - self.y_search_threshold <= y <= b.p4.y + self.y_search_threshold) and self.check_valid(
                    b.text_value, value_type=value_type):
                return b
        return None

    def find_alternative_key(self, bounds, value_bound):
        options = []
        bounds = BoundBox.merge_box(bounds, dx=0.8, merge_box=False)
        # check for left keys
        for bound in bounds:
            if not isinstance(bound, BoundBox) or not isinstance(value_bound, BoundBox):
                continue
            if bound.text_value == value_bound.text_value or bound.p4.y > value_bound.p4.y + self.y_threshold or bound.p1.y < value_bound.p1.y + self.y_threshold:
                continue

            if abs(bound.p4.y - value_bound.p1.y) > 80 or bound.p2.x > value_bound.p1.x + 10:
                continue

            if self.check_valid(bound.text_value) is False:
                continue

            options.append(bound)

        if len(options) == 0:
            for bound in bounds:
                if not isinstance(bound, BoundBox) or not isinstance(value_bound, BoundBox):
                    continue
                if bound.text_value == value_bound.text_value or bound.p4.y > value_bound.p4.y + self.y_threshold:
                    continue

                if abs(bound.p4.y - value_bound.p1.y) > 80:
                    continue

                if self.check_valid(bound.text_value) is False:
                    continue
                candidate_center = int((bound.p1.x + bound.p2.x) / 2)
                if abs(bound.p2.x - value_bound.p1.x) <= self.x_threshold or abs(
                        bound.p1.x - value_bound.p2.x) <= self.x_threshold or \
                        (value_bound.p1.x < candidate_center < value_bound.p2.x):
                    options.append(bound)

        #         filtered_options = []
        #         for opt in options:
        #             cleaned_text = self.check_valid(opt.text_value, return_text=True)
        #             if cleaned_text:
        #                 opt.text_value = self.check_valid(opt.text_value, return_text=True)
        #                 filtered_options.append(opt)
        print("Last ", options)
        if len(options) == 2:
            return BoundBox.merge_box(options, dx=100, merge_box=False)
        return options


def add_annotation_starter(anno,bounds):
    print("Starter!!!")
    for key in anno.vocab.keys():
        options_for_key = []
        for b in bounds:
            if any([w in b.text_value.lower() for w in anno.vocab[key]]):
                new_tagged_item = {}
                new_tagged_item['x'] = b.p1.x
                new_tagged_item['y'] = b.p1.y
                new_tagged_item['label'] = key
                new_tagged_item['value'] = b.text_value
                new_tagged_item['width'] = abs(b.p2.x - b.p1.x)
                new_tagged_item['height'] = abs(b.p4.y - b.p1.y)
                options_for_key.append(new_tagged_item)

    print("End of starter: ", options_for_key)
    return options_for_key


def add_annotation(anno,img_path, bboxes, starter=False):
    bbox_list = deepcopy(bboxes)

    def find_item(bbox_l, val):
        for ind, bb in enumerate(bbox_l):
            if bb['label'] == val:
                return bb, ind
        return None

    a_bounds = anno.ocr(img_path, merge=True)
    if starter:
        new_boxes = add_annotation_starter(a_bounds)
        for box in new_boxes:
            bbox_list.append(box)
    value_labels = [item['label'] for item in bbox_list if 'key' not in item['label']]
    keys_labels = [item['label'] for item in bbox_list if 'key' in item['label']]

    if anno.key_find:
        for v in value_labels:
            if not any([v.replace("_", "") in key.replace("_", "") for key in keys_labels]):
                item_bbox, index = find_item(bbox_list, v)
                item_center_x = item_bbox['x']  # + int(item_bbox['width'] / 2)
                item_center_y = item_bbox['y']  # + item_bbox['height']
                founded_bound = anno.search_bound(a_bounds, item_center_x, item_center_y, value_type=v)
                print("Founded ", founded_bound)
                if founded_bound is None:
                    item_center_x = item_bbox['x'] + int(item_bbox['width'] / 2)
                    item_center_y = item_bbox['y'] + item_bbox['height']
                    founded_bound = anno.search_bound(a_bounds, item_center_x, item_center_y, value_type=v)
                    print("2 Founded ", founded_bound)
                    if founded_bound is None:
                        item_bbox['status'] = 0
                        continue
                keys_options = anno.find_alternative_key(a_bounds, founded_bound)
                if len(keys_options) > 0 and keys_options[0] is not None:
                    new_tagged_item = {}
                    new_tagged_item['x'] = keys_options[0].p1.x
                    new_tagged_item['y'] = keys_options[0].p1.y
                    new_tagged_item['label'] = v + "_key"
                    new_tagged_item['value'] = keys_options[0].text_value
                    new_tagged_item['width'] = abs(keys_options[0].p2.x - keys_options[0].p1.x)
                    new_tagged_item['height'] = abs(keys_options[0].p4.y - keys_options[0].p1.y)

                    bbox_list.append(new_tagged_item)
                    item_bbox['value'] = founded_bound.text_value
                    item_bbox['x'] = founded_bound.p1.x
                    item_bbox['y'] = founded_bound.p1.y
                    item_bbox['width'] = abs(founded_bound.p2.x - founded_bound.p1.x)
                    item_bbox['height'] = abs(founded_bound.p4.y - founded_bound.p1.y)
                    bbox_list.pop(index)
                    bbox_list.append(item_bbox)

    # find ocr value
    for box in bbox_list:
        if 'value' not in box:
            item_center_x = box['x'] + int(box['width'] / 2)
            item_center_y = box['y'] + box['height']
            founded_bound = anno.search_bound(a_bounds, item_center_x, item_center_y, value_type='text')
            print("Key Founded ", founded_bound)
            if founded_bound:
                box['value'] = founded_bound.text_value.lower()

    return bbox_list

# path = r'E:\Aviv\ocr_project\1_Images\1_Images\20210428_151140.jpg'
# a_bounds = anno.ocr(path, merge=True)
# for i, b in enumerate(a_bounds):
#     print(i, b)

# keys_options = anno.find_alternative_key(a_bounds, a_bounds[0])
# print(keys_options)
