from PIL.Image import Image

from widget import BBoxWidget
import ipywidgets as widgets
import os
import json
import base64
from annotation import add_annotation
from multiprocessing import Pool, Value

###########################################
# this code should run in cell of jupyter notebook!
###########################################

# encoding image function
def encode_image(filepath):
    im = Image.open(filepath)
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64," + encoded

# declaring of widget object
testwidget = BBoxWidget(
    image=encode_image('images/10.jpg'),
    classes=['date', 'r_number', 'client_name', 'total_amount', 'r_number_key', 'total_amount_key', 'salesperson',
             'salesperson_key', 'client_name_key'],
)

number_of_boxes = Value('d', 0)

w_out = widgets.Output()
start = False

# event to handle ocr and annotation changes
def on_bbox_change(change):
    global testwidget, number_of_boxes, starter, start
    try:
        w_out.clear_output(wait=True)
        with w_out:
            if number_of_boxes.value > 0 and len(testwidget.bboxes) < number_of_boxes.value - 1:
                return False
            if start:
                return False
            start = True
            bboxes_all = add_annotation(testwidget.bboxes, starter)
            if starter:
                starter = False
            bboxes_all = [{**bbox} for bbox in bboxes_all]
            render(bboxes_all)
            start = False
            # os.kill(os.getpid(),SIGABRT)

    except Exception as e:
        print("Exception! ", e)


# render function to reload bounding boxes
def render(bboxes):
    global testwidget, number_of_boxes

    # swap positions to trigger rendering seems like its the only way
    def swapPositions(aa, pos1, pos2):
        first_ele = aa.pop(pos1)
        second_ele = aa.pop(pos2 - 1)
        aa.insert(pos1, first_ele)
        aa.insert(pos2, first_ele)

        return aa

    print("Im rendering", number_of_boxes, len(bboxes))
    if len(bboxes) > 1:
        testwidget.bboxes = swapPositions(bboxes, 0, 1)
        number_of_boxes.value = len(testwidget.bboxes)
    # return bboxes


testwidget.observe(on_bbox_change, names=['bboxes'])

w_container = widgets.VBox([
    testwidget,
    w_out,
])


@testwidget.on_submit
def submit():
    global anno
    print("Submit start")
    # image_file = files[w_progress.value]
    for item in testwidget.bboxes:
        if 'key' not in item['label']:
            continue

        if item['label'] not in anno.vocab.keys():
            word_vocab[item['label']] = []
            anno.vocab[item['label']] = []

        if 'value' in item.keys() and item['value'].lower() in anno.vocab[item['label']]:
            continue
        if 'value' in item.keys() and 'status' in item.keys() and item['status'] != 0:
            anno.vocab[item['label']].append(item['value'].lower())
        else:
            anno.vocab[item['label']].append(item['value'].lower())

    # save annotations for current image
    with open(save_path, 'w') as f:
        json.dump(testwidget.bboxes, f, indent=4)

    if save_vocabs:
        with open(save_vocab_path, 'w') as f:
            json.dump(anno.vocab, f, indent=4)

    print("Saved!")
    # move on to the next file
    # skip()


# settings for vocabulary and annotation file
save_path = 'output.json'
save_vocab_path = 'vocab.json'
save_vocabs = True
word_vocab = {}
starter = True

