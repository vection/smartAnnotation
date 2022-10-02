from PIL.Image import Image
from annotation import add_annotation, AnnotationTool
from widget import BBoxWidget
import ipywidgets as widgets
import os
import json
import base64
import time
from multiprocessing import Value
from pdf2image import convert_from_bytes

###########################################
# this code should run in cell of jupyter notebook!
###########################################

# required convertion functions
def encode_image(filepath):
    global current_image
    im = Image.open(filepath)
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    current_image = im
    return "data:image/jpg;base64," + encoded

def encode_pdf(file_path):
    global current_image
    with open(file_path, 'rb') as f:
        image_bytes = f.read()
    imgs = convert_from_bytes(image_bytes, dpi=120,
                              poppler_path="D://Downloads//poppler-0.68.0_x86//poppler-0.68.0\\bin")
    imgs[0].save('temp.jpg')
    return encode_image('temp.jpg')


# set folder of pdf files
files_path = "E://Aviv//ocr_project//archive (2)//InvoiceData//InvoiceDataPDF//"
current_image = None
# set testwidget initial settings
testwidget = BBoxWidget(
    image=encode_pdf(files_path + "97.pdf"),
    classes=['date', 'r_number', 'client_name', 'total_amount', 'r_number_key', 'total_amount_key', 'salesperson',
             'salesperson_key', 'client_name_key'],
)
files_progress = widgets.IntProgress(value=0, max=len(files_path), description='Progress')

key_finder_checkbox = widgets.Checkbox(value=True, description='Key finder')

number_of_boxes = Value('d', 0)
files = os.listdir(files_path)
w_out = widgets.Output()
start = False




# event to handle ocr and annotation changes
def on_bbox_change(change):
    global testwidget, number_of_boxes, starter, start, current_image
    try:
        w_out.clear_output(wait=True)
        with w_out:
            if number_of_boxes.value > 0 and len(testwidget.bboxes) < number_of_boxes.value - 1:
                return False
            if start:
                return False
            start = True
            bboxes_all = add_annotation(current_image, testwidget.bboxes, starter)
            if starter:
                starter = False
            bboxes_all = [{**bbox} for bbox in bboxes_all]
            render(bboxes_all)
            time.sleep(2)
            start = False

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

    if len(bboxes) > 1:
        testwidget.bboxes = swapPositions(bboxes, 0, 1)
        number_of_boxes.value = len(testwidget.bboxes)
    # return bboxes


def change_key_finder(change):
    global key_finder_checkbox, anno
    if change['new']:
        anno.key_find = True
    else:
        anno.key_find = False


testwidget.observe(on_bbox_change, names=['bboxes'])

key_finder_checkbox.observe(change_key_finder, names=['value'])

w_container = widgets.VBox([
    key_finder_checkbox,
    files_progress,
    testwidget,
    w_out,
])


@testwidget.on_skip
def skip():
    global starter
    files_progress.value += 1
    # open new image in the widget
    image_file = files[files_progress.value]
    if 'pdf' in image_file:
        testwidget.image = encode_pdf(os.path.join(files_path, image_file))
    else:
        testwidget.image = encode_image(os.path.join(files_path, image_file))
    testwidget.bboxes = []
    starter = False


@testwidget.on_submit
def submit():
    global anno
    print("Submit start")
    image_file = files[files_progress.value]
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
        json.dump({image_file: testwidget.bboxes}, f, indent=4)

    if save_vocabs:
        with open(save_vocab_path, 'w') as f:
            json.dump(anno.vocab, f, indent=4)

    print("Saved!")
    # move on to the next file
    skip()


# settings for vocabulary and annotation file
save_path = 'output.json'
save_vocab_path = 'vocab.json'
save_vocabs = True
word_vocab = {}
# enable/disable automatically finding keys from vocabulary after first mark
starter = True
anno = AnnotationTool(vocab='vocab.json')