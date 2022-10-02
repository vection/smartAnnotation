# :wave: Smart Annotation Tool :wave:
### Easy and smart annotation tool based on jupyter notebook 

Fun tool to annotate your documents using jupter notebook, if you have low volume of data and you want easy, smart and free annotation tool you arrived to the right place. This tool based on jupyter-bbox-widget.
The motivation is to create smart tool in jupter notebook environment that supports OCR annotation.

The extra feature designed for Key-value assignments or cases of correlations between key and value on coordinates axis.
By enabling Key Finder checkbox the script will try to find best options for his key.


## How it works :vulcan_salute:
First we set BBoxWidget with correct parameters which is list of images (depending on your sitatuation) and all classes.

In annotation tool class there are few types of values - number,text; this can help to filter final value that will be registered in dataset.
By default all values declared as text.

After marking first VALUE class the tool tries to find the corresponding text value from ocr and record it in annotation file.

Extra feature available to detect also corresponding keys automatically using the vocabulary. 
When all bounds filled, hitting sumbit button will save the keys in vocabulary and for next image it will try to annotate automatically.

Vocabulary file and json annotation files will saved each sumbit event.

Note that rendering is quite unstable and may affect some unrelated results.


## Output format
### Annotation file
```
[
    {
        "x": 20,
        "y": 18,
        "label": "r_number_key",
        "value": "Customer Invoices",
        "width": 186,
        "height": 14
    },
    {
        "x": 199,
        "y": 13,
        "width": 155,
        "height": 29,
        "label": "r_number",
        "value": "/ INV/2018/0004"
    },
    {
        "x": 728,
        "y": 397,
        "width": 84,
        "height": 9,
        "label": "salesperson",
        "value": "Administrator"
    },
    {
        "x": 554,
        "y": 393,
        "label": "salesperson_key",
        "value": "Administrator",
        "width": 95,
        "height": 28
    },
    {
        "x": 204,
        "y": 259,
        "width": 206,
        "height": 9,
        "label": "client_name",
        "value": "Demo"
    },
    {
        "x": 37,
        "y": 252,
        "label": "client_name_key",
        "value": "Customer",
        "width": 68,
        "height": 16
    }
]
```


## How to take it from here :point_up_2:
- can prepare layoutlmv3 dataset from annotation file and make it easy to train proper detection model which will feed himself afterwards.
- use better OCR service to raise detection accuracy.


## Short video example
https://user-images.githubusercontent.com/28596354/193272615-f105262c-c5da-43a6-aba8-875fb9256b1b.mp4

## Original Libraries:

[jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget)

[boundbox](https://github.com/akash1729/boundbox)

