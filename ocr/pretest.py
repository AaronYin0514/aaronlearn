import argparse
import time
import os
import re
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

BLACK = ["中国移动", "Q我吧", "手机在线", "-WiFi"]
EBLACK = ["发送", "三", "网"]

class ImageModel:
    def __init__(self, path, data):
        self.ori_path = path
        self.file_name = os.path.split(path)[-1]
        self.name = os.path.splitext(self.file_name)[0]
        # self.boxs = [line[0] for line in data]
        self.txts = [line[1][0] for lines in data for line in lines if self.is_available(line[1])]
        # self.scories = [line[1][1] for line in data]

    def __len__(self):
        return len(self.txts)
    
    def __getitem__(self, index):
        return self.txts[index]
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < len(self):
            item = self[self.idx]
            self.idx += 1
            return item
        else:
            raise StopIteration
    
    def is_available(self, data):
        if data[1] <= 0.5:
            return False
        for b in BLACK:
            if data[0].strip().find(b) != -1:
                return False
        for b in EBLACK:
            if data[0].strip() == b:
                return False
        pattern = "^[`~!@#$%^&*()_\\−+=<>?:\"\\{\\}|,./;'\\[\\]·~！@#￥%……&*（）——《》？：“”【】、；‘’'，。、×0-9\\sa-zA-Z]+$"
        if re.search(pattern, data[0]):
            return False
        return True

_ocr = PaddleOCR()

def parseImage(path):
    if not os.path.isfile(path):
        raise "输入的不是一个文件"
    print(f"图片地址：{path}")
    return ImageModel(path, _ocr.ocr(path, cls=False))

def write(path, images):
    for model in images:
        if len(model):
            des_path = os.path.join(path, f"{model.name}.txt")
            with open(des_path, 'w') as file:
                for text in model:
                    file.write(f"{text}\n")         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR解析')
    parser.add_argument('-d', '--document', type=str, required=True, help="图片目录")

    args = parser.parse_args()

    print('开始解析...')
    start_time = time.time()
    document = args.document

    if not os.path.isdir(document):
        raise "document必须是一个目录路径"
    else:
        print(f"文件目录: {document}")

    # a = os.listdir(document)
    # a.sort()
    # print(a)

    paths = [os.path.join(document, path) for path in os.listdir(document) if path.endswith('JPG')]
    images = [parseImage(path) for path in paths]

    write(document, images)

    end_time = time.time()
    print(f"解析结束，耗时：{(end_time - start_time): .2f}秒", )
    