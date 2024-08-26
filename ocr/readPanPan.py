import argparse
import time
import os
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

class ImageModel:
    def __init__(self, path, data):
        self.ori_path = path
        self.boxs = [line[0] for line in data]
        self.txts = [line[1][0] for line in data]
        self.scories = [line[1][1] for line in data]

def parseImage(path):
    if not os.path.isfile(path):
        raise "输入的不是一个文件"
    print(f"图片地址：{path}")
    return ImageModel(path, PaddleOCR().ocr(path, cls=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR解析')
    parser.add_argument('-d', '--document', type=str, required=True, help="图片目录")
    parser.add_argument('-o', '--outDocument', type=str, required=True, help="输出目录")

    args = parser.parse_args()

    print('开始解析...')
    start_time = time.time()
    document = args.document
    out_document = args.outDocument

    if not os.path.isdir(document):
        raise "document必须是一个目录路径"
    else:
        print(f"文件目录: {document}")

    if not os.path.isdir(out_document):
        raise "out_document必须是一个目录路径"
    else:
        print(f"输出目录: {out_document}")

    paths = [os.path.join(document, path) for path in os.listdir(document) if path.endswith('JPG')]
    images = [parseImage(path) for path in paths]
    print(paths)

    end_time = time.time()
    print(f"解析结束，耗时：{(end_time - start_time): .2f}秒", )
    