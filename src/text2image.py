# -*- coding: utf-8 -*-
import os, re, io, argparse, math, json, cv2
from PIL import Image, ImageFont, ImageDraw
import pygame
from skimage import io as image_io
import numpy as np

g_script_folder = os.path.dirname(os.path.abspath(__file__))
g_fonts_folder = os.path.join(g_script_folder, "../resource/")
input_dir = '../input/'
output_dir = '../misc/'

g_re_first_word = re.compile((u""
    + u"(%(prefix)s+\S%(postfix)s+)" # 标点
    + u"|(%(prefix)s*\w+%(postfix)s*)" # 单词
    + u"|(%(prefix)s+\S)|(\S%(postfix)s+)" # 标点
    + u"|(\d+%%)" # 百分数
    ) % {
    "prefix": u"['\"\(<\[\{‘“（《「『]",
    "postfix": u"[:'\"\)>\]\}：’”）》」』,;\.\?!，、；。？！]",
})

pygame.init()

def getFontForPyGame(font_name="wqy-zenhei.ttc", font_size=14):
    return pygame.font.Font(os.path.join(g_fonts_folder, font_name), font_size)

def makeConfig(cfg=None):
    if not cfg or type(cfg) != dict:
        cfg = {}
    default_cfg = {
        "width": 1600, # px
        "padding": (15, 18, 20, 18),
        "line-height": 20, #px
        "title-line-height": 32, #px
        "font-size": 14, # px
        "title-font-size": 24, # px
        "font-family": "wqy-zenhei.ttc",
        "font-color": (0, 0, 0),
        "font-antialiasing": True, # 字体是否反锯齿
        "background-color": (255, 255, 255),
        "first-line-as-title": True,
        "break-word": False,
    }

    default_cfg.update(cfg)

    return default_cfg

def makeLineToWordsList(line, break_word=False):
    """将一行文本转为单词列表"""
    if break_word:
        return [c for c in line]

    lst = []
    while line:
        ro = g_re_first_word.match(line)
        end = 1 if not ro else ro.end()
        lst.append(line[:end])
        line = line[end:]

    return lst


def makeLongLineToLines(long_line, start_x, start_y, width, line_height, font, cn_char_width=0):
    u"""将一个长行分成多个可显示的短行"""
    txt = long_line
    if not txt:
        return [None]

    words = makeLineToWordsList(txt, True)
    lines = []

    if not cn_char_width:
        cn_char_width, h = font.size("汉")
    avg_char_per_line =  math.ceil(width / cn_char_width)

    line_x = start_x
    line_y = start_y

    while words:

        tmp_words = words[:avg_char_per_line]
        tmp_ln = "".join(tmp_words)
        w, h = font.size(tmp_ln)
        wc = len(tmp_words)
        while w < width and wc < len(words):
            wc += 1
            tmp_words = words[:wc]
            tmp_ln = "".join(tmp_words)
            w, h = font.size(tmp_ln)
        while w > width and len(tmp_words) > 1:
            tmp_words = tmp_words[:-1]
            tmp_ln = "".join(tmp_words)
            w, h = font.size(tmp_ln)
            
        if w > width and len(tmp_words) == 1:
            # 处理一个长单词或长数字
            line_y = makeLongWordToLines(
                tmp_words[0], line_x, line_y, width, line_height, font, lines
            )
            words = words[len(tmp_words):]
            continue

        line = {
            "x": line_x,
            "y": line_y,
            "text": tmp_ln,
            "font": font,
        }

        line_y += line_height
        words = words[len(tmp_words):]

        lines.append(line)

        if len(lines) >= 1:
            # 去掉长行的第二行开始的行首的空白字符
            while len(words) > 0 and not words[0].strip():
                words = words[1:]

    return lines

def makeLongWordToLines(long_word, line_x, line_y, width, line_height, font, lines):
    if not long_word:
        return line_y

    c = long_word[0]
    char_width, char_height = font.size(c)
    default_char_num_per_line = width / char_width

    while long_word:

        tmp_ln = long_word[:default_char_num_per_line]
        w, h = font.size(tmp_ln)
        
        l = len(tmp_ln)
        while w < width and l < len(long_word):
            l += 1
            tmp_ln = long_word[:l]
            w, h = font.size(tmp_ln)
        while w > width and len(tmp_ln) > 1:
            tmp_ln = tmp_ln[:-1]
            w, h = font.size(tmp_ln)

        l = len(tmp_ln)
        long_word = long_word[l:]

        line = {
            "x": line_x,
            "y": line_y,
            "text": tmp_ln,
            "font": font,
            }

        line_y += line_height
        lines.append(line)
        
    return line_y

def makeMatrix(txt, font, title_font, cfg):
    width = cfg["width"]
    data = {
        "width": width,
        "height": 0,
        "lines": [],
    }

    a = txt.split("\n")
    cur_x = cfg["padding"][3]
    cur_y = cfg["padding"][0]
    cn_char_width, h = font.size(u"汉")

    for ln_idx, ln in enumerate(a):
        ln = ln.rstrip()
        if ln_idx == 0 and cfg["first-line-as-title"]:
            f = title_font
            line_height = cfg["title-line-height"]
        else:
            f = font
            line_height = cfg["line-height"]
        current_width = width - cur_x - cfg["padding"][1]
        lines = makeLongLineToLines(ln, cur_x, cur_y, current_width, line_height, f, cn_char_width=cn_char_width)
        cur_y += line_height * len(lines)

        data["lines"].extend(lines)

    data["height"] = cur_y + cfg["padding"][2]

    return data


def makeImage(data, cfg):
    width, height = data["width"], data["height"]
    im = Image.new("RGB", (width, height), cfg["background-color"])
    dr = ImageDraw.Draw(im)

    for ln_idx, line in enumerate(data["lines"]):
        makeLine(im, line, cfg)

    return im

def makeLine(im, line, cfg):
    if not line:
        return
    bio = io.BytesIO()
    x, y = line["x"], line["y"]
    text = line["text"]
    font = line["font"]
    rtext = font.render(text, cfg["font-antialiasing"], cfg["font-color"], cfg["background-color"])
    pygame.image.save(rtext, bio)

    bio.seek(0)
    ln_im = Image.open(bio)

    im.paste(ln_im, (x, y))

def txt2im(txt, out_file, cfg=None, neighbor=5):
    txt = txt.replace(",", "，").replace(";", "；").replace(":", "：").replace("?", "？")
    cfg = makeConfig(cfg)
    font = getFontForPyGame(cfg["font-family"], cfg["font-size"])
    title_font = getFontForPyGame(cfg["font-family"], cfg["title-font-size"])
    data = makeMatrix(txt, font, title_font, cfg)
    if "lines" not in data:
        return
    im = makeImage(data, cfg)
    im.save(input_dir + "image_set/" + out_file)

    meta = {
        "file_name": out_file,
        "font-size": cfg["font-size"],
        "font-family": cfg["font-family"],
        "line-height": cfg["line-height"],
        "width": cfg["width"],
        "height": data["height"],
        "x": data["lines"][0]["x"],
        "y": data["lines"][0]["y"],
        "text": data["lines"][0]["text"],
        "neighbor": neighbor
    }
    meta = img2mosaic(meta)
    with open(input_dir + "meta_data.jsonl", "a") as file_out:
        json.dump(meta, file_out, ensure_ascii=False)
        file_out.write("\n")

def mosaic_arr(frame, mosaic_area, neighbor=5):
    fh, fw = frame.shape[0], frame.shape[1]
    x, y, w, h = mosaic_area
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y + int(neighbor / 2)][j + x+ int(neighbor / 2)].tolist()  
            #color = np.mean(frame[i + y:i + y+neighbor][j + x:j + x+neighbor]).tolist()  
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1) 
            cv2.rectangle(frame, left_up, right_down, color, -1)


def img2mosaic(meta):
    file_path = input_dir + 'image_set/' + meta['file_name']
    img = image_io.imread(file_path)
    begin_white = meta['y']
    vertical_start = meta['y']
    while vertical_start < meta['height'] and img[vertical_start].min() == 255:
        vertical_start += 1
    vertical_end = vertical_start
    while vertical_end < meta['height'] and img[vertical_end].min() < 255:
        vertical_end += 1
    end_white = vertical_end
    while end_white < meta['height'] and img[end_white].min() == 255:
        end_white += 1
    
    horizontal_end = meta['x']
    while horizontal_end < meta['width']:
        if img[vertical_start - 3: vertical_end + 3, max(horizontal_end -  meta['line-height'] * 2, meta['x']): horizontal_end + meta['line-height'] * 2, :].min() < 255:
            horizontal_end = horizontal_end + meta['line-height']
        else:
            break

    mosaic_area_X = (meta['x'], horizontal_end)
    mosaic_area_Y = (vertical_start - 3,vertical_end + 3)
    mosaic_area = (mosaic_area_X[0], vertical_start - 3, horizontal_end - meta['x'], vertical_end - vertical_start + 6)
    meta['mosaic_area'] = mosaic_area

    mosaic_arr(img, mosaic_area, neighbor=meta["neighbor"])
    #meta['mosaic_feature'] = 
    image_io.imsave(file_path + '.mosaic_%d.png' % meta["neighbor"], img)

    with open(input_dir + "dataset.jsonl", "a") as file_out:
        json.dump({
            "file_name": file_path + '.mosaic_%d.png' % meta["neighbor"],
            "txt": meta["text"],
            "mosaic": img[vertical_start - 3: vertical_end + 3, meta['x']: horizontal_end, 0].tolist()
        }, file_out, ensure_ascii=False)
        file_out.write("\n")

    return meta


def show_img(txt='很快，好吃，味道足，量大', out_file="demo.png", neighbor=4):
    txt = txt.replace(",", "，").replace(";", "；").replace(":", "：").replace("?", "？")
    cfg = makeConfig(None)
    font = getFontForPyGame(cfg["font-family"], cfg["font-size"])
    title_font = getFontForPyGame(cfg["font-family"], cfg["title-font-size"])
    data = makeMatrix(txt, font, title_font, cfg)
    im = makeImage(data, cfg)
    im.save(output_dir + out_file)
    meta = {
        "file_name": out_file,
        "font-size": cfg["font-size"],
        "font-family": cfg["font-family"],
        "line-height": cfg["line-height"],
        "width": cfg["width"],
        "height": data["height"],
        "x": data["lines"][0]["x"],
        "y": data["lines"][0]["y"],
        "text": data["lines"][0]["text"],
        "neighbor": neighbor
    }

    file_path = output_dir + out_file
    img = image_io.imread(file_path)
    begin_white = meta['y']
    vertical_start = meta['y']
    while vertical_start < meta['height'] and img[vertical_start].min() == 255:
        vertical_start += 1
    vertical_end = vertical_start
    while vertical_end < meta['height'] and img[vertical_end].min() < 255:
        vertical_end += 1
    end_white = vertical_end
    while end_white < meta['height'] and img[end_white].min() == 255:
        end_white += 1
    
    horizontal_end = meta['x']
    while horizontal_end < meta['width']:
        if img[vertical_start - 3: vertical_end + 3, max(horizontal_end -  meta['line-height'] * 2, meta['x']): horizontal_end + meta['line-height'] * 2, :].min() < 255:
            horizontal_end = horizontal_end + meta['line-height']
        else:
            break

    mosaic_area_X = (meta['x'], horizontal_end)
    mosaic_area_Y = (vertical_start - 3,vertical_end + 3)
    mosaic_area = (mosaic_area_X[0], vertical_start - 3, horizontal_end - meta['x'], vertical_end - vertical_start + 6)
    meta['mosaic_area'] = mosaic_area

    mosaic_arr(img, mosaic_area, neighbor=meta["neighbor"])
    image_io.imsave(file_path + '.mosaic_%d.png' % meta["neighbor"], img)


def test():
    text = "打工人即打工仔，现在很多上班族的自称，他们起早贪黑，拿着微薄的工资，做着辛苦的工作，在屈辱里努力表现出倔强，在平凡中透露着追求。"
    show_img(text, "test.png", neighbor=3)
    show_img(text, "test.png", neighbor=4)
    show_img(text, "test.png", neighbor=5)
    show_img(text, "test.png", neighbor=6)
    show_img(text, "test.png", neighbor=7)
    show_img("蚂蚁准备上市前，杭州一栋大楼的员工都沸腾了。好多员工要变成千万富翁，基本无心工作。", "demo.png", neighbor=5)

if __name__ == "__main__":
    test()

  