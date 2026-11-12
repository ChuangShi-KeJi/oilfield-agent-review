import re

_PAGE_RE = re.compile(r"=====\s*第\s*(\d+)\s*页\s*=====")
_SENT_RE = re.compile(r"[。；;！!？?\n]")
_NORM_RE = re.compile(r"[\s，,。；;：:、\"'“”‘’（）()\[\]【】《》\-—_/|·]+")
_COORD_RE = re.compile(r"[°′″]|经纬|坐标|N\s*\d+°|E\s*\d+°")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
