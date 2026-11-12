# -*- coding: utf-8 -*-
# 通用 PDF 解析：pymupdf 读字(+OCR回退) + pdfplumber 抽表格 + 清洗去重
# 用法（命令行）:
#   python core.py          # 处理当前文件夹全部 PDF
#   python core.py 文件.pdf # 只处理这个文件
#   python core.py 某文件夹  # 处理该文件夹全部 PDF
# 结果自动存到 <输入位置>\output_txt\ 下，每个 PDF 一个 txt
#   （可用环境变量 PDF_TXT_OUT_DIR 指定别的输出目录）
# 用法（被网页调用）:
#   from core import extract_text
#   text = extract_text(路径 / 文件流 / bytes)   # 三种都行，返回清洗后的整篇文字
import sys, re, os, glob, warnings, threading
warnings.filterwarnings("ignore", message=r"The `fitz` API is deprecated.*")
import fitz            # pymupdf：直接读文字层 + 渲染图片
import pdfplumber      # 抽表格（按线还原单元格）
dpi = 200

# OCR 模型只加载一次并全局复用：多文件上传时不再每个扫描件重载几百 MB 模型。
# easyocr 的 Reader 不是线程安全的，所以 readtext 调用用锁串行化。
_READER = None
_READER_LOCK = threading.Lock()
_OCR_LOCK = threading.Lock()


def _get_reader():
    """惰性加载 easyocr（没装就直接抛 ModuleNotFoundError，由调用方兜底）。"""
    global _READER
    with _READER_LOCK:
        if _READER is None:
            import easyocr
            _READER = easyocr.Reader(["ch_sim", "en"], gpu=False)
        return _READER

def find_inputs(path):
    """输入可以是单个文件或文件夹，返回待处理的 pdf 列表。"""
    if os.path.isfile(path):
        return [path]
    # Windows 大小写不敏感，一个 glob 取全部文件再按扩展名过滤，避免重复
    return sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(".pdf"))

def looks_ok(s):
    """粗略判断这段文字是否真读出来了：非空、乱码少、有一定真实字符。"""
    s = s.strip()
    if not s:
        return False
    if s.count("\ufffd") > max(3, len(s) * 0.05):
        return False
    real = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", s))
    return real / len(s) > 0.3

def clean_text(s):
    """去掉乱码/控制字符/零宽符，合并空行，去掉完全重复的相邻行。"""
    s = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\u200b\u200c\u200d\u200e\u200f\ufeff]", "", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if out and ln == out[-1]:   # 相邻完全重复视为 OCR 残留，去掉
            continue
        out.append(ln)
    return "\n".join(out)

def extract_one(path, reader, progress=None):
    """处理单个 PDF，返回清洗后的整篇文字（含每页标记和表格）。"""
    with fitz.open(path) as doc:
        pages = [page.get_text() or "" for page in doc]
        need_ocr = [i for i, t in enumerate(pages) if not looks_ok(t)]
        if need_ocr:
            import numpy as np
            if reader is None:
                reader = _get_reader()       # 全局复用，不再每个文件重载模型
        for i in need_ocr:                       # 读不出的页单独 OCR（内存直接喂，不落盘）
            best = ""
            for scale in (dpi, dpi + 100):
                pix = doc[i].get_pixmap(dpi=scale, colorspace=fitz.csRGB)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                with _OCR_LOCK:              # easyocr Reader 非线程安全，串行化
                    best = "\n".join(reader.readtext(img, detail=0, paragraph=True))
                if looks_ok(best):
                    break
            pages[i] = best

    tables = []
    with pdfplumber.open(path) as pdf:           # 抽表格（对扫描/OCR页无效）
        for page in pdf.pages:
            tables.append(page.extract_tables() or [])

    out = []
    for i, t in enumerate(pages):
        if i in need_ocr:
            out.append("【本页无文字层（图片/截图/空白页），已用 OCR 识别，内容可能不准确】")
        out.append(f"===== 第 {i + 1} 页 =====")
        out.append(t)
        page_tables = tables[i] if i < len(tables) else []   # 两个库页数不一致时兜底
        for tb in page_tables:
            out.append("--- 表格 ---")
            for row in tb:
                out.append(" | ".join("" if c is None else str(c) for c in row))
    return clean_text("\n".join(out)), reader

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    inputs = find_inputs(path)
    if not inputs:
        print("没找到 PDF 文件，退出。"); return
    base = os.path.dirname(path) if os.path.isfile(path) else path
    # 默认写在输入目录旁；需要换位置时用环境变量覆盖（不写死在某个盘）
    outdir = os.environ.get("PDF_TXT_OUT_DIR") or os.path.join(base, "output_txt")
    os.makedirs(outdir, exist_ok=True)
    reader = None
    for f in inputs:
        text, reader = extract_one(f, reader)
        stem = os.path.splitext(os.path.basename(f))[0]
        dest = os.path.join(outdir, f"{stem}_文字.txt")
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"PASS {os.path.basename(f)} -> {dest}")

if __name__ == "__main__":
    main()
# ========== 给网页版写的适配器 ==========
def extract_text(source, reader=None):
    """专门供外部（Streamlit / 脚本）调用的统一入口。

    source 允许三种形态，统一委托给内核 extract_one：
      - 文件路径（str / os.PathLike）
      - 上传的文件流（任何带 .read() 的对象，如 BytesIO / st.UploadedFile）
      - 原始字节（bytes / bytearray）
    解析失败一律抛异常（不吞），由调用方决定怎么兜底和提示。
    """
    import tempfile
    path, tmp_path = None, None
    try:
        if isinstance(source, (bytes, bytearray)):
            payload = bytes(source)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(payload)
            tmp_path = tmp.name
            path = tmp_path
        elif hasattr(source, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(source.read())
            tmp_path = tmp.name
            path = tmp_path
        else:
            path = os.fspath(source)        # str / PathLike：直接用，不再复制一份
        text, _ = extract_one(path, reader)
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass