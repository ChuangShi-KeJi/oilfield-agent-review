# -*- coding: utf-8 -*-
"""
app.py — 油气田钻井/压裂方案智能生成系统（完整版）
====================================================
功能总览：
1. asyncio + ThreadPoolExecutor 并发解析多个 PDF（不卡 UI，实时反馈）。
2. 解析文本经 RecursiveCharacterTextSplitter 切块后，用 langchain_chroma + HuggingFaceEmbeddings
   写入本地向量库（./chroma_db），并用 @st.cache_resource 缓存。
3. LangGraph 主流程：
       START -> geology_node(提取参数) -> safety_node(安全审查)
       safety_node --条件边--> 不通过且 retry_count < 2  -> geology_node（重试）
                  --条件边--> 不通过且 retry_count >= 2 -> emergency_node（熔断报警）
                  --条件边--> 通过                     -> generate_node（生成最终方案）
4. SqliteSaver 持久化：thread_id 会同步写进 URL(?tid=...)，刷新页面自动读回，
   State 进度不丢（可恢复 / 可继续 HITL 审核）。
5. Streamlit UI：多 PDF 上传、st.status 实时状态、Markdown 表格结果区、HITL 人工审核。
6. LLM 调用云端 DeepSeek（OpenAI 兼容端点）：base_url = https://api.deepseek.com，
   model = deepseek-chat。API Key 三种来源任选其一（见下）。

运行：
    pip install -r requirements.txt
    streamlit run app.py

DeepSeek API Key 三种方式任选其一：
    - 左侧侧边栏输入框（存 session，刷新后需要重填）
    - .streamlit/secrets.toml 里写 DEEPSEEK_API_KEY = "sk-..."
    - 环境变量 DEEPSEEK_API_KEY

可调的环境变量（都有默认值，一般不用改）：
    DEEPSEEK_BASE_URL  默认 https://api.deepseek.com
    DEEPSEEK_MODEL     默认 deepseek-chat
    LLM_TIMEOUT        默认 120 秒

数据目录（向量库 + 检查点库）默认在代码目录旁；如需换位置（例如放到可写磁盘），
设置环境变量 APP_DATA_DIR 即可 —— 代码里没有任何写死的盘符路径。
"""

import asyncio
import difflib
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import traceback
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import pandas as pd
import streamlit as st

from core import extract_text

# ---------------- LangChain / LangGraph imports ----------------
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# 注：HuggingFaceEmbeddings 改成按需导入（见 get_embeddings）——
# 默认走本地嵌入，没必要为了启动页面去加载 torch。

# ================================================================
# 0. 基础配置（全部从 __file__ 派生，不写死盘符）
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认放在代码目录旁；要让程序跑在 Program Files / 只读盘 / 打包成 exe 时也能写，
# 用环境变量 APP_DATA_DIR 指到一个可写目录即可。
DATA_DIR = (os.environ.get("APP_DATA_DIR") or "").strip() or BASE_DIR
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "langgraph_checkpoints.sqlite")
MAX_RETRIES = 2  # 安全审查最多打回重试 2 次

# ---------------- 云端 LLM 配置（全局唯一一处，所有节点都走这里） ----------------
# DeepSeek 兼容 OpenAI 协议，所以仍然用 ChatOpenAI 这个客户端，只把地址指到 DeepSeek。
DEEPSEEK_BASE_URL = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT") or "120")

try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError:
    # 目录不可写时不在这里崩，真正用到时再给出可读的报错
    pass

st.set_page_config(page_title="油气田钻井/压裂方案智能生成系统", page_icon="🛢️", layout="wide")


def _get_api_key() -> str:
    """API Key 获取优先级：侧边栏输入框 > .streamlit/secrets.toml > 环境变量。"""
    key = str(st.session_state.get("api_key") or "").strip()
    if key:
        return key
    try:
        secret = st.secrets["DEEPSEEK_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        # 没有 secrets.toml 属于正常情况，不该影响页面
        pass
    return str(os.environ.get("DEEPSEEK_API_KEY") or "").strip()


@st.cache_resource(show_spinner="正在初始化 DeepSeek 客户端...")
def get_llm(api_key: str) -> ChatOpenAI:
    """所有节点共用的唯一 LLM 客户端：指向云端 DeepSeek。
    地质 / 安全 / 生成 三个节点都调这个函数，改配置只改上面那几个常量。"""
    if not api_key:
        raise RuntimeError("未配置 DeepSeek API Key（侧边栏 / secrets.toml / 环境变量 DEEPSEEK_API_KEY）")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=4096,
        timeout=LLM_TIMEOUT,
    )


# ================================================================
# 1. 并发 PDF 解析：asyncio + ThreadPoolExecutor
# ================================================================
def _parse_bytes(data: bytes) -> str:
    """线程池中执行：把上传的 bytes 落成临时文件，再调 core.extract_text（传路径）。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return extract_text(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


async def parse_pdfs_concurrently(items, on_done=None) -> dict:
    """
    并发解析多个 PDF。
    items: [(filename, bytes), ...]
    on_done: 回调 fn(filename, text)，在事件循环所在线程（即 Streamlit 主线程）里执行，
             因此可以安全地 st.write 实时更新界面。
    事件循环跑在 Streamlit 脚本线程里，await 让出控制权时 UI 就能刷新 —— 不会卡死网页。
    """
    loop = asyncio.get_running_loop()
    results: dict = {}

    max_workers = min(4, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pdf-parser") as pool:
        tasks = {
            loop.run_in_executor(pool, _parse_bytes, data): name
            for name, data in items
        }
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                name = tasks[fut]
                try:
                    text = fut.result()
                except Exception as e:  # 单个文件失败不影响其他文件
                    text = f"[EXTRACT_ERROR] {type(e).__name__}: {e}"
                results[name] = text
                if on_done:
                    on_done(name, text)
    return results


# ================================================================
# 2. 本地向量库：langchain_chroma + 嵌入模型（缓存）
# ================================================================
EMBED_BACKEND_ENV = (os.environ.get("EMBED_BACKEND") or "local").strip().lower()
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalHashEmbeddings(Embeddings):
    """零依赖、零下载的本地嵌入：字符 1~3-gram 哈希到固定维度后归一化。

    为什么要有这个：HuggingFace 模型首次要下载几百 MB，本机网络连不上 HF 时
    整条规范入库会直接失败（业务闭环就断了）。规范条款都是短句，字符 n-gram
    的相似度对“压力系数不得超过1.4”这类检索完全够用，而且**离线必然可用**。

    注意必须用 crc32 而不是内置 hash()：后者对字符串每进程随机加盐，
    重启后同样的文本算出不同向量，已入库的旧向量就全部错位了。
    """

    backend = "local"

    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def _vec(self, text) -> list:
        s = re.sub(r"\s+", " ", str(text or ""))
        vec = [0.0] * self.dim
        for n, weight in ((1, 1.0), (2, 1.6), (3, 2.0)):
            for i in range(max(1, len(s) - n + 1)):
                idx = zlib.crc32(s[i:i + n].encode("utf-8")) % self.dim
                vec[idx] += weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts) -> list:
        return [self._vec(t) for t in texts]

    def embed_query(self, text) -> list:
        return self._vec(text)


def embed_backend() -> str:
    """当前嵌入后端名（会进 collection 名，避免不同后端的向量混在一个库里）。"""
    return "hf" if EMBED_BACKEND_ENV in ("hf", "huggingface") else "local"


@st.cache_resource(show_spinner="正在准备嵌入模型...")
def get_embeddings() -> Embeddings:
    """默认本地哈希嵌入（离线可跑）；要语义效果更好就设 EMBED_BACKEND=hf 用 HF 模型。
    HF 加载失败时自动回退本地，绝不因为“模型下不下来”把功能打死。"""
    if embed_backend() == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            st.session_state["embed_warn"] = (
                "HuggingFace 嵌入模型加载失败，已自动改用本地哈希嵌入。\n\n"
                + traceback.format_exc()[-800:]
            )
    return LocalHashEmbeddings()


def _vector_count(vectorstore) -> int:
    """取 collection 里的条数。私有属性跨版本会变，所以留一个公开 API 兜底。"""
    try:
        return int(vectorstore._collection.count())
    except Exception:
        try:
            return len(vectorstore.get()["ids"])
        except Exception:
            return 0


@st.cache_resource(show_spinner="正在构建/加载本地向量库 Chroma ...")
def build_vectorstore(fingerprint: str, _docs_data):
    """
    fingerprint: 所有上传文件内容的哈希（内容变了才会重建 collection，避免重复入库）。
    _docs_data:  [(filename, text), ...]
                 参数名以下划线开头，Streamlit 不参与哈希 —— 否则每次 rerun 都要把
                 几 MB 正文重新哈希一遍；这样缓存键只用 fingerprint，语义也更准确。
    """
    embeddings = get_embeddings()
    collection_name = f"pdfs_{embed_backend()}_{fingerprint[:16]}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    # collection 已有数据就不再重复写入
    if _vector_count(vectorstore) == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [
            Document(page_content=text, metadata={"source": name})
            for name, text in (_docs_data or [])
        ]
        if docs:
            vectorstore.add_documents(splitter.split_documents(docs))
    return vectorstore


def files_fingerprint(items) -> str:
    """对 [(name, bytes)] 计算整体哈希，作为向量库与缓存的 key。"""
    h = hashlib.sha256()
    for name, data in items:
        h.update(name.encode("utf-8"))
        h.update(data)
    return h.hexdigest()


# ================================================================
# 2.5 公司规范知识库（独立 collection，与工程 PDF 库互不干扰）
# ================================================================
SPEC_FILE_LABEL = "公司安全规范"     # 溯源时显示的规范名


def spec_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _decode_text(raw: bytes) -> str:
    """txt 可能是 utf-8 / utf-8-sig / GBK，逐个试，别让编码问题变成“规范为空”."""
    if not raw:
        return ""
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "ignore")


def _split_spec(text: str, chunk_size: int = 600, overlap: int = 1) -> list:
    """按**整行**切分规范：条款编号和条文不会被劈成两半，方便后面定位“第 N 条”。
    每块带 line_from / line_to，出问题时能回到原文第几行核对。"""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    docs, i = [], 0
    while i < len(lines):
        start = i
        buf, length = [], 0
        while i < len(lines) and length < chunk_size:
            buf.append(lines[i])
            length += len(lines[i]) + 1
            i += 1
        docs.append(
            Document(
                page_content="\n".join(buf),
                metadata={"source": SPEC_FILE_LABEL, "line_from": start + 1, "line_to": i},
            )
        )
        if i < len(lines) and overlap:
            i = max(start + 1, i - overlap)   # 回退一行做重叠，至少前进 1 行，避免死循环
    return docs


@st.cache_resource(show_spinner="正在构建公司规范知识库...")
def build_spec_store(fingerprint: str, _spec_text: str):
    """规范库与 PDF 库分开存：collection 名带 spec_ 前缀，内容变了才重建。"""
    embeddings = get_embeddings()
    collection_name = f"spec_{embed_backend()}_{fingerprint[:16]}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    if _vector_count(vectorstore) == 0:
        docs = _split_spec(_spec_text)
        if docs:
            vectorstore.add_documents(docs)
    return vectorstore


class LocalSpecRetriever:
    """向量库不可用时的兜底检索：按查询词在条款块里的命中数排序。
    接口与向量库 retriever 一致（invoke(query) -> list[Document]），上层代码无需改。"""

    def __init__(self, docs, k: int = 8):
        self.docs = docs or []
        self.k = k

    @staticmethod
    def _terms(query) -> list:
        words = [w for w in re.split(r"[\s,，、;；:：/|]+", str(query or "")) if w]
        grams = [w[i:i + 2] for w in words for i in range(len(w) - 1)]
        return list(dict.fromkeys(words + grams))

    def invoke(self, query, config=None, **kwargs):
        terms = self._terms(query)
        def _score(d):
            c = getattr(d, "page_content", "")
            return sum(c.count(t) for t in terms)
        ranked = sorted(self.docs, key=_score, reverse=True)
        hit = [d for d in ranked[:self.k] if _score(d) > 0]
        return hit or ranked[:min(2, len(ranked))]

    # as_retriever 的调用方可能把 search_kwargs 传进来，这里兼容一下
    def as_retriever(self, search_kwargs=None, **kwargs):
        self.k = int((search_kwargs or {}).get("k", self.k) or self.k)
        return self


def make_spec_retriever(text: str) -> tuple:
    """返回 (retriever, 状态说明, 错误堆栈或 None)。
    先用 Chroma 向量库；**任何异常都退到本地条款检索**——强制引用比对不能因为
    向量库/嵌入模型的环境问题而停摆（条文校验本身还有全量原文兜底）。"""
    fp = spec_fingerprint(text)
    try:
        vs = build_spec_store(fp, text)
        return (vs.as_retriever(search_kwargs={"k": 8}),
                f"Chroma 向量检索 · {_vector_count(vs)} 个片段 · 嵌入后端 {embed_backend()}",
                None)
    except Exception:
        return (LocalSpecRetriever(_split_spec(text), k=8),
                f"本地条款检索（向量库不可用，已自动兜底）· 嵌入后端 {embed_backend()}",
                traceback.format_exc())


# ================================================================
# 3. LangGraph 主流程：AgentState + 反思机制
# ================================================================
class AgentState(TypedDict):
    raw_text: dict            # {文件名: 提取文本}
    extracted_params: dict    # geology_node 提取的地质参数
    param_citations: dict     # 每个参数在原文里的定位 {参数: {value,file,page,snippet}}
    conflict_report: dict     # 强制引用冲突报告 {is_safe, violations[], hard_problems[]}
    risk_level: str           # low / medium / high / rejected / unknown
    safety_report: str        # safety_node 审查结论（含条文引用）
    safety_pass: bool         # 审查是否通过
    final_report: str         # 最终方案（Markdown）
    retry_count: int          # 已被打回重试次数（上限 MAX_RETRIES）
    hitl: bool                # 是否启用 HITL 人工审核
    emergency_triggered: bool # 是否已熔断报警


def parse_json_safe(text: str):
    """从 LLM 输出中尽力解析 JSON（容忍 ```json 围栏、前后废话）。"""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start: end + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def _as_bool(value) -> bool:
    """安全闸门专用：只认白名单里的“真”，"false" 这种字符串一律当假。
    （直接用 bool("false") 会得到 True，是这类校验的经典坑。）"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "pass", "ok", "1", "通过")
    return False


def is_garbled(text: str) -> bool:
    """乱码检测：替换符过多 / 可读字符占比过低 / 文本过短。"""
    if not isinstance(text, str) or not text.strip() or len(text.strip()) < 20:
        return True
    if text.count("\ufffd") / len(text) > 0.05:
        return True
    useful = sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return useful / len(text) < 0.30


# ---------------- 强制引用：定位原文 / 校验条文（不信任模型自称的出处） ----------------
# core.py 每页开头都会写 "===== 第 N 页 ====="，靠它把字符偏移换算成页码。
_PAGE_RE = re.compile(r"=====\s*第\s*(\d+)\s*页\s*=====")
# 切句：句末标点 + 换行 + “；”
_SENT_RE = re.compile(r"[。；;！!？?\n]")
# 比对前先把空白和标点都去掉，避免“1.4”和“1.4 ”被判成两回事
_NORM_RE = re.compile(r"[\s，,。；;：:、\"'“”‘’（）()\[\]【】《》\-—_/|·]+")


def _norm(s) -> str:
    return _NORM_RE.sub("", str(s or ""))


def _page_of(text: str, offset: int) -> int | None:
    """把字符偏移换算成页码：取该位置之前最后一个页标记。"""
    page = None
    for m in _PAGE_RE.finditer(text, 0, max(0, offset)):
        page = int(m.group(1))
    return page


def _value_token(value) -> str:
    """从一个参数值里取出可用于全文检索的“锚点”：
    数字优先（1.58 / 3200），没有数字就取中文连续子串。"""
    s = str(value or "").strip()
    if not s:
        return ""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        return m.group(0)
    zh = "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")
    return zh[:8] if len(zh) >= 2 else ""


def _sentence_around(text: str, idx: int, max_len: int = 160) -> str:
    """取包含 idx 的那句话（按句末标点/换行切），过长就从 idx 附近截。"""
    ls = max(text.rfind("\n", 0, idx), -1) + 1
    # 往前找句末标点
    left = ls
    for m in _SENT_RE.finditer(text, ls, idx):
        left = m.end()
    right = len(text)
    m = _SENT_RE.search(text, idx)
    if m:
        right = m.end()
    snippet = text[left:right].strip()
    if len(snippet) < 12:      # 太短（例如数字单独一行）就向前后各扩一行
        ls2 = text.rfind("\n", 0, left) + 1
        m2 = _SENT_RE.search(text, right)
        re2 = text.find("\n", right + 1)
        snippet = text[ls2:re2 if re2 > 0 else len(text)].strip()
    if len(snippet) > max_len:
        core = text[idx:idx + 8]
        p = snippet.find(core)
        start = max(0, (p if p >= 0 else 0) - 60)
        snippet = ("…" if start > 0 else "") + snippet[start:start + max_len] + "…"
    return snippet


# 参数名 → 原文里常见的表述（用来给候选出现位置打分，挑“最像这个参数”的那一处）
_KEY_HINTS = {
    "pressure": ("压力", "压力系数", "地层压力"),
    "formation": ("地层",),
    "depth": ("井深", "深度"),
    "lithology": ("岩性",),
    "porosity": ("孔隙度",),
    "permeability": ("渗透率",),
    "density": ("密度",),
    "temperature": ("温度",),
    "well": ("井",),
}
# 坐标/经纬度里的数字最容易和参数值撞车，要压低优先级
_COORD_RE = re.compile(r"[°′″]|经纬|坐标|N\s*\d+°|E\s*\d+°")


def _key_hints(key: str) -> tuple:
    k = str(key or "").lower()
    hints = []
    for tok, words in _KEY_HINTS.items():
        if tok in k:
            hints.extend(words)
    zh = "".join(ch for ch in str(key) if "\u4e00" <= ch <= "\u9fff")
    if zh:
        hints.append(zh)
    return tuple(dict.fromkeys(hints))


def locate_in_raw(value, raw_text: dict, key: str = "") -> dict:
    """在原文里找这个参数值，返回 {value,file,page,snippet,token}；找不到返回 {}。
    同一个数字可能在原文出现多次（例如 18.6 既可能是压力也可能是坐标），
    所以按“句子里有没有该参数的关键词”打分，取最高分那一处，而不是碰运气取第一个。"""
    token = _value_token(value)
    if not token:
        return {}
    hints = _key_hints(key)
    best, best_score = {}, -99
    for name, text in (raw_text or {}).items():
        if not isinstance(text, str):
            continue
        start = 0
        while True:
            idx = text.find(token, start)
            if idx < 0:
                break
            start = idx + len(token)
            nl = text.find("\n", idx)
            line = text[text.rfind("\n", 0, idx) + 1:nl if nl > 0 else len(text)]
            # 跳过页标记行、OCR 提示行，否则 "第 3 页" 里的 3 会被当成参数值
            if line.strip().startswith("=====") or line.strip().startswith("【本页无文字层"):
                continue
            snippet = _sentence_around(text, idx)
            score = 0
            if hints and any(h in snippet for h in hints):
                score += 3
            if _COORD_RE.search(snippet) and not any(h in snippet for h in hints):
                score -= 4          # 坐标里的数字，明显不是我们要的参数
            if score > best_score:
                best_score = score
                best = {"value": value, "token": token, "file": name,
                        "page": _page_of(text, idx), "snippet": snippet}
    return best or {"value": value, "token": token}


def _clip_page(page) -> str:
    return f"第 {page} 页" if page else "页码未知"


def format_citations(citations: dict) -> str:
    """把参数溯源信息排成模型能读的清单。"""
    if not citations:
        return "（未能定位到原文，请按原文谨慎判断）"
    out = []
    for k, c in citations.items():
        if not isinstance(c, dict):
            continue
        if c.get("snippet"):
            out.append(
                f"- 【{k}】= {c.get('value')}　←　《{c.get('file')}》{_clip_page(c.get('page'))}："
                f"“{c.get('snippet')}”"
            )
        else:
            out.append(f"- 【{k}】= {c.get('value')}　←　（原文中未定位到该值，存疑）")
    return "\n".join(out)


def _rule_body(rule_source: str) -> str:
    """把 "《公司安全规范》第3条：压力系数不得超过1.4" 里的条文正文抽出来。"""
    s = str(rule_source or "").strip()
    parts = [p.strip() for p in re.split(r"[：:]", s) if p.strip()]
    return parts[-1] if parts else s


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _similar_in_line(base_n: str, base_l: str, threshold: float = 0.6) -> bool:
    """把子句和规范里“最长公共片段所在的那一段”比相似度。
    直接整行比会被前后文稀释（一行里有条号、补充说明），所以先对齐再比。"""
    if not base_n or not base_l:
        return False
    sm = difflib.SequenceMatcher(None, base_n, base_l, autojunk=False)
    m = sm.find_longest_match(0, len(base_n), 0, len(base_l))
    if m.size < max(3, len(base_n) // 3):
        return False
    start = max(0, m.b - m.a)
    window = base_l[start:start + len(base_n)]
    if not window:
        return False
    return difflib.SequenceMatcher(None, base_n, window, autojunk=False).ratio() >= threshold


def _clause_hit(clause: str, source_lines: list) -> bool:
    """一个子句是否能在规范里落地：
    1) 去标点后整句命中 → 算；
    2) 否则要求**数字阈值必须逐字一致**（1.4 和 9.9 绝不能算同一回事），
       且去掉数字后的文字相似度 ≥ 0.6（容忍“不能超过 / 不得超过”这类转述）。"""
    n = _norm(clause)
    if len(n) < 4:
        return False
    joined = "".join(source_lines)          # 已 normalize 的行
    if n in joined:
        return True
    nums = _NUM_RE.findall(n)
    if not nums:
        return False                        # 没有数字又没有原句命中 → 不敢认
    base_n = _NUM_RE.sub("", n)
    for line in source_lines:
        if not all(x in line for x in nums):
            continue                        # 数字对不上，直接否
        if _similar_in_line(base_n, _NUM_RE.sub("", line)):
            return True
    return False


def quote_coverage(quote: str, source: str) -> tuple[float, str]:
    """条文引用的落地程度：把引用拆成子句，看有几成能在规范原文里找到。
    返回 (命中比例, 命中的第一个子句)。纯编造的基本 0；转述能过；改数字过不了。"""
    if not quote or not str(source or "").strip():
        return 0.0, ""
    source_lines = [_norm(ln) for ln in str(source).splitlines() if _norm(ln)]
    clauses = [c for c in re.split(r"[，,。；;、\n]", str(quote)) if len(_norm(c)) >= 4]
    if not clauses:
        clauses = [str(quote)]
    hits = [c for c in clauses if _clause_hit(c, source_lines)]
    return len(hits) / max(1, len(clauses)), (hits[0] if hits else "")


def normalize_violation(v: dict, citations: dict, raw_text: dict, verify_source: str) -> dict:
    """把模型给出的违规项做“落地校正”：
    1) 参数出处用我们自己定位到的页码/原句覆盖（模型最爱在这里编页码）；
    2) 条文引用必须能在已上传规范里找到，否则打上未核实标记（fail-closed，仍然拦）。"""
    out = {
        "parameter": str(v.get("parameter") or v.get("param") or "未指明参数").strip(),
        "extracted_value": str(v.get("extracted_value") or v.get("value") or "").strip(),
        "extracted_source": str(v.get("extracted_source") or v.get("source") or "").strip(),
        "rule_source": str(v.get("rule_source") or v.get("rule") or "").strip(),
        "comment": str(v.get("comment") or v.get("reason") or "").strip(),
    }
    # ---- 参数出处：用真实定位覆盖 ----
    key = out["parameter"]
    c = citations.get(key)
    if c is None:
        for k, cc in (citations or {}).items():
            if key and (key in k or k in key):
                c = cc
                break
    if isinstance(c, dict) and c.get("snippet"):
        out["extracted_source"] = (
            f"《{c.get('file')}》{_clip_page(c.get('page'))}：“{c.get('snippet')}”"
        )
        out["citation_verified"] = True
        if c.get("value"):
            out["extracted_value"] = str(c.get("value"))
    else:
        probe = locate_in_raw(out["extracted_value"], raw_text, out["parameter"])
        if probe.get("snippet"):
            out["extracted_source"] = (
                f"《{probe.get('file')}》{_clip_page(probe.get('page'))}：“{probe.get('snippet')}”"
            )
            out["citation_verified"] = True
        else:
            out["citation_verified"] = False
            out["extracted_source"] = out["extracted_source"] or "⚠️ 未能在原始 PDF 中定位到该数值"
    # ---- 规范条文：必须落地 ----
    ratio, matched = quote_coverage(_rule_body(out["rule_source"]), verify_source)
    out["rule_verified"] = ratio >= 0.6
    out["rule_match_ratio"] = round(ratio, 2)
    if matched and not out["rule_verified"]:
        out["rule_note"] = f"仅能匹配到片段：“{matched}”"
    if not out["rule_verified"]:
        out["comment"] = (
            out["comment"]
            + "　⚠️ 该条文引用未能在已上传规范中定位（疑似模型臆造），已按不安全处理，请人工核对原文。"
        ).strip()
    return out


# ---------------- 节点 1：油气地质专家，提取地层与工程参数 ----------------
# 注意：第二个参数必须标注成 RunnableConfig，LangGraph 才会把 config 注入进来。
# 标成 dict（或漏标注）在 langgraph 1.x 上会直接报 missing positional argument。
def geology_node(state: AgentState, config: RunnableConfig) -> dict:
    # 被打回重做时，把上一轮审查意见带进 prompt，否则同一份资料会重复犯同样的错
    retry_count = int(state.get("retry_count", 0) or 0)
    retry_note = ""
    if retry_count > 0:
        retry_note = (
            f"\n\n【上一轮被打回的原因（第 {retry_count} 次打回）】\n"
            f"{state.get('safety_report', '') or '（未记录）'}\n"
            "请务必针对上述问题修正你的参数提取结果。"
        )

    raw_text = state.get("raw_text") or {}
    context = "\n\n".join(
        f"【文件: {name}】\n{text[:3000]}" for name, text in raw_text.items()
    )

    # 从本地向量库检索相关片段，辅助参数提取（RAG）
    rag_context = ""
    retriever = (config or {}).get("configurable", {}).get("retriever")
    if retriever:
        try:
            docs = retriever.invoke("井深 地层压力 岩性 孔隙度 渗透率")
            rag_context = "\n".join(f"- {d.page_content}" for d in docs)
        except Exception:
            rag_context = ""

    prompt = f"""你是资深油气勘探开发地质专家，请提取地层与工程参数。

【工程资料原文】
{context}

【向量库检索补充】
{rag_context}
{retry_note}

请严格输出 JSON（不要输出多余解释），字段如下：
{{
  "well_name": "井名",
  "depth": "井深（含单位）",
  "formation_pressure": "地层压力系数或压力值",
  "lithology": "岩性描述",
  "risk_factors": ["关键风险1（如井漏/井喷）", "关键风险2"]
}}"""
    try:
        llm = get_llm(_get_api_key())
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", "") or ""
        params = parse_json_safe(content)
        if not isinstance(params, dict):
            # 解析失败也把模型原文留一点，方便排查（不再用 dir() 猜变量在不在）
            params = {"_parse_error": True,
                      "_raw": (content[:500] if isinstance(content, str) else str(content)[:500])}
    except Exception as e:
        params = {"_parse_error": True, "_error": f"{type(e).__name__}: {e}"}

    return {"extracted_params": params}


# ---------------- 节点 2：安全专家，强制条文比对（带引用溯源） ----------------
def safety_node(state: AgentState, config: RunnableConfig) -> dict:
    """铁血闭环的核心：参数逐条对照公司规范，输出带【页码级引用】的 JSON 冲突报告。
    设计原则：
      1. 页码和原句由程序从 PDF 原文里算出来，不采信模型自称的出处；
      2. 条文引用必须能在已上传规范里找到，找不到就打“未核实”标记，且依然拦（fail-closed）；
      3. 任何一项违规 → safety_pass=False → 条件边打回重做 / 熔断。
    """
    cc = ((config or {}).get("configurable") or {}) if isinstance(config, dict) else {}
    spec_retriever = cc.get("spec_retriever")
    spec_text = str(cc.get("spec_text") or "")

    raw_text = state.get("raw_text") or {}
    params = state.get("extracted_params") or {}
    if not isinstance(params, dict):
        params = {"_parse_error": True, "_raw": str(params)[:500]}

    # ---- 步骤 1：给每个参数在原文里定位（页码 + 原句）----
    citations = {}
    for k, v in params.items():
        if str(k).startswith("_"):
            continue
        citations[str(k)] = locate_in_raw(v, raw_text, str(k))
    citations = {k: c for k, c in citations.items() if c}

    # ---- 步骤 2：检索公司规范（专用 retriever）----
    param_words = " ".join(k for k in params.keys() if not str(k).startswith("_"))
    query = f"井控 压力系数 井深 井漏 井喷 安全规范 上限 禁止 {param_words}".strip()
    spec_chunks = []
    if spec_retriever:
        try:
            spec_chunks = [d.page_content for d in (spec_retriever.invoke(query) or []) if getattr(d, "page_content", "")]
        except Exception:
            spec_chunks = []
    spec_context = "\n\n".join(f"[规范片段 {i + 1}]\n{c}" for i, c in enumerate(spec_chunks)) or "（未检索到规范片段）"
    # 公司规范一般就几页：能全文喂就直接喂全文，检索漏召回也不会让模型漏掉关键条款
    if spec_text and len(spec_text) <= 20000:
        spec_context = f"【公司规范全文】\n{spec_text}"
    spec_chunk_count = max(1, len(spec_chunks))
    # 条文的落地校验要拿“全量规范 + 检索片段”一起比，片段可能只覆盖一部分
    verify_source = ("\n".join(spec_chunks) + "\n" + spec_text).strip()

    # ---- 步骤 3：硬性阻断项（这些跟条文比对无关，但一律不许放行）----
    hard_problems = []
    texts = [t for t in raw_text.values() if isinstance(t, str) and t.strip()]
    if not texts or all(is_garbled(t) for t in texts):
        hard_problems.append("原始文本疑似乱码或为空（可能是扫描件，需要 OCR）")
    if params.get("_parse_error"):
        hard_problems.append(f"参数提取失败：{params.get('_error', params.get('_raw', '未知'))}")
    if not verify_source:
        hard_problems.append("未上传公司安全规范，无法执行强制条文比对")
    if is_garbled(json.dumps(params, ensure_ascii=False, default=str)):
        hard_problems.append("提取参数中存在乱码")

    # ---- 步骤 4：条文比对（模型只负责“比”，出处由程序回填）----
    review_prompt = f"""你是油气工程安全审查专家。请把【地质参数】逐条对照【公司规范】，找出违规项。

【地质参数（已定位到原文，出处以此为准）】
{format_citations(citations)}

【参数 JSON】
{json.dumps(params, ensure_ascii=False, indent=2, default=str)}

【公司规范检索片段】
{spec_context}

【判定要求】
1. 只要有一条参数违反规范，is_safe 必须为 false；
2. 每条违规必须写明违背了规范的哪一条，并**原样引用该条的关键句**（不许改写、不许编造、不许编号对不上）；
3. extracted_source 必须写清“哪个文件、第几页、原句”；
4. 参数全部合规时，violations 输出空数组、is_safe 为 true；
5. 只输出 JSON，不要解释文字、不要 Markdown 围栏。

【输出格式】
{{
  "is_safe": false,
  "violations": [
    {{
      "parameter": "压力系数",
      "extracted_value": "1.58",
      "extracted_source": "地质设计 PDF 第 2 页：“地层压力系数1.58”",
      "rule_source": "《公司安全规范》第3条：压力系数不得超过1.4",
      "comment": "严重违规，存在井喷风险，必须打回！"
    }}
  ]
}}"""
    report, violations, is_safe, raw_answer = None, [], False, ""
    try:
        llm = get_llm(_get_api_key())
        resp = llm.invoke(review_prompt)
        raw_answer = getattr(resp, "content", "") or ""
        report = parse_json_safe(raw_answer)
    except Exception as e:
        hard_problems.append(f"安全审查模型异常：{type(e).__name__}: {e}")

    if isinstance(report, dict):
        is_safe = _as_bool(report.get("is_safe", report.get("pass", False)))
        raw_viol = report.get("violations")
        if raw_viol is None:
            raw_viol = report.get("issues") or []
        if isinstance(raw_viol, dict):
            raw_viol = [raw_viol]
        for v in (raw_viol if isinstance(raw_viol, list) else []):
            if isinstance(v, dict):
                violations.append(normalize_violation(v, citations, raw_text, verify_source))
    elif not hard_problems:
        hard_problems.append(
            "审查模型未返回可解析的 JSON 冲突报告"
            + (f"（模型原文：{raw_answer[:200]}）" if raw_answer else "")
        )

    # 模型自相矛盾时以“有违规”为准：安全闸门只认最坏情况
    if is_safe and violations:
        hard_problems.append(f"模型给了 is_safe=true，但同时列出 {len(violations)} 条违规，按违规处理")
        is_safe = False
    if not is_safe and not violations and not hard_problems:
        hard_problems.append("模型判定不安全但未给出具体违规项，需人工复核")

    passed = bool(is_safe) and not violations and not hard_problems

    # ---- 步骤 5：生成可直接展示的文字结论（含两处引用）----
    lines = []
    if hard_problems:
        lines.append("【硬性阻断】")
        lines += [f"- {p}" for p in hard_problems]
    if violations:
        lines.append(f"【发现 {len(violations)} 条违规】")
        for i, v in enumerate(violations, 1):
            lines.append(
                f"{i}. {v.get('parameter')} = {v.get('extracted_value')}\n"
                f"   地质来源：{v.get('extracted_source')}\n"
                f"   规范来源：{v.get('rule_source')}"
                + ("" if v.get("rule_verified") else "　〔条文未核实〕") + "\n"
                f"   判定：{v.get('comment')}"
            )
    if passed:
        lines.append(
            f"判定：通过 —— 已比对 {len(citations)} 项参数 / {len(spec_chunks)} 条规范片段，未发现违规。"
        )
    safety_report = "\n".join(lines) or "（无审查结论）"

    risk_level = "high" if not passed else "medium"
    new_retry = int(state.get("retry_count", 0) or 0) + (0 if passed else 1)

    return {
        "safety_pass": passed,
        "risk_level": risk_level,
        "safety_report": safety_report,
        "retry_count": new_retry,
        "param_citations": citations,
        "conflict_report": {
            "is_safe": passed,
            "violations": violations,
            "hard_problems": hard_problems,
            "checked_params": len(citations),
            "spec_chunk_count": spec_chunk_count,
            "model_is_safe": bool(is_safe),
        },
    }


# ---------------- 条件边：safety_node 之后去哪里 ----------------
def route_after_safety(state: AgentState) -> str:
    if state.get("safety_pass"):
        return "generate"                    # 审查通过 -> 生成最终方案
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "emergency"                   # 重试 2 次仍不通过 -> 熔断报警
    return "retry"                           # 打回 geology_node 重做


# ---------------- 节点 3：熔断报警 ----------------
def emergency_node(state: AgentState) -> dict:
    report = f"""## 🚨 熔断报警（Emergency）

安全专家已连续 **{MAX_RETRIES} 次** 审查不通过，系统已熔断，禁止自动生成方案，请人工介入！

| 项目 | 内容 |
|---|---|
| 当前风险等级 | **{state.get('risk_level', 'unknown')}** |
| 已重试次数 | {state.get('retry_count', 0)} / {MAX_RETRIES} |
| 审查失败原因 | {state.get('safety_report', '-')} |

**建议**：
1. 检查 PDF 是否为扫描件（需 OCR）；
2. 核对原始勘察资料是否完整；
3. 修改资料后，更换新的 thread_id 重新发起分析。
"""
    return {
        "final_report": report,
        "emergency_triggered": True,
        "risk_level": "high",
    }


# ---------------- 节点 4：生成最终方案（含 HITL 人工审核闸门） ----------------
def generate_node(state: AgentState) -> dict:
    # HITL：启用人工审核时，在此处 interrupt，等待前端人工确认后 resume
    if state.get("hitl"):
        decision = interrupt({
            "question": "请人工审核地质参数与安全审查结论，确认后才生成最终方案",
            "extracted_params": state.get("extracted_params", {}),
            "param_citations": state.get("param_citations", {}),
            "conflict_report": state.get("conflict_report", {}),
            "risk_level": state.get("risk_level", "unknown"),
            "safety_report": state.get("safety_report", ""),
        })
        if not (isinstance(decision, dict) and decision.get("approved")):
            return {
                "final_report": "> ❌ **人工审核未通过**：本次方案生成已被人工否决。"
                                f"\n\n审核意见：{(decision or {}).get('comment', '未填写')}",
                "risk_level": "rejected",
            }

    params = state.get("extracted_params", {})
    prompt = f"""你是资深油气钻完井与压裂方案设计总工。请基于以下已通过安全审查的地层与工程参数，
生成一份结构清晰的钻井/压裂施工方案（Markdown 格式）。

【已审查通过的地层与工程参数】
{json.dumps(params, ensure_ascii=False, indent=2, default=str)}

【安全审查结论（含条文引用，写方案时请据此标注依据）】
{state.get('safety_report', '') or '（未记录）'}

【风险等级】{state.get('risk_level')}

要求：
1. 必须包含至少一个漂亮的 Markdown 表格（参数汇总表 / 分项施工措施表）；
2. 内容包含：井况概况、地层与工程参数汇总表、工程风险分析（井漏/井喷/井壁失稳等）、
   钻井液与井控措施、压裂段簇与泵注参数建议、监测与应急预案；
3. 语言专业、条理清晰，用二级/三级标题组织。"""
    # 流水线里唯一直接产出成品的调用：失败必须兜住，不能让页面红屏
    try:
        llm = get_llm(_get_api_key())
        resp = llm.invoke(prompt)
        content = (getattr(resp, "content", "") or "").strip()
    except Exception as e:
        return {
            "final_report": (
                "> ⚠️ **最终方案生成失败**，参数与审查结论已保留，可稍后重试。\n\n"
                f"错误信息：`{type(e).__name__}: {e}`\n\n"
                "常见原因：网络中断 / API Key 失效 / 模型服务限流。"
            ),
        }
    if not content:
        return {"final_report": "> ⚠️ 模型返回了空内容，请重试或更换模型。"}
    return {"final_report": content}


# ---------------- 组图 + SqliteSaver 持久化 ----------------
@st.cache_resource(show_spinner="正在编译 LangGraph 工作流（含 SqliteSaver）...")
def get_graph():
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(AgentState)
    builder.add_node("geology_node", geology_node)
    builder.add_node("safety_node", safety_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("emergency_node", emergency_node)

    builder.add_edge(START, "geology_node")
    builder.add_edge("geology_node", "safety_node")
    builder.add_conditional_edges(
        "safety_node",
        route_after_safety,
        {"retry": "geology_node", "generate": "generate_node", "emergency": "emergency_node"},
    )
    builder.add_edge("generate_node", END)
    builder.add_edge("emergency_node", END)

    return builder.compile(checkpointer=checkpointer)


# ================================================================
# 4. 会话 ID：写进 URL，刷新能读回
# ================================================================
def _tid_from_query() -> str:
    """从 URL 的 ?tid=... 读回会话 ID（格式不对就当没有）。"""
    try:
        val = st.query_params.get("tid")
    except Exception:
        return ""
    if isinstance(val, list):
        val = val[0] if val else ""
    tid = str(val or "").strip()
    return tid if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", tid) else ""


def _sync_tid_to_query(tid: str) -> None:
    try:
        st.query_params["tid"] = tid
    except Exception:
        pass


# 必须在任何 widget 之前把会话 ID 定下来，否则侧边栏输入框会拿旧值覆盖回去
_pending_tid = st.session_state.pop("_pending_tid", None)
if _pending_tid:
    st.session_state["thread_id"] = _pending_tid
if not str(st.session_state.get("thread_id") or "").strip():
    st.session_state["thread_id"] = _tid_from_query() or uuid.uuid4().hex
if _pending_tid or "tid_box" not in st.session_state:
    st.session_state["tid_box"] = st.session_state["thread_id"]


# ================================================================
# 5. Streamlit UI
# ================================================================
st.title("🛢️ 油气田钻井/压裂方案智能生成系统")
st.caption("多 PDF 并发解析 → 本地向量库 → LangGraph 反思工作流（安全审查 / 重试 / 熔断）→ HITL 人工审核")

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("⚙️ 配置")
    # 用 key 直接绑定 session：输入 / 清空都会即时反映到 st.session_state["api_key"]。
    # 注意：widget 绑定 key 之后，本轮脚本里不能再往同一个 key 赋值（Streamlit 会直接抛错），
    # 所以这里不做任何手工同步，读到空串时 _get_api_key() 会自动回落到 secrets / 环境变量。
    st.text_input(
        "DeepSeek API Key", key="api_key", type="password",
        help="也可写入 .streamlit/secrets.toml 或环境变量 DEEPSEEK_API_KEY。",
    )
    st.caption(f"模型：`{DEEPSEEK_MODEL}`　接口：`{DEEPSEEK_BASE_URL}`")

    # ---------------- 企业规范上传：强制引用比对的基准 ----------------
    st.divider()
    st.subheader("📘 公司安全规范")
    spec_file = st.file_uploader(
        "上传《公司安全规范》(.txt / .md)", type=["txt", "md"], key="spec_uploader",
        help="安全审查节点会把地质参数逐条对照这里的条文；每条违规都必须引用到具体条款才放行。",
    )
    spec_text, spec_retriever = "", None
    if spec_file is not None:
        spec_text = _decode_text(spec_file.getvalue()).strip()
        if len(spec_text) < 30:
            st.error("规范内容过短（不足 30 字符），请确认文件没传错。")
            spec_text = ""
        else:
            spec_retriever, spec_note, spec_err = make_spec_retriever(spec_text)
            st.caption(f"✅ 规范已就绪：{len(spec_text)} 字符 · {spec_note}")
            if spec_err:
                with st.expander("向量库降级详情（排查用，点开看堆栈）"):
                    st.code(spec_err[-1500:], language="text")
    else:
        st.caption("⚠️ 未上传：强制引用比对没有基准，点「开始分析」会被拦下。")
    # 存进 session：主区按钮 / 待执行任务都要用（retriever 只是对象句柄，不进 checkpoint）
    st.session_state["spec_text"] = spec_text
    st.session_state["spec_retriever"] = spec_retriever

    tid_input = st.text_input(
        "会话 ID (thread_id)", key="tid_box",
        help="会话 ID 已同步到浏览器地址栏（?tid=...）。刷新页面会自动读回，"
             "LangGraph 的进度（State）就还在，不会丢。",
    )
    if str(tid_input or "").strip():
        st.session_state["thread_id"] = str(tid_input).strip()
    _sync_tid_to_query(st.session_state["thread_id"])
    st.caption(f"当前生效：`{st.session_state['thread_id']}`")

    hitl_enabled = st.checkbox("🧑‍⚖️ 启用 HITL 人工审核", value=True)
    st.caption("启用后，生成最终方案前会暂停，等待人工确认。")

    if st.button("🗑️ 开启新会话（换 thread_id）"):
        for _k in ("raw_text", "fingerprint", "parse_errors", "last_error"):
            st.session_state.pop(_k, None)
        st.session_state["_pending_tid"] = uuid.uuid4().hex
        st.rerun()

    st.divider()
    st.caption(f"数据目录：`{DATA_DIR}`\n\n向量库：`{CHROMA_PERSIST_DIR}`\n\n检查点库：`{CHECKPOINT_DB_PATH}`")

# ---------------- 主区：上传 ----------------
uploaded_files = st.file_uploader(
    "📤 上传工程 PDF（支持多选，将并发解析）",
    type=["pdf"],
    accept_multiple_files=True,
)

try:
    graph = get_graph()
except Exception as e:
    st.error(f"工作流初始化失败：{type(e).__name__}: {e}\n\n请检查数据目录是否可写：`{DATA_DIR}`")
    st.stop()


def _cfg(retriever=None, spec_retriever=None, spec_text="") -> dict:
    """图的 config。retriever / spec_retriever / spec_text 都不是可序列化状态，
    只随 config 传给本次运行，不进 checkpoint。"""
    return {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "retriever": retriever,
            "spec_retriever": spec_retriever,
            "spec_text": spec_text,
        }
    }


def _peek_snapshot():
    """读当前会话的 checkpoint 快照；任何异常都不许冒到页面。"""
    try:
        return graph.get_state(_cfg())
    except Exception:
        return None


def _has_pending_interrupt(snap) -> bool:
    """图是否停在某个 interrupt 上 —— 这是“要不要显示审核面板”的唯一事实来源，
    不再依赖 st.session_state 里那个一刷新就没的标记。"""
    if snap is None:
        return False
    if getattr(snap, "interrupts", None):
        return True
    for task in (getattr(snap, "tasks", None) or ()):
        if getattr(task, "interrupts", None):
            return True
    return False


def _node_label(node: str, delta: dict, attempt: int = 0) -> str:
    """把节点执行情况翻译成用户友好的状态文案。"""
    if node == "geology_node":
        n = attempt or (int(delta.get("retry_count", 0) or 0) + 1)
        return f"🔍 地质专家正在提取参数...（第 {n} 次尝试）"
    if node == "safety_node":
        if delta.get("safety_pass"):
            return "✅ 安全审查通过，准备生成最终方案"
        if int(delta.get("retry_count", 0) or 0) >= MAX_RETRIES:
            return "🚨 连续审查失败，触发熔断报警！"
        return f"⚠️ 被安全专家打回重做中...（第 {delta.get('retry_count', 0)} 次）"
    if node == "generate_node":
        return "📝 正在生成最终方案..."
    if node == "emergency_node":
        return "🚨 熔断报警！等待人工介入"
    return f"⚙️ {node} 执行完成"


def _render_conflict(cr) -> None:
    """把冲突报告渲染成“哪条参数 / 地质出处 / 规范出处”的三段式报错。"""
    if not isinstance(cr, dict):
        st.warning("未生成冲突报告（安全审查可能未执行）。")
        return
    hard = cr.get("hard_problems") or []
    for h in hard:
        st.error(f"⛔ 硬性阻断：{h}")
    for i, v in enumerate(cr.get("violations") or [], 1):
        note = "" if v.get("rule_verified") else (
            "\n\n> ⚠️ 该条文引用未能在已上传规范中定位（疑似模型臆造），已按不安全处理，请人工核对原文。"
        )
        st.error(
            f"**🚫 违规 {i}：{v.get('parameter')} = {v.get('extracted_value')}**\n\n"
            f"**地质来源（原文定位）**：{v.get('extracted_source')}\n\n"
            f"**规范来源（公司规范）**：{v.get('rule_source')}\n\n"
            f"**判定意见**：{v.get('comment')}{note}"
        )
    if not hard and not (cr.get("violations") or []):
        st.success(
            f"✅ 强制比对通过：{cr.get('checked_params', 0)} 项参数 / "
            f"{cr.get('spec_chunk_count', 0)} 条规范片段，未发现违规。"
        )


def _run_stream(inputs, cfg) -> bool:
    """流式执行图，实时更新 st.status；返回是否发生 interrupt。
    所有异常在这里收口：宁可给一句人话报错，也不让页面红屏、不让 status 永远停在“执行中”。"""
    interrupted, attempt = False, 0
    st.session_state["last_error"] = None
    with st.status("⏳ LangGraph 工作流执行中...", expanded=True) as status:
        try:
            for chunk in graph.stream(inputs, config=cfg, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupted = True
                    status.update(label="⏸️ 等待 HITL 人工审核...", state="running")
                    continue
                for node, delta in chunk.items():
                    d = delta if isinstance(delta, dict) else {}
                    if node == "geology_node":
                        attempt += 1
                    label = _node_label(node, d, attempt)
                    st.write(f"**{label}**")
                    if node == "safety_node" and not d.get("safety_pass", True):
                        st.warning(f"审查结论：{d.get('safety_report', '')}")
                        _render_conflict(d.get("conflict_report") or {})
                    status.update(label=label, state="running")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            st.session_state["last_error"] = msg
            status.update(label="⚠️ 执行出错，已中止", state="error")
            st.error(f"工作流执行失败：{msg}")
            return interrupted

        if not interrupted:
            emergency = False
            try:
                emergency = bool((graph.get_state(cfg).values or {}).get("emergency_triggered"))
            except Exception:
                pass
            status.update(
                label="🚨 已熔断" if emergency else "✅ 工作流执行完毕",
                state="error" if emergency else "complete",
            )
    return interrupted


def _run_pipeline(files, hitl_on: bool) -> None:
    """一次完整的分析：校验 Key → 校验文件 → 并发解析 → 入库 → 跑图。每一步都有兜底。"""
    if not _get_api_key():
        st.error(
            "未配置 DeepSeek API Key：请在左侧侧边栏填写，或写入 `.streamlit/secrets.toml`，"
            "或设置环境变量 `DEEPSEEK_API_KEY`。"
        )
        return

    # 强制引用闭环的前置条件：没有规范就没有比对基准，宁可不开跑也不给“假通过”
    if not str(st.session_state.get("spec_text") or "").strip():
        st.error(
            "未上传《公司安全规范》：强制引用溯源必须以规范条文为基准。\n\n"
            "请在左侧「📘 公司安全规范」上传 .txt / .md 文件后再点「开始分析」。"
        )
        return

    items = [(f.name, f.getvalue()) for f in files]
    empty = [n for n, b in items if not b]
    if empty:
        st.warning("以下文件是 0 字节空文件，已跳过：" + "、".join(empty))
    items = [(n, b) for n, b in items if b]
    if not items:
        st.error("没有可解析的 PDF 文件。")
        return

    fp = files_fingerprint(items)
    st.session_state["fingerprint"] = fp
    st.session_state["last_error"] = None

    # 第一步：并发解析
    with st.status("📥 正在并发解析 PDF（asyncio + ThreadPoolExecutor）...", expanded=True) as s1:
        def _on_parsed(name, text):
            if text and not str(text).startswith("[EXTRACT_ERROR]"):
                st.write(f"✅ **{name}** 解析完成（{len(text)} 字符）")
            else:
                st.write(f"❌ **{name}** 解析失败：{str(text)[:200]}")

        try:
            results = asyncio.run(parse_pdfs_concurrently(items, on_done=_on_parsed))
        except Exception as e:
            s1.update(label="📥 解析阶段整体失败", state="error")
            st.error(f"解析阶段整体失败：{type(e).__name__}: {e}")
            return

    valid = {n: t for n, t in results.items() if t and not str(t).startswith("[EXTRACT_ERROR]")}
    failed = {n: t for n, t in results.items() if n not in valid}
    s1.update(label=f"📥 解析完成：成功 {len(valid)}/{len(items)} 个文件",
              state="complete" if valid else "error")
    st.session_state["raw_text"] = valid
    st.session_state["parse_errors"] = failed

    # 解析失败的条目绝不进下游：否则报错文本会被当成“原文”喂给大模型，
    # 下游的“乱码检测”还判它合格（英文报错的可读字符占比很高），整条流水线给出假结论。
    if not valid:
        st.error("所有 PDF 都解析失败，已停止分析。常见原因：扫描件没装 easyocr、文件加密、文件损坏。")
        return
    if failed:
        st.warning(f"有 {len(failed)} 个文件解析失败，已排除在分析之外：" + "、".join(failed.keys()))

    # 第二步：写入本地向量库（缓存）
    retriever = None
    try:
        vs = build_vectorstore(fp, list(valid.items()))
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        st.success(f"📚 向量库就绪：共 {_vector_count(vs)} 个文本块已入库（@st.cache_resource 缓存）")
    except Exception as e:
        st.warning(f"向量库构建失败，跳过 RAG 检索继续分析：{type(e).__name__}: {e}")

    # 第三步：跑 LangGraph
    _run_stream(
        {
            "raw_text": valid,
            "extracted_params": {},
            "param_citations": {},
            "conflict_report": {},
            "risk_level": "unknown",
            "safety_report": "",
            "safety_pass": False,
            "final_report": "",
            "retry_count": 0,
            "hitl": bool(hitl_on),
            "emergency_triggered": False,
        },
        cfg=_cfg(
            retriever,
            st.session_state.get("spec_retriever"),
            str(st.session_state.get("spec_text") or ""),
        ),
    )


# ---------------- 上一轮“点了开始、但需要先换会话 ID”的待执行任务 ----------------
_job = st.session_state.pop("_pending_run", None)
if _job is not None and uploaded_files:
    _run_pipeline(uploaded_files, _job.get("hitl", True))

# ---------------- 主按钮：开始分析 ----------------
if uploaded_files and st.button("🚀 开始分析", type="primary"):
    # 同一个 thread_id 上如果还挂着一条没处理完的人工审核，新分析会跟旧 State 混在一起。
    # 这里自动换一个新会话 ID 重跑，保证每次分析都是干净的一轮。
    if _has_pending_interrupt(_peek_snapshot()):
        st.session_state["_pending_run"] = {"hitl": bool(hitl_enabled)}
        st.session_state["_pending_tid"] = uuid.uuid4().hex
        st.info("当前会话还有一条未处理的人工审核，已自动为本次分析分配新的会话 ID。")
        st.rerun()
    else:
        _run_pipeline(uploaded_files, hitl_enabled)

# ---------------- HITL 人工审核面板（由快照推导，刷新后依然在） ----------------
_snap = _peek_snapshot()
if _has_pending_interrupt(_snap):
    _vals = _snap.values or {}
    st.divider()
    st.subheader("🧑‍⚖️ HITL 人工审核")
    st.info("生成节点已 **interrupt 暂停**。审核后点确认即可继续；刷新页面不影响，进度已在 SqliteSaver 里。")

    c1, c2 = st.columns(2)
    c1.metric("当前风险等级", _vals.get("risk_level", "unknown"))
    c2.metric("已重试次数", f"{_vals.get('retry_count', 0)} / {MAX_RETRIES}")
    with st.expander("📋 待审核的地质参数", expanded=True):
        st.json(_vals.get("extracted_params", {}))
    st.warning(f"安全审查意见：{_vals.get('safety_report', '-')}")
    if _vals.get("conflict_report"):
        _render_conflict(_vals.get("conflict_report"))
    if st.session_state.get("last_error"):
        st.error(f"上一次操作失败：{st.session_state['last_error']}")

    comment = st.text_input("人工审核意见（可选）")
    ba, bb = st.columns(2)
    if ba.button("✅ 确认通过，继续生成", type="primary"):
        _run_stream(Command(resume={"approved": True, "comment": comment}), _cfg())
        st.rerun()
    if bb.button("❌ 否决，终止本次生成"):
        _run_stream(Command(resume={"approved": False, "comment": comment}), _cfg())
        st.rerun()

# ---------------- 结果区（同样由快照推导，不依赖 session 标记） ----------------
_snap = _peek_snapshot()
_vals = (_snap.values or {}) if _snap is not None else {}

if st.session_state.get("last_error") and not _has_pending_interrupt(_snap):
    st.error(f"上一次操作失败：{st.session_state['last_error']}")

if _vals:
    st.divider()
    st.subheader("📊 分析结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("风险等级", _vals.get("risk_level", "unknown"))
    m2.metric("重试次数", f"{_vals.get('retry_count', 0)} / {MAX_RETRIES}")
    m3.metric("安全审查", "通过" if _vals.get("safety_pass") else "未通过")
    m4.metric("熔断状态", "🚨 已熔断" if _vals.get("emergency_triggered") else "正常")

    if _vals.get("emergency_triggered"):
        st.error("🚨 系统已熔断：连续审查失败，请人工介入！")
    if _vals.get("risk_level") == "rejected":
        st.warning("本次方案生成被人工否决。")

    params = _vals.get("extracted_params") or {}
    if isinstance(params, dict) and params.get("_parse_error"):
        st.error(f"参数提取失败：{params.get('_error') or params.get('_raw', '未知')}")

    with st.expander("🔍 提取的地质参数（表格）", expanded=not _vals.get("final_report")):
        if isinstance(params, dict) and params:
            rows = [[k, str(v)] for k, v in params.items() if not str(k).startswith("_")]
            if rows:
                st.table(pd.DataFrame(rows, columns=["参数", "值"]))
            else:
                st.write("（暂无可用参数）")
        else:
            st.write("（暂无参数）")

    # 参数原文溯源：页码级定位，方便人工回原文核对
    cites = _vals.get("param_citations") or {}
    if isinstance(cites, dict) and cites:
        with st.expander(f"📍 参数原文溯源（{len(cites)} 项，已定位到页码）"):
            rows = []
            for k, c in cites.items():
                if not isinstance(c, dict):
                    continue
                rows.append([
                    k, str(c.get("value", "")), str(c.get("file", "")),
                    _clip_page(c.get("page")), str(c.get("snippet", ""))[:160],
                ])
            if rows:
                st.table(pd.DataFrame(
                    rows, columns=["参数", "值", "文件", "页码", "原文原句"],
                ))

    # 强制引用比对结果：违规项逐条打印两处出处
    cr = _vals.get("conflict_report") or {}
    if isinstance(cr, dict) and cr:
        st.markdown("### 🛡️ 强制引用比对（公司规范 × 地质参数）")
        _render_conflict(cr)

    if _vals.get("safety_report"):
        with st.expander("🛡️ 安全审查结论（文字版）"):
            st.write(_vals["safety_report"])

    if _vals.get("final_report"):
        st.markdown("### 📄 最终处置方案")
        st.markdown(_vals["final_report"])

if st.session_state.get("parse_errors"):
    errs = st.session_state["parse_errors"]
    with st.expander(f"⚠️ {len(errs)} 个文件解析失败（未参与分析）"):
        for _name, _text in errs.items():
            st.write(f"**{_name}**：{str(_text)[:500]}")
