import json
import re

from bs4 import Comment
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, cast

import numpy as np
from anytree import Node, RenderTree
import bs4
from anytree import PreOrderIter
from anytree.exporter import DotExporter
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
import os
import pathlib


def trim_path(path):
    #  is leaf, remove the tag
    if path["is_leaf"]:
        path["tag"].decompose()
        return
    #  not leaf, remove the text directly under the tag
    else:
        for c in path["tag"].contents:
            if not isinstance(c, bs4.element.Tag):
                # print(c)
                #  remove the text node
                c.extract()

def truncate_input(html, chat_tokenizer, max_context_window=30000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = chat_tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = chat_tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # print(f"html truncated to {max_context_window} tokens")
    return html


def simplify_html(soup, keep_attr=False):
    for script in soup(["script", "style"]):
        script.decompose()
    #  remove all attributes
    if not keep_attr:
        for tag in soup.find_all(True):
            tag.attrs = {}
    #  remove empty tags recursively
    while True:
        removed = False
        for tag in soup.find_all():
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break
    #  remove href attributes
    for tag in soup.find_all("a"):
        del tag["href"]
    #  remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    def concat_text(text):
        text = "".join(text.split("\n"))
        text = "".join(text.split("\t"))
        text = "".join(text.split(" "))
        return text

    # remove all tags with no text
    for tag in soup.find_all():
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    #  if html is not wrapped in a html tag, wrap it

    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


class TEIEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        #  truncate to 1024 tokens approximately
        for i in range(len(texts)):
            text = texts[i]
            words = text.split(" ")
            if len(words) > 1024:
                text = " ".join(words[:1024])
                texts[i] = text
            texts[i] = texts[i].strip()
            if not texts[i]:
                texts[i] = "Some padding text"

        _model_kwargs = self.model_kwargs or {}
        try:
            #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
            responses = self.client.post(
                json={"inputs": texts, **_model_kwargs}, task=self.task
            )
        except Exception as e:
            print(f"error: {e}, texts: {texts}")
        return json.loads(responses.decode())


def nodenamefunc(node):
    return f"{node.name}|{node.prob}|{node.input_ids}"


class TokenDotExporter(DotExporter):
    def __init__(self, node, **kwargs):
        super().__init__(node, **kwargs)

    def __iter__(self):
        # prepare
        indent = " " * self.indent
        nodenamefunc = self.nodenamefunc or self._default_nodenamefunc
        nodeattrfunc = self.nodeattrfunc or self._default_nodeattrfunc
        edgeattrfunc = self.edgeattrfunc or self._default_edgeattrfunc
        edgetypefunc = self.edgetypefunc or self._default_edgetypefunc
        filter_ = self.filter_ or self._default_filter
        return self.__iter(indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_)

    def __iter_nodes(self, indent, nodenamefunc, nodeattrfunc, filter_):
        for node in PreOrderIter(self.node, filter_=filter_, stop=self.stop, maxlevel=self.maxlevel):
            nodename = nodenamefunc(node)
            nodeattr = nodeattrfunc(node)
            nodeattr = " {%s}" % nodeattr if nodeattr is not None else ""
            yield '%s%s' % (DotExporter.esc(nodename), nodeattr)

    def __iter(self, indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_):
        for node in self.__iter_nodes(indent, nodenamefunc, nodeattrfunc, filter_):
            yield node


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


def split_tree(soup: bs4.BeautifulSoup, max_node_words=512) -> List[Tuple[bs4.element.Tag, List[str], bool]]:
    word_count = len(soup.get_text().split())
    if word_count > max_node_words:
        possible_trees = [(soup, [])]
        target_trees = []  # [(tag, path, is_leaf)]
        #  split the entire dom tee into subtrees, until the length of the subtree is less than max_node_words words
        #  find all possible trees
        while True:
            if len(possible_trees) == 0:
                break
            tree = possible_trees.pop(0)
            tag_children = defaultdict(int)
            bare_word_count = 0
            #  count child tags
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    tag_children[child.name] += 1
            _tag_children = {k: 0 for k in tag_children.keys()}

            #  check if the tree can be split
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    #  change child tag with duplicate names
                    if tag_children[child.name] > 1:
                        new_name = f"{child.name}{_tag_children[child.name]}"
                        new_tree = (child, tree[1] + [new_name])
                        _tag_children[child.name] += 1
                        child.name = new_name
                    else:
                        new_tree = (child, tree[1] + [child.name])
                    word_count = len(child.get_text().split())
                    #  add node with more than max_node_words words, and recursion depth is less than 64
                    if word_count > max_node_words and len(new_tree[1]) < 64:
                        possible_trees.append(new_tree)
                    else:
                        target_trees.append((new_tree[0], new_tree[1], True))
                else:
                    bare_word_count += len(str(child).split())

            #  add leaf node
            if len(tag_children) == 0:
                target_trees.append((tree[0], tree[1], True))
            #  add node with more than max_node_words bare words
            elif bare_word_count > max_node_words:
                target_trees.append((tree[0], tree[1], False))
    else:
        soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]
        if len(soup_children) == 1:
            target_trees = [(soup_children[0], [soup_children[0].name], True)]
        else:
            # add an html tag to wrap all children
            new_soup = bs4.BeautifulSoup("", 'html.parser')
            new_tag = new_soup.new_tag("html")
            new_soup.append(new_tag)
            for child in soup_children:
                new_tag.append(child)
            target_trees = [(new_tag, ["html"], True)]
    return target_trees


def trim_html_tree(html, paths, is_leaf, node_tree, chat_tokenizer, node_tokenizer, max_context_window,
                   log_threshold=1e-9):
    #  reconstruct the tree
    root = TokenIdNode(-1, prob=1.0, input_ids=[])
    for nidx, line in enumerate(node_tree):
        node, prob, input_ids = line.split("|")
        node = int(node)
        prob = float(prob)
        input_ids = json.loads(input_ids)
        if node == -1:
            # root node
            pass
        else:
            # token = node_tokenizer.decode([node])
            parent = root
            for tid in input_ids:
                find_child = False
                for c in parent.children:
                    if c.input_ids[-1] == tid:
                        parent = c
                        find_child = True
                        break
                if not find_child:
                    break
            node = TokenIdNode(node, parent=parent, prob=prob, input_ids=input_ids)

    #  sort paths by prob
    path_probs = []
    for pidx in range(len(paths)):
        path = paths[pidx]
        str_path = "<" + "><".join(path) + ">"
        path_tokens = node_tokenizer.encode(str_path, add_special_tokens=False)
        path_token_prob = []
        node = root
        for tid in path_tokens:
            for c in node.children:
                if c.name == tid:
                    node = c
                    # print(f"token id: {tid}, token: {node_tokenizer.decode([tid])}, prob: {node.prob}")
                    path_token_prob.append(node.prob)
                    break
        path_probs.append(path_token_prob)

    average_depth = np.mean([len(p) for p in paths])
    #  node prob is the last token prob that is not 1.0
    node_probs = [np.log(path_probs[i][-1]) if path_probs[i][-1] > log_threshold else np.log(log_threshold) for
                  i in range(len(path_probs))]
    path_log_probs = [np.sum([np.log(p) if p > log_threshold else np.log(log_threshold) for p in path]) / average_depth
                      for path in path_probs]

    paths = [{"path": paths[i], "path_prob": path_log_probs[i], "node_prob": node_probs[i],
              "is_leaf": is_leaf[i]} for i in range(len(paths))]
    #  sort paths by prob

    soup = bs4.BeautifulSoup(html, 'html.parser')
    for idj in range(len(paths)):
        path = paths[idj]["path"]
        tag = soup
        for p in path:
            for child in tag.contents:
                if isinstance(child, bs4.element.Tag):
                    if child.name == p:
                        tag = child
                        break

        paths[idj]["tag"] = tag
        paths[idj]["token_length"] = len(chat_tokenizer.encode(str(tag), add_special_tokens=False))
        paths[idj]["prob"] = paths[idj]["path_prob"]
    #  sort paths by prob
    paths = sorted(paths, key=lambda x: x["prob"], reverse=True)
    total_token_length = sum([p["token_length"] for p in paths])

    #  remove low prob paths
    while total_token_length > max_context_window:
        if len(paths) == 1:
            break
        discarded_path = paths.pop()
        total_token_length -= discarded_path["token_length"]
        trim_path(discarded_path)

    total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
    while total_token_length > max_context_window:
        if len(paths) == 1:
            break
        discarded_path = paths.pop()
        trim_path(discarded_path)
        total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

    if total_token_length > max_context_window:
        # loguru.logger.warning(f"dataset {dataset} sample {idx} cannot be trimmed to {max_context_window} tokens")
        html_trim = truncate_input(simplify_html(soup), chat_tokenizer, max_context_window)
    else:
        html_trim = simplify_html(soup)

    assert len(chat_tokenizer.encode(
        html_trim,
        add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

    return html_trim


def clean_xml(html):
    # remove tags starts with <?xml
    html = re.sub(r"<\?xml.*?>", "", html)
    # remove tags starts with <!DOCTYPE
    html = re.sub(r"<!DOCTYPE.*?>", "", html)
    # remove tags starts with <!DOCTYPE
    html = re.sub(r"<!doctype.*?>", "", html)
    return html


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("body", "Body"),
]


class HTMLSplitter(HTMLSectionSplitter):
    def split_html_by_headers(
            self, html_doc: str
    ) -> List[Dict[str, Optional[str]]]:
        try:
            from bs4 import BeautifulSoup, PageElement  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Unable to import BeautifulSoup/PageElement, \
                    please install with `pip install \
                    bs4`."
            ) from e

        soup = BeautifulSoup(html_doc, "html.parser")
        headers = list(self.headers_to_split_on.keys())
        sections = []

        headers = soup.find_all(["body"] + headers)

        for i, header in enumerate(headers):
            header_element: PageElement = header
            if i == 0:
                current_header = "#TITLE#"
                current_header_tag = "h1"
                section_content: List = []
            else:
                current_header = header_element.text.strip()
                current_header_tag = header_element.name
                section_content = []
            for element in header_element.next_elements:
                if i + 1 < len(headers) and element == headers[i + 1]:
                    break
                if isinstance(element, str):
                    section_content.append(element)
            content = " ".join(section_content).strip()

            if content != "":
                sections.append({
                    "header": current_header,
                    "content": content,
                    "tag_name": current_header_tag,
                })

        return sections

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML file

        Args:
            file: HTML file
        """
        file_content = file.getvalue()
        file_content = self.convert_possible_tags_to_header(file_content)
        sections = self.split_html_by_headers(file_content)

        return [
            Document(
                cast(str, section["content"]),
                metadata={
                    self.headers_to_split_on[
                        str(section["tag_name"])
                    ]: section["header"]
                },
            )
            for section in sections
        ]

    def convert_possible_tags_to_header(self, html_content: str) -> str:
        if self.xslt_path is None:
            return html_content

        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        try:
            tree = etree.fromstring(html_content, parser)
        except:
            open("error_html.html", "w").write(html_content)

        # document transformation for "structure-aware" chunking is handled with xsl.
        # this is needed for htmls files that using different font sizes and layouts
        # check to see if self.xslt_path is a relative path or absolute path
        if not os.path.isabs(self.xslt_path):
            xslt_path = pathlib.Path(__file__).parent / self.xslt_path

        xslt_tree = etree.parse("html4rag/converting_to_header.xslt")
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        return str(result)
