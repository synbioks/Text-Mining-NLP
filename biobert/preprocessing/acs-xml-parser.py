import os
import re
import sys
import pickle
import getopt
import numpy as np
from pprint import pprint
from lxml import etree
from tqdm import tqdm

# input_path = os.path.abspath("../acs-data/unzip-files/")
# output_path = os.path.abspath("../acs-data/processed-files/")

# arguments
input_path = None
output_path = None
persist_section = True

# function to collect matching files and dirs
def collect_files(root, res, pattern="", collect_dirs=True):
    
    # go through all item in the dir
    for item in os.listdir(root):
        
        # process item
        item_path = os.path.join(root, item)
        item_is_dir = os.path.isdir(item_path)
        
        # pull valid file in res
        if re.match(pattern, item_path):
            if not item_is_dir or collect_dirs:
                res.append(item_path)
        
        # recursively collect all files
        if item_is_dir:
            collect_files(item_path, res, pattern, collect_dirs)

# helps to extract text from paragraph
def p_helper(node):
    
    # <p/> does not have text
    if node.text is None:
        return ""
    
    # each paragarph is put into a line
    line_list = [node.text]
    for child in node:

        # get the text inside the child if the tag isn't 
        # named-content and inline-formula
        # and the text following the child
        if not child.tag in ("named-content", "inline-formula"):
            line_list.append(" ".join(child.xpath(".//text()")))
        line_list.append(child.tail)

    # there might be none in line_list
        
    # re dark magic
    line = " ".join(line_list)
    line = line.strip()
    line = line.replace("\n", " ")

    # clean up consecutive spaces
    line = re.sub("\s+", " ", line)

    # fix the space around punctuation
    line = re.sub("\s([.,\):;])", r"\1", line)
    line = re.sub("\(\s", r"(", line)
    line = re.sub("\s*([-/])\s*", r"\1", line)
    return line

def kwd_helper(node):
    
    # return a keyword string
    kwd_tokens = node.xpath(".//text()")
    kwd = " ".join(kwd_tokens).replace("\n", " ").strip()
    kwd = re.sub("\s+", " ", kwd)
    return kwd

# this returns interesting titles
# for example: intro, method, and results
# return None for non interesting titles
def title_helper(node):
    
    # extract text from title node
    title = " ".join(node.xpath(".//text()"))
    title = title.replace("\n", " ")
    title = re.sub("\s+", " ", title)
    title = title.strip()
    title = title.lower()
    
    # categorize title
    res = []
    if "intro" in title:
        res.append("introduction")
    if "result" in title:
        res.append("result")
    if "discuss" in title:
        res.append("discussion")
    if "material" in title:
        res.append("materials")
    if "method" in title or "procedure" in title:
        res.append("method")
    if "summary" in title:
        res.append("summary")
    return res

def extract_body(root):
    
    # we are interested in the text in the body section
    curr_title = []
    text = []
    text_nodes = root.xpath("/article/body//*[self::p or (self::title and not(ancestor::caption))]")
    for text_node in text_nodes:
        
        # handle title
        if text_node.tag == "title":
            tmp_title = title_helper(text_node)
            if (not persist_section or len(tmp_title) > 0):
                curr_title = tmp_title
        
        # handle paragraph
        elif text_node.tag == "p":
            text.append({
                "text": p_helper(text_node),
                "section": curr_title
            })
    return text

def extract_abstract(root):
    
    # get the abstract paragraph
    abstract = []
    abstract_nodes = root.xpath("//abstract/p")
    if abstract_nodes:
        abstract.append(p_helper(abstract_nodes[0]))
    return abstract

def extract_keywords(root):
    
    # get the keywords
    keywords = []
    kwd_nodes = root.xpath("//kwd-group/kwd")
    for kwd_node in kwd_nodes:
        keywords.append(kwd_helper(kwd_node))
    return keywords

if __name__ == "__main__":

    opts, _ = getopt.getopt(sys.argv[1:], "", ["input=", "output=", "persist-sec="])
    for opt, arg in opts:
        if opt == "--input":
            input_path = arg
        elif opt == "--output":
            output_path = arg
        elif opt == "--persist-sec":
            persist_section = arg
    
    # check if the arguments are provided
    if input_path is None:
        print("input_path not provided")
        exit(1)
    if output_path is None:
        print("output_path not provided")
        exit(1)

    # collect all xml files
    xml_paths = []
    collect_files(input_path, xml_paths, pattern=".*\.xml$", collect_dirs=False)
    print(f"total xml files: %d" % len(xml_paths))

    # parse the files
    for xml_path in tqdm(xml_paths):
        
        # print("\nparsing %s" % xml_path)
        
        # get the root of the xml
        root = etree.parse(xml_path).getroot()
        
        # create a dictionary holding the xml data
        xml_data = {
            "keywords": extract_keywords(root),
            "abstract": extract_abstract(root),
            "body": extract_body(root)
        }
        
        # pickle the data
        # name the file to <pub #>.pkl
        pub_num = xml_path.split("/")[-1].split(".")[0]
        with open(os.path.join(output_path, pub_num + ".pkl"), "wb") as f:
            pickle.dump(xml_data, f)
