"""stanford parser"""
import os
import warnings

import nltk
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser

STANFORD_PARSER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "module", "stanford_parser")
STANFORD_PARSER_PATH = os.path.join(STANFORD_PARSER_DIR, "stanford-parser-full-2020-11-17", "stanford-parser.jar")
STANFORD_PARSER_MODEL_PATH = os.path.join(STANFORD_PARSER_DIR, "stanford-corenlp-4.2.0-models-english.jar")


def POSTagAnalysis(text, stanfordparser_path, stanfordparser_model_path):
    """POSタグの分析"""
    # POSタグの分析用
    p = StanfordParser(path_to_jar=stanfordparser_path, path_to_models_jar=stanfordparser_model_path)

    # POSタグの分析(iterator形式で返ってくる)
    out = p.raw_parse(text)
    # outの型をlist_iteratorからlistへ
    out = list(out)

    # Treeを取得 ※テキストは一つと仮定，増えるとout[1]などに格納されるかも
    tree = out[0]

    return tree


def dependenceAnalysis(text, stanfordparser_path, stanfordparser_model_path):
    """係り受け関係の分析"""
    # 係り受け関係の分析用
    dep_parser = StanfordDependencyParser(path_to_jar=stanfordparser_path, path_to_models_jar=stanfordparser_model_path)

    # 係り受け関係分析 (iterator形式で返ってくる)
    out = dep_parser.raw_parse(text)
    # outの型をlist_iteretorからlistへ
    out = list(out)

    # parseを取得 ※テキストは一つと仮定，増えるとout[1]などに格納されるかも
    parse = out[0]

    return parse


def getNodes(parent, np_phrases, pp_phrases):
    """名詞句と前置詞句を再帰的に探索する"""
    previous_np = ""
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == "ROOT":
                print("======== Sentence =========")
                print("Sentence:", " ".join(node.leaves()))
            else:
                if node.label() == "NP":
                    np_phrases.add(" ".join(node.leaves()))
                    previous_np = " ".join(node.leaves())
                if (node.label() == "PP" or node.label() == "ADVP") and previous_np != "":
                    pp_phrases.add(previous_np + " " + " ".join(node.leaves()))

            getNodes(node, np_phrases, pp_phrases)
    return np_phrases, pp_phrases


def getVerbPhrases(parent, vp_phrases):
    """動詞句を再帰的に探索する"""
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == "ROOT":
                print("======== Sentence =========")
                print("Sentence:", " ".join(node.leaves()))
            else:
                if node.label() == "VP":
                    vp_phrases.add(" ".join(node.leaves()))
            getVerbPhrases(node, vp_phrases)
    return vp_phrases


def get_all_np(instruction):
    """すべての名詞句(np)を取得する"""
    tree = POSTagAnalysis(instruction, STANFORD_PARSER_PATH, STANFORD_PARSER_MODEL_PATH)
    np_phrases, pp_phrases = getNodes(tree, set(), set())
    return np_phrases | pp_phrases


def get_all_vp(instruction):
    """すべての動詞句(vp)を取得する"""
    tree = POSTagAnalysis(instruction, STANFORD_PARSER_PATH, STANFORD_PARSER_MODEL_PATH)
    vp_phrases = getVerbPhrases(tree, set())
    return vp_phrases


def get_longest_np(instruction):
    """最も文字数の多い名詞句を取得する"""
    all_np = get_all_np(instruction)
    return max(all_np, key=len)


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    instructions = [
        "This is a pen.",
        "When I was a child, I lived in California for 4 years, and that experience made me love spending time in nature, especially national parks like Yosemite.",
    ]

    for inst in instructions:
        # POSタグの分析（上記で定義した関数）
        tree = POSTagAnalysis(inst, STANFORD_PARSER_PATH, STANFORD_PARSER_MODEL_PATH)
        print("===============")
        print(inst)
        print("-----------------")
        print(tree)
        print()
        print("-----------------")
        # np_phrases, pp_phrases = getNodes(tree, set(), set())
        np_phrases = get_all_np(inst)
        print("--NP--")
        for phrase in np_phrases:
            print(phrase)
        # print('--NP+alpha--')
        # for phrase in pp_phrases:
        #     if phrase not in np_phrases:
        #         print(phrase)
        print("===============")
        vp_phrases = get_all_vp(inst)
        print("--VP--")
        for phrase in vp_phrases:
            print(phrase)
        print()
