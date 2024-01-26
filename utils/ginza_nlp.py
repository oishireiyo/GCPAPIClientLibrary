# Standard modules
import os
import sys
from typing import Union

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import spacy
from spacy.pipeline import EntityRuler
import ginza

class GiNZANaturalLanguageProcessing(object):
  def __init__(self, model: str='ja_ginza_electra', split_mode: str='C'):
    self.nlp = spacy.load(model)
    ginza.set_split_mode(self.nlp, split_mode)

  # 文境界解析
  def get_sentences(self, text: str) -> list[str]:
    doc = self.nlp(text)
    return doc.sents

  # 文節
  def get_bunsetu_spans(self, text: str) -> list[str]:
    doc = self.nlp(text)
    bunsetu = ginza.bunsetu_spans(doc)
    return bunsetu
  
  def get_bunsetu_phrase_spans(self, text: str) -> list[str]:
    doc = self.nlp(text)
    bunsetu_phrase = ginza.bunsetu_phrase_spans(doc)
    return bunsetu_phrase
  
  def get_bunsetu_dependencies(self, text: str):
    doc = self.nlp(text)
    dependencies = []
    for sent in doc.sents:
      for span in ginza.bunsetu_spans(sent):
        for token in span.lefts:
          dependencies.append((token, span))
    return dependencies

  # 形態素解析
  def print_syntactic_annotation(self, text: str) -> None:
    '''
    tokenの主なプロパティ
    * token.i: トークン番号
    * token.text: テキスト
    * token.lemma_: レンマ
    * token.tag_: 日本語の品詞タグ
    * token.pos_: Universal Dependenciesの品詞タグ
    * token.dep_: 構文従属関係
    * token.head: 構文上の親のトークン
    * token.children: 構文上の子のトークン
    * token.lefts: 構文上の左のトークン
    * token.rights: 構文上の右のトークン
    '''
    annotation = self.nlp(text)
    for sentence in annotation.sents:
      for token in sentence:
        print(
          token.i,
          token.orth_,
          token.lemma_,
          token.norm_,
          token.morph.get("Reading"),
          token.pos_,
          token.morph.get("Inflection"),
          token.tag_,
          token.dep_,
          token.head.i,
        )
      print('EOS')

  def _get_token_dependencies(self, text: str, symbols: Union[list[str], None]=None) -> list[tuple]:
    doc = self.nlp(text)
    dependencies = []
    for sent in doc.sents:
      for token in sent:
        if symbols is None:
          dependencies.append((token, token.head, token.children))
        else:
          if token.dep_ in symbols:
            dependencies.append((token, token.head, token.children))
    return dependencies

  def get_all_token_dependencies(self, text: str) -> list[tuple]:
    dependencies = self._get_token_dependencies(text=text, symbols=None)
    return dependencies

  def get_subject_token_dependencies(self, text: str) -> list[tuple]:
    dependencies = self._get_token_dependencies(text=text, symbols=['nsubj', 'iobj'])
    return dependencies

  # 固有表現抽出
  def print_named_entities(self, text: str) -> None:
    '''
    entの主なプロパティ。
    * ent.text: テキスト
    * ent.label_: ラベル
    * ent.start_char: 開始位置
    * ent.end_char: 終了位置
    '''
    doc = self.nlp(text)
    for ent in doc.ents:
      print(
        ent.text,
        ent.label_,
        ent.start_char,
        ent.end_char,
      )
    print('EOS')

  def add_named_entries(self, rules: list[dict[str, str]]) -> None:
    ruler = self.nlp.add_pipe('entity_ruler')
    ruler.add_patterns(rules)

  def get_named_entties(self, text: str) -> list:
    doc = self.nlp(text)
    return doc.ents

if __name__ == '__main__':
  obj = GiNZANaturalLanguageProcessing()
  subject_list = obj.get_all_token_dependencies(text='この商品はよく効きます。この商品はよく売れます。')
  obj.get_bunsetu_dependencies(text='この商品はよく効きます。この商品はよく売れます。')
  obj.add_named_entries(
    rules=[
      {'label': 'Person', 'pattern': 'サツキ'},
      {'label': 'Person', 'pattern': 'メイ'},
    ]
  )
  obj.print_named_entities(
    text='小学生のサツキと妹のメイは、母の療養のために父と一緒に初夏の頃の農村へ引っ越してくる。'
  )
  obj.print_syntactic_annotation(text='昨日から胃がキリキリと痛い。ただ、熱は無い。')