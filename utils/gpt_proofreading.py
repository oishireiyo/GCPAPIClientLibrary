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

sys.path.append('../../OpenAI/src/') # Need to be updated
from TextGeneration import TextGeneration

llm = TextGeneration()
llm.add_message_entry_as_specified_role_with_text_content(
  role='system',
  text='あなたは入力された文章の誤字脱字を修正するアシスタントです。'
)
llm.add_message_entry_as_specified_role_with_text_content(
  role='user',
  text=(
    '今まで掃除当番は1週間ごとの交代制としましたが、出社人数の増加により3日ごとに変更します。当番表は添付のExcelをご覧ください。'
    '当番の方は、冷蔵庫上のホワイトボードに自分の名前を書いてくだしい。当番を後退するときは次に当番をになる方が名前を書いてください。'
    '当番の期間中にテレワークや休業となる日ある場合は、誰かに代わりを頼んでください。'
    '代わりを頼んだ時はほわとボードに誰が代わりに当番をしているかわかるように書いてくdさい。'
  )
)
llmanswers = llm.execute()