from google.cloud import language_v1
from google.cloud import language_v2
from google.oauth2 import service_account

class GoogleCloudLanguage_v1(object):
  def __init__(self, credential_path: str='../.apikeys/projectkonoha-a15ab1901476.json'):
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    self.client = language_v1.LanguageServiceClient(credentials=credentials)

  def create_request_body(self, text: str, language_code: str='ja'):
    document_type_in_plane_text = language_v1.Document.Type.PLAIN_TEXT
    document = {
      'content': text,
      'type_': document_type_in_plane_text,
      'language': language_code,
    }
    encoding_type = language_v1.EncodingType.UTF8
    request_body = {'document': document, 'encoding_type': encoding_type}

    return request_body

  def get_syntax(self, text: str, language_code: str='ja'):
    # 指定されたテキストを一連の文とトークン(通常は単語)に分割し、それらのトークンに関する言語情報を提供する。
    response = self.client.analyze_syntax(
      request=self.create_request_body(text=text, language_code=language_code)
    )

    for token in response.tokens:
      # Get the text content of this token. Usually a word or punctuation.
      text = token.text
      print(f'Token text: {text.content}')
      print(f'Location of this token in overall document: {text.begin_offset}')

      # Get the part of speech information for this token.
      part_of_speech = token.part_of_speech
      # Get the tag, e.g. NOUN, ADJ for Adjective, et al.
      print('Part of Speech tag: {}'.format(
        language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
      ))
      # Get the voice, e.g. ACTIVE or PASSIVE
      print('Voice: {}'.format(
        language_v1.PartOfSpeech.Voice(part_of_speech.voice).name
      ))
      # Get the tense, e.g. PAST, FUTURE, PRESENT, et al.
      print('Tense: {}'.format(
        language_v1.PartOfSpeech.Tense(part_of_speech.tense).name
      ))

      # Get the lemma of the token.
      print(f'Lemma: {token.lemma}')
      # Get the dependency tree parse information for this token.
      dependency_edge = token.dependency_edge
      print(f'Head token index: {dependency_edge.head_token_index}')
      print('Label: {}'.format(
        language_v1.DependencyEdge.Label(dependency_edge.Label).name
      ))

    print(f'Language of the text: {response.language}')

  def get_entity_sentiment(self, text: str, language_code: str='ja'):
    # エンティティ分析と感情分析の両方を組み合わせたものであり、テキスト内でエンティティについて表現された感情(ポジティブかネガティブか)の特定を試みること。
    response = self.client.analyze_entity_sentiment(
      request=self.create_request_body(text=text, language_code=language_code)
    )

    for entity in response.entities:
      print(f'Representative name for the entity: {entity.name}')
      # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al.
      print(f'Entity type: {language_v1.Entity.Type(entity.type_).name}')
      # Get the saliebce score associated with the entity in the [0.0, 1.0] range
      print(f'Salience score: {entity.salience}')

      # Get the affregate sentiment expressed for this entity in the provided document.
      sentiment = entity.sentiment
      print(f'Entity sentiment score: {sentiment.score}')
      print(f'Entity sentiment magnitude: {sentiment.magnitude}')

      # For many known entities, the metadata is a Wikipedia URL (wikipedia_url) and Knownledge Graph MID (mid).
      # Some entity types may have additional metadata, e.g. ADDRESS entities may have metadata for the address steet_name, postal_code, et al.
      for metadata_name, metadata_value in entity.metedata.items():
        print(f'{metadata_name} = {metadata_value}')

      # Loop over the mentions of this entity in the input document.
      for mention in entity.mentions:
        print(f'Mention text: {mention.text.content}')
        print('Mention Type: {}'.format(
          language_v1.EntityMention.Type(mention.type_).name
        ))

    print(f'Language of the text: {response.language}')

  def get_classify(self, text: str, language_code: str='ja'):
    # ドキュメントを分析し、ドキュメント内で見つかったテキストに適応されるコンテンツカテゴリのリストを返す。
    content_categories_version = (
      language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
    )
    request_body = self.create_request_body(text=text, language_code=language_code)
    request_body['classification_model_options'] = {
      'v2_model': {'content_categories_version': content_categories_version}
    }
    response = self.client.classify_text(request=request_body)

    for category in response.categories:
      # Get the name of the category representing the document.
      print(f'Category name: {category.name}')
      # Get the confidence. Number representing how certain the classifier is that this category represents the provided text.
      print(f'Confidence: {category.confidence}')

class GoogleCloudLanguage_v2(object):
  def __init__(self, credential_path: str='../.apikeys/projectkonoha-a15ab1901476.json'):
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    self.client = language_v2.LanguageServiceClient(credentials=credentials)

  def create_request_body(self, text, language_code: str='ja'):
    document_type_in_plane_text = language_v2.Document.Type.PLAIN_TEXT
    document = {
      'content': text,
      'type_': document_type_in_plane_text,
      'language_code': language_code,
    }
    encoding_type = language_v2.EncodingType.UTF8
    request_body = {'document': document, 'encoding_type': encoding_type}

    return request_body

  def get_sentiment(self, text: str, language_code: str='ja'):
    # 指定されたテキストを調べ、そのテキストの背景にある感情的な考え方を分析する。
    # 具体的には、執筆者の考え方がポジティブか、ネガティブか、ニュートラルか判断する。
    response = self.client.analyze_sentiment(
      request=self.create_request_body(text=text, language_code=language_code)
    )

    print(f'Document sentimet score: {response.document_sentiment.magnitude}')
    print(f'Document sentiment magnitude: {response.document_sentiment.magnitude}')

    for sentence in response.sentences:
      print(f'Sentence text: {sentence.text.content}')
      print(f'Sentence sentiment score: {sentence.sentiment.score}')
      print(f'Sentence sentiment magnitude: {sentence.sentiment.magnitude}')

    print(f'Language of the text: {response.language_code}')

  def get_entities(self, text: str, language_code: str='ja'):
    # 指定されたテキストに既知のエンティティ(著名人、ランドマークなどの固有名詞)が含まれていないかを調べ、
    # それらのエンティティに関する情報を返す。
    response = self.client.analyze_sentities(
      request=self.create_request_body(text=text, language_code=language_code)
    )

    for entity in response.entities:
      print(f'Representative name for the entry: {entity.name}')

      print(f'Entity type: {language_v2.Entity.Type(entity.type_).name}')

      for metadata_name, metadate_value in entity.metadata.items():
        print(f'{metadata_name}: {metadate_value}')

      for mention in entity.mentions:
        print(f'Mention type: {language_v2.ENtityMention.Type(mention.type_).name}')

        print(f'Probability score: {mention.probability}')

    print(f'Language of the text: {response.language_code}')

if __name__ == '__main__':
  obj = GoogleCloudLanguage_v2()
  obj.get_entities(text='', language_code='ja')