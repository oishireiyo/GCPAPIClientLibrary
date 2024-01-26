from google.cloud import vision
from google.oauth2 import service_account

class GoogleCloudVision(object):
  def __init__(self, credential_path: str='../.apikeys/projectkonoha-a15ab1901476.json'):
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    self.client = vision.ImageAnnotatorClient(credentials=credentials)

  # 画像オブジェクトを生成
  def create_image_localpath(self, image_path: str):
    with open(image_path, 'rb') as image_file:
      content = image_file.read()

    image = vision.Image(content=content)
    return image
  
  def create_image_url(self, url: str):
    image = vision.Image()
    image.source.image_url = url
    return image

  # エラーが発生した場合の処理
  def error_handling(self, response):
    if response.error.message:
      raise Exception(
        '{}\n'
        'For more infor on error messages, check:\n'
        'https://cloud.google.com/apis/design/errors'.format(response.error.message)
      )

  # テキスト検出
  def _detect_text_payload(self, image, language_hints: list[str]=['ja']):
    response = self.client.text_detection(
      image=image,
      image_context={'language_hints': language_hints},
    )
    texts = response.text_annotations

    for text in texts:
      print(f'\n{text.description}')

      vertices = [
        f'({vertex.x},{vertex.y})' for vertex in text.bounding_poly.vertices
      ]
      print('bounds: {}'.format(','.join(vertices)))

    self.error_handling(response=response)

  def detect_text_localpath(self, image_path: str, language_hints: list[str]=['ja']):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_text_payload(image=image, language_hints=language_hints)

  def detect_text_url(self, url: str, language_hints: list[str]=['ja']):
    image = self.create_image_url(url=url)
    self._detect_text_payload(image=image, language_hints=language_hints)

  # ドキュメントテキスト検出
  def _detect_document_payload(self, image, language_hints: list[str]=['ja']):
    response = self.client.document_text_detection(
      image=image,
      image_context={'language_hints': language_hints},
    )

    for page in response.full_text_annotation.pages:
      for block in page.blocks:
        print(f'\nBlock confidence: {block.confidence}\n')

        for paragraph in block.paragraphs:
          print('Paragraph confidence: {}'.format(paragraph.confidence))

          for word in paragraph.words:
            word_text = ''.join([symbol.text for symbol in word.symbols])
            print('Word text: {} (confidence: {})'.format(
              word_text, word.confidence
            ))

            for symbol in word.symbols:
              print('\tSymbol: {} (confidence: {})'.format(
                symbol.text, symbol.confidence
              ))

    self.error_handling(response=response)

  def detect_document_localpath(self, image_path: str, language_hints: list[str]=['ja']):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_document_payload(image=image, language_hints=language_hints)

  def detect_document_url(self, url: str, language_hints: list[str]=['ja']):
    image = self.create_image_url(url=url)
    self._detect_document_payload(image=image, language_hints=language_hints)

  # クリップヒントの検出
  def _detect_crop_hints_payload(self, image):
    crop_hists_params = vision.CropHintsParams(aspect_ratios=[1.77])
    image_context = vision.ImageContext(crop_hists_params=crop_hists_params)

    response = self.client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints

    for n, hint in enumerate(hints):
      print(f'\nCrop Hint: {n}')

      vertices = [
        f'({vertex.x},{vertex.y})' for vertex in hint.bounding_poly.vertices
      ]

      print('bounds: {}'.format(','.join(vertices)))

    self.error_handling(response=response)

  def detect_crop_hints_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_crop_hints_payload(image=image)

  def detect_crop_hints_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_crop_hints_payload(image=image)

  # 顔検出
  def _detect_faces_payload(self, image):
    response = self.client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
      "UNKNOWN",
      "VERY_UNLIKELY",
      "UNLIKELY",
      "POSSIBLE",
      "LIKELY",
      "VERY_LIKELY",
    )
    print('Faces:')
    for face in faces:
      print(f'anger: {likelihood_name[face.anger_likelihood]}')
      print(f'joy: {likelihood_name[face.joy_likelihood]}')
      print(f'surprise: {likelihood_name[face.surprise_likelihood]}')

      vertices = [
        f'({vertex.x},{vertex.y})' for vertex in face.bounding_poly.vertices
      ]

      print('face bounds: {}'.format(','.join(vertices)))

    self.error_handling(response=response)

  def detect_faces_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_faces_payload(image=image)

  def detect_faces_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_faces_payload(image=image)

  # 画像プロパティの検出
  def _detect_properties_payload(self, image):
    response = self.client.image_properties(image=image)
    properties = response.image_properties_annotation

    print('Properties:')
    for color in properties.dominant_colors.colors:
      print(f'frac: {color.pixel_fraction}')
      print(f'\tr: {color.color.red}')
      print(f'\tg: {color.color.green}')
      print(f'\tb: {color.color.blue}')
      print(f'\ta: {color.color.alpha}')

    self.error_handling(response=response)

  def detect_properties_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_properties_payload(image=image)

  def detect_properties_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_properties_payload(image=image)

  # ラベル検出
  def _detect_labels_payload(self, image):
    response = self.client.label_detection(image=image)
    labels = response.label_annotations

    print('Lables:')
    for label in labels:
      print(label.description)

    self.error_handling(response=response)

  def detect_labels_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_labels_payload(image=image)

  def detect_labels_url(self, url):
    image = self.create_image_url(url=url)
    self._detect_labels_payload(image=image)

  # ランドマーク検出
  def _detect_landmarks_payload(self, image):
    response = self.client.landmark_detection(image=image)
    landmarks = response.landmark_annotations

    print('Landmarks:')
    for landmark in landmarks:
      print(landmark.description)
      for location in landmark.locations:
        lat_lng = location.lat_lng
        print(f'Latitude: {lat_lng.latitude}')
        print(f'Longitude: {lat_lng.longitude}')

    self.error_handling(response=response)

  def detect_landmarks_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_landmarks_payload(image=image)

  def detect_landmarks_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_landmarks_payload(image=image)

  # ロゴ検出
  def _detect_logos_payload(self, image):
    response = self.client.logo_detection(image=image)
    logos = response.logo_annotations

    print('Logos:')
    for logo in logos:
      print(logo.description)

    self.error_handling(response=response)

  def detect_logos_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_logos_payload(image=image)

  def detect_logos_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_logos_payload(image=image)

  # オブジェクト検出
  def _detect_objects_payload(self, image):
    response = self.client.object_localization(image=image)
    objects = response.localized_object_annotations

    print(f'Number of objects found: {len(objects)}')
    for object in objects:
      print(f'\n{object.name} (confidence: {object.score})')
      print('Normalized bounding polygon vertices:')
      for vertex in object.bounding_poly.normalized_vertices:
        print(f' - ({vertex.x}, {vertex.y})')

    self.error_handling(response=response)

  def detect_objects_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_objects_payload(image=image)

  def detect_objects_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_objects_payload(image=image)

  # 不適切なコンテンツの検出
  def _detect_safe_search_payload(self, image):
    response = self.client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    likelihood_name = (
      "UNKNOWN",
      "VERY_UNLIKELY",
      "UNLIKELY",
      "POSSIBLE",
      "LIKELY",
      "VERY_LIKELY",
    )
    print('Safe search:')
    print(f'adult: {likelihood_name[safe.adult]}')
    print(f'medical: {likelihood_name[safe.medical]}')
    print(f'spoofed: {likelihood_name[safe.spoofed]}')
    print(f'violence: {likelihood_name[safe.violence]}')
    print(f'racy: {likelihood_name[safe.racy]}')

    self.error_handling(response=response)

  def detect_safe_search_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_safe_search_payload(image=image)

  def detect_safe_search_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_safe_search_payload(image=image)

  # ウェブエンティティとページを検出
  def _detect_web_payload(self, image):
    response = self.client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
      for label in annotations.best_guess_labels:
        print(f'\nBest guess label: {label.label}')

    if annotations.pages_with_matching_images:
      print(
        '\n{} Pages with matching images found: '.format(
          len(annotations.pages_with_matching_images)
        )
      )

      for page in annotations.pages_with_matching_images:
        print(f'\n\tPage url : {page.url}')

        if page.full_matching_images:
          print(
            '\t{} Full Matches found: '.format(len(page.full_matching_images))
          )

          for image in page.full_matching_images:
            print(f'\t\tImage url : {image.url}')

        if page.partial_matching_images:
          print(
            '\t{} Partial Matches found: '.format(len(page.partial_matching_images))
          )

          for image in page.partial_matching_images:
            print(f'\t\tImage url: {image.url}')

    if annotations.web_entities:
      print('\n{} Web entities found: '.format(len(annotations.web_entities)))

      for entity in annotations.web_entities:
        print(f'\n\tScore : {entity.score}')
        print(f'\tDescription: {entity.description}')

    if annotations.visually_similar_images:
      print(
        '\n{} visually similar images found:\n'.format(
          len(annotations.visually_similar_images)
        )
      )

      for image in annotations.visually_similar_images:
        print(f'\tImage url : {image.url}')

    self.error_handling(response=response)

  def detect_web_localpath(self, image_path: str):
    image = self.create_image_localpath(image_path=image_path)
    self._detect_web_payload(image=image)

  def detect_web_url(self, url: str):
    image = self.create_image_url(url=url)
    self._detect_web_payload(image=image)

if __name__ == '__main__':
  obj = GoogleCloudVision()
  obj.detect_text_localpath(image_path='../assets/smple.png')