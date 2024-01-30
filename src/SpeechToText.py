import os
import sys

from google.cloud import speech
from google.oauth2 import service_account

class GoogleCloudSpeechToText(object):
  def __init__(self, credential_path: str='../.apikeys/projectkonoha-a15ab1901476.json'):
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    self.client = speech.SpeechClient(credentials=credentials)

  def create_audio_localpath(self, audio_path: str):
    with open(audio_path, 'rb') as audio_file:
      content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    return audio

  def create_config(self, language_code: str='ja', model: str='default'):
    if model in ["command_and_search", "phone_call", "video", "default"]:
      config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        model=model,
      )
      return config
    else:
      print(f'No such model is supported. {model}')
      sys.exit(1)

  def get_transcribe(self, audio_path: str, language_code: str='ja', model: str='default'):
    audio = self.create_audio_localpath(audio_path=audio_path)
    config = self.create_config(language_code=language_code, model=model)

    response = self.client.recognie(audio=audio, config=config)

    for i, result in enumerate(response.results):
      alternative = result.alternatives[0]
      print('-' * 20)
      print(f'First alternative of result {i}')
      print(f'Transcript: {alternative.transcript}')

if __name__ == '__main__':
  obj = GoogleCloudSpeechToText()