# Standard python modules
import os
import sys
import math

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
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Handmade modules
sys.path.append(os.pardir)
from src.Vision import GoogleCloudVision
from utils.utils import MovingAverage

class FacialExpressionAnalysis(object):
  def __init__(self) -> None:
    self.vision = GoogleCloudVision()
    self.interval_in_sec = 1.0

  def remove_appendix(self, file_path: str) -> str:
    file_path = file_path.split('/')[-1]
    return ''.join(file_path.split('.')[:-1])

  def read_frame(self, video_capture, iframe: int, image_path: str) -> bool:
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, iframe)
    ret, frame = video_capture.read()
    if ret: cv2.imwrite(image_path, frame)
    return ret
  
  def convert_results_to_dataframe(self, faces):
    results = pd.DataFrame()
    for face in faces:
      _result = pd.DataFrame({
        'anger': [face.anger_likelihood],
        'joy': [face.joy_likelihood],
        'surprise': [face.surprise_likelihood],
        'roll_angle': [face.roll_angle],
        'pan_angle': [face.pan_angle],
        'tilt_angle': [face.tilt_angle],
        'detection_confidence': [face.detection_confidence],
        'sorrow_likelihood': [face.sorrow_likelihood],
        'under_exposed_likelihood': [face.under_exposed_likelihood],
        'blurred_likelihood': [face.blurred_likelihood],
        'headwear_likelihood': [face.headwear_likelihood],
      })
      results = pd.concat([results, _result], ignore_index=True)
    return results

  def detect_image(self, image_path: str):
    faces = self.vision.detect_faces_localpath(image_path=image_path)
    results = self.convert_results_to_dataframe(faces=faces)
    return results

  def detect_video(self, video_path: str, interval_in_sec: float=1.0):
    if not os.path.isfile(video_path):
      logger.error('No such video file was found.')
      sys.exit(1)

    self.interval_in_sec = interval_in_sec
    video_capture = cv2.VideoCapture(video_path)
    nframes = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = math.ceil(video_capture.get(cv2.CAP_PROP_FPS))

    results = pd.DataFrame()
    # for iframe in range(0, nframes, int(fps * self.interval_in_sec)):
    for iframe in range(0, 1000, int(fps * self.interval_in_sec)):
      image_path = '../deliverables/' + self.remove_appendix(video_path) + '_frame_%d.png' % (iframe)
      if self.read_frame(video_capture=video_capture, iframe=iframe, image_path=image_path):
        _results = self.detect_image(image_path=image_path)
        if not _results.isna().any().any():
          _results['frame'] = iframe
          results = pd.concat([results, _results], ignore_index=True)
        else:
          logger.warning('No face was detected for frame: %4d' % (iframe))
    return results

  def make_plots(self, results: pd.DataFrame, columns: list[str], plot_name: str, moving_average_in_sec: float=1.0) -> None:
    window_size = math.ceil(moving_average_in_sec / self.interval_in_sec)

    plt.figure()

    Xs = results['frame']
    for column in columns:
      Ys = results[column]
      Ys = MovingAverage(array=Ys, window_size=window_size)
      plt.plot(Xs, Ys, label=column)

    plt.title('Expectations')
    plt.xlabel('Frame number')
    plt.ylabel('Likelyhood')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)

  def save_as_csv(self, results: pd.DataFrame, csvfile: str='../deliverables/output.csv'):
    results.to_csv(csvfile, index=False)

  def read_results(self, csvfile: str):
    results = pd.read_csv(csvfile)
    return results

if __name__ == '__main__':
  obj = FacialExpressionAnalysis()
  results = obj.detect_video(video_path='../assets/aho.mp4', interval_in_sec=1.0)
  obj.make_plots(
    results=results,
    columns=['anger', 'joy', 'surprise'],
    plot_name='../deliverables/emotions.png',
    moving_average_in_sec=3.0,
  )
  obj.make_plots(
    results=results,
    columns=['roll_angle', 'pan_angle', 'tilt_angle'],
    plot_name='../deliverables/poses.png',
    moving_average_in_sec=3.0,
  )
  obj.save_as_csv(results=results)