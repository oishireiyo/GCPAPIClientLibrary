# Standard modules
import os
import sys
from typing import Union

# Web scraping
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

class SeleniumHTMLParser(object):
  '''
  A web driver is a component or tool that facilitates communication between a web browser and your automation script or program.
  It acts as a bridge, enabling your code to control and interact with the browser programmatically.
  Web drivers are commonly used in the context of browser automation, web scraping, and testing.
  '''
  def __init__(self):
    self.url = None    
    self.driver = webdriver.Chrome(options=self.driver_options())

  def driver_options(self):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--start-maximized')
    return options

  def set_url(self, url: str):
    self.url = url
    self.driver.get(url)

  # id属性で要素を検索する
  def get_element_by_id(self, id: str):
    return self.driver.find_element(By.ID, id)

  def get_elements_by_id(self, id: str):
    return self.driver.find_elements(By.ID, id)

  # name属性で要素を検索する
  def get_element_by_name(self, name: str):
    return self.driver.find_element(By.NAME, name)

  def get_elements_by_name(self, name: str):
    return self.driver.find_elements(By.NAME, name)

  # class属性で要素を検索する
  def get_element_by_class(self, class_: str):
    return self.driver.find_element(By.CLASS_NAME, class_)

  def get_elements_by_class(self, class_: str):
    return self.driver.find_elements(By.CLASS_NAME, class_)

  # tag名で要素を検索する
  def get_element_by_tag(self, tag: str):
    return self.driver.find_element(By.TAG_NAME, tag)

  def get_elements_by_tag(self, tag: str):
    return self.driver.find_elements(By.TAG_NAME, tag)

  # xpathで要素を検索する
  # https://www.octoparse.jp/blog/xpath-introduction
  def get_element_by_xpath(self, xpath: str):
    return self.driver.find_element(By.XPATH, xpath)

  def get_elements_by_xpath(self, xpath: str):
    return self.driver.find_elements(By.XPATH, xpath)

  def get_element_by_xpath_with_relative_tag_attribute(self, relative_tag: str, attr_name: str, attr_value: str):
    return self.driver.find_element(By.XPATH, f'//{relative_tag}[@{attr_name}="{attr_value}"]')

  def get_elements_by_xpath_with_relative_tag_attribute(self, relative_tag: str, attr_name: str, attr_value: str):
    return self.driver.find_elements(By.XPATH, f'//{relative_tag}[@{attr_name}="{attr_value}"]')

  def get_element_by_xpath_with_relative_tag_contains_attribute(self, relative_tag: str, attr_name: str, attr_value: str):
    return self.driver.find_element(By.XPATH, f'//{relative_tag}[contains(@{attr_name},"{attr_value}")]')

  def get_elements_by_xpath_with_relative_tag_contains_attribute(self, relative_tag: str, attr_name: str, attr_value: str):
    return self.driver.find_elements(By.XPATH, f'//{relative_tag}[contains(@{attr_name},"{attr_value}")]')

  def get_element_by_xpath_with_relative_tag_text(self, relative_tag: str, text: str):
    return self.driver.find_element(By.XPATH, f'//{relative_tag}[text()="{text}"]')

  def get_elements_by_xpath_with_relative_tag_text(self, relative_tag: str, text: str):
    return self.driver.find_elements(By.XPATH, f'//{relative_tag}[text()="{text}"]')

  def get_element_by_xpath_with_relative_tag_contains_text(self, relative_tag: str, text: str):
    return self.driver.find_element(By.XPATH, f'//{relative_tag}[contains(text(),"{text}")]')

  def get_elements_by_xpath_with_relative_tag_contains_text(self, relative_tag: str, text: str):
    return self.driver.find_elements(By.XPATH, f'//{relative_tag}[contains(text(),"{text}")]')

  def get_element_by_xpath_with_relative_tag_position(self, relative_tag: str, position: str):
    return self.driver.find_element(By.XPATH, f'//{relative_tag}[position(){position}]')

  def get_elements_by_xpath_with_relative_tag_position(self, relative_tag: str, position: str):
    return self.driver.find_elements(By.XPATH, f'//{relative_tag}[position(){position}]')

  def get_element_by_xpath_except_tag(self, tag: str):
    return self.driver.find_element(By.XPATH, f'//*[not({tag})]')

  def get_elements_by_xpath_except_tag(self, tag: str):
    return self.driver.find_elements(By.XPATH, f'//*[not({tag})]')

  # CSSセレクタで要素を検索する
  def get_element_by_css(self, css: str):
    return self.driver.find_element(By.CSS_SELECTOR, css)

  def get_elements_by_css(self, css: str):
    return self.driver.find_elements(By.CSS_SELECTOR, css)

  # リンクテキストで要素を検索する
  def get_element_by_link(self, link: str):
    return self.driver.find_element(By.LINK_TEXT, link)

  def get_elements_by_link(self, link: str):
    return self.driver.find_elements(By.LINK_TEXT, link)

  # リンクテキストの部分一致で要素を検索する
  def get_element_by_partial_link(self, partial_link: str):
    return self.driver.find_element(By.PARTIAL_LINK_TEXT, partial_link)

  def get_elements_by_partial_link(self, partial_link: str):
    return self.driver.find_elements(By.PARTIAL_LINK_TEXT, partial_link)

  # 要素から特徴量を取得
  def get_text_of_element(self, element):
    return element.text

  def get_tag_of_element(self, element):
    return element.tag_name

  def get_attribute_of_element(self, element, attribute: str):
    return element.get_attribute(attribute)

  def get_is_displayed_of_element(self, element):
    return element.is_displayed()

  def get_is_enabled_of_element(self, element):
    return element.is_enabled()

  def get_location_of_element(self, element):
    return element.location

  def get_size_of_element(self, element):
    return element.size

  # 要素の操作
  def clear_element(self, element) -> None:
    element.clear()

  def click_element(self, element) -> None:
    element.click()

  def submit_element(self, element) -> None:
    element.submit()

  def fill_value_element(self, element, value: str) -> None:
    element.send_keys(value)

  # 全てのテキストを取得
  def get_all_texts(self):
    # script, styleを含む要素を削除する
    elements = self.get_elements_by_xpath(xpath='//*[not(self::script) and not(self::style)]')
    text = ''.join([self.get_text_of_element(element=element) for element in elements])

    # テキストを改行毎にリストに入れ、リスト内の要素の前後の空白を削除
    lines = [line.strip() for line in text.splitlines()]

    # リストの空白要素以外を全て文字列に渡す
    text = '\n'.join(line for line in lines if line)

    return text

  # 全てのテキスト / 位置を取得する
  def get_all_texts_locations_sizes(self):
    contents = []

    # script, styleを含む要素を削除する
    elements = self.get_elements_by_xpath(xpath='//*[not(self::script) and not(self::style)]')
    for element in elements:
      text = self.get_text_of_element(element=element)
      if text:
        location = self.get_location_of_element(element=element)
        size = self.get_size_of_element(element=element)

      contents.append((text, location, size))

    return contents

  # 全てのイメージのみを取得
  def cutout_filename(self, path: str) -> str:
    return os.path.basename(path)

  def cutout_filename_replace_appendix(self, path: str, appendix: str='png') -> str:
    file = os.path.basename(path)
    file_without_appendix = os.path.splitext(file)[0]
    return f'{file_without_appendix}.{appendix}'

  def save_all_images(self, appendix: str='png'):
    elements = self.get_elements_by_tag(tag='img')
    urls = [self.get_attribute_of_element(element=element, attribute='src') for element in elements]
    for url in urls:
      response = requests.get(url)
      with open(f'assets/{self.cutout_filename(path=url)}', 'wb') as file:
        file.write(response.content)

  # ウィンドウサイズを取得する
  def get_window_size(self) -> None:
    self.driver.get_window_size()

  # スクリーンショットを取得する
  def get_screenshot(self, filename: str) -> None:
    self.driver.get_screenshot_as_file(filename)

  # セッションの終了
  def quit(self):
    self.driver.quit()

if __name__ == '__main__':
  parser = SeleniumHTMLParser()
  parser.set_url(url='https://www.selenium.dev/documentation/webdriver/getting_started/first_script/')
  element = parser.get_element_by_tag(tag='p')
  text = parser.get_all_texts()
  print(text)
  parser.save_all_images()
  contents = parser.get_all_texts_locations_sizes()
  print(contents)

  parser.quit()