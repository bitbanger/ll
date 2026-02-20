import atexit
import os

from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


class Sel:
	def __init__(self, url=None, headless=False, linger=False, download_dir=None):
		self.headless = headless
		self.last_loaded_url = None

		if download_dir is None:
			download_dir = os.path.join(os.environ['HOME'], 'Downloads')

		self.options = Options()
		self.options.set_preference('permissions.default.stylesheet', 2)
		self.options.set_preference('permissions.default.image', 2)
		self.options.set_preference('browser.download.folderList', 2)
		self.options.set_preference('browser.download.manager.showWhenStarting', False)
		self.options.set_preference('browser.download.dir', os.path.abspath(download_dir))
		self.options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')
		if self.headless:
			self.options.add_argument("--headless")

		self.driver = webdriver.Firefox(options=self.options)

		self._closed = False
		if not linger:
			atexit.register(self.close)

		if url is not None:
			self.load(url)


	@staticmethod
	@contextmanager
	def tmp(*a, **kw):
		try:
			sel = Sel(*a, **kw)
			yield sel
		finally:
			if 'linger' not in kw or not kw['linger']:
				sel.close()


	def load(self, url):
		self.driver.get(url)
		self.last_loaded_url = url


	def load_new_window(self, url):
		self.driver.execute_script(f'window.open("{url}","_blank");')


	def xpath(self, tag='*', **kw):
		return f'//{tag}' + ''.join([f'[@{k}="{v}"]' for k, v in kw.items()])


	def el(self, wait=10, **kw):
		return WebDriverWait(self.driver, wait).until(
			EC.presence_of_element_located((By.XPATH, self.xpath(**kw))))


	def click(self, **kw):
		self.el(**kw).click()


	def click_at(self, **kw):
		ActionChains(self.driver).move_to_element_with_offset(self.el(**kw), 5, 5).click().perform()


	def type(self, txt, **kw):
		self.el(**kw).send_keys(txt)


	def select(self, txt, **kw):
		Select(self.el(**kw)).select_by_visible_text(txt)


	def screenshot(self, name='page.png'):
		self.driver.save_screenshot(name)


	def src(self):
		return self.driver.page_source


	def source(self):
		return self.src()


	def close(self):
		if self._closed:
			return
		self.driver.quit()
		self._closed = True
