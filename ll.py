import __main__
import csv as _csv
import getpass
import hashlib
import importlib
import inspect
import io
import json as _json
import keyring
import Levenshtein
import matplotlib.pyplot as plt
import os
import pickle as pkl
import re
import requests
import select
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib

from base64 import b64encode, b64decode
from bs4 import BeautifulSoup as Soup
from collections import defaultdict, namedtuple
from contextlib import contextmanager as _cm
from datetime import datetime, timedelta
from dotenv import find_dotenv, load_dotenv
from functools import wraps
from rich import print as richprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import track as _track
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from term_image.image import from_url
from uuid import uuid4

from sel import Sel

ospj = os.path.join

b64_alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/='
_bytes = bytes
def base64(s, encoding='utf-8', bytes=False):
	if not (isinstance(s, str) or isinstance(s, _bytes)):
		raise Exception("Need str or bytes as input")
	
	if isinstance(s, str):
		b = s.encode(encoding=encoding)
	else:
		b = s
		s = b.decode(encoding=encoding)

	if s == '':
		return ''

	if all(c in b64_alpha for c in s):
		# Input is base64
		ret = b64decode(b)
	else:
		# Input is NOT base64
		ret = b64encode(b)

	return ret if bytes else ret.decode()
b64 = base64

def dd(factory=list):
	return defaultdict(factory)

_olddir = __builtins__['dir']

_isatty = sys.stdout.isatty()
def isatty():
	global _isatty
	return _isatty

_oldprint = __builtins__['print']
def oldprint(*a, **kw):
	global _oldprint
	return _oldprint(*a, **kw)

def print(*a, synt=None, **kw):
	if synt is not None:
		return syntax(*a, synt=synt, **kw)

	if len(a) == 1 and isinstance(a[0], str):
		if (t:=a[0].strip()).startswith('{') and t.endswith('}'):
			try:
				txt = _json.dumps(_json.loads(a[0]), indent=2)
				return richprint(txt)
			except _json.JSONDecodeError as e:
				pass

	if not isatty():
		return oldprint(*a, **kw)

	try:
		_print(*a, **kw)
	except Exception as e:
		if isinstance(e, KeyboardInterrupt):
			raise e
		try:
			richprint(*a, **kw)
		except:
			oldprint(*a, **kw)

_ll_global_console = Console()
def _print(*a, **kw):
	global _ll_global_console

	# from term_image.image import BaseImage
	if any('KittyImage' in str(type(x)) for x in a):
		return oldprint(*a, **kw)
	if any((len(bs:=_bytes(x.strip(), encoding='utf-8'))>0 and bs[0]==27) for x in a):
		return oldprint(*a, **kw)
	if len(a)==1:
		fl = str(a[0]).strip().split('\n')[0].lower()
		if '<!doctype' in fl or '<html' in fl:
			return richprint(Syntax(Soup(a[0], 'html.parser').prettify(), 'html'))

	# richprint(*a, **kw)
	for i, e in enumerate(a):
		if isinstance(str(e)):
			for k, v in {
				'green]': 'dark_sea_green3]',
				'red]': 'light_coral]',
			}.items():
				a[i] = a[i].replace(k, v)
	_ll_global_console.print(*a, **kw)
	
__builtins__['print'] = print # fuggit


os.system('')

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
gray = (225,225,225)
white = (255,255,255)
def cprint(rgb, *args, **kwargs):
	if rgb == white:
		print(*args, **kwargs)
		return
	r, g, b = rgb
	print(f'\033[38;2;{r};{g};{b}m', end='')
	print(end='', *args, **kwargs)
	print('\033[0m', end='\n')

alpha = 'abcdefghijklmnopqrstuvwxyz'
alpha += alpha.upper()

nums = '0123456789'
digits = nums

def sleep(*a, **kw):
	return time.sleep(*a, **kw)

def is_file(fn):
	if not fexists(fn):
		err(f"couldn't find [grey70]{fn}[/grey70]")
	return os.path.isfile(fn)
isfile = is_file

def is_dir(fn):
	if not fexists(fn):
		err(f"couldn't find [grey70]{fn}[/grey70]")
	return os.path.isdir(fn)
isdir = is_dir

def uuid():
	return str(uuid4())

def regf(regex, multiline=True, all=False):
	def _(s):
		if all:
			return re.findall(regex, s, *[re.MULTILINE]*multiline)
		res = re.search(regex, s, *[re.MULTILINE]*multiline)
		if res is None:
			return None
		try:
			return res.group(1)
		except:
			return res.group(0)
	return _

def plot(ys, title='', xlabel='', ylabel=''):
	plt.plot(range(len(ys)), ys)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if title:
		plt.title(title)
	plt.show()

def histo(*vals):
	if len(vals)==1 and isinstance(vals[0], list):
		vals = vals[0]

	plt.hist(vals)
	plt.show()

def flatten(ls):
	f = []
	for l in ls:
		for e in l:
			f.append(e)
	return f

def dedupe_itr(l):
	seen = set()
	for e in l:
		if e in seen:
			continue
		seen.add(e)
		yield e

def dedupe(l):
	r = dedupe_itr(l)
	if type(l) == list:
		r = list(r)
	return r

def rule(title=None, pre_space=1, post_space=1):
	a = ['']*pre_space + [Rule(title=title, style='grey30')] + ['']*post_space
	richprint(*a)

def read(fn, strip=True, b=False):
	fn = fix_path(fn)

	if not os.path.exists(fn):
		return None
	with open(fn, 'r'+'b'*b) as f:
		return f.read().strip() if (strip and not b) else f.read()

def bread(fn):
	return read(fn, strip=False, b=True)

def mkdir(d, exist_ok=True):
	if not d:
		return
	d = fix_path(d)

	os.makedirs(d, exist_ok=exist_ok)

def last_line(fn):
	# Code taken from: https://stackoverflow.com/a/54278929
	with open(fn, 'rb') as f:
		try:  # catch OSError in case of a one line file
			f.seek(-2, os.SEEK_END)
			while f.read(1) != b'\n':
				f.seek(-2, os.SEEK_CUR)
		except OSError:
			f.seek(0)
		last_line = f.readline().decode()

	return last_line

def append(fn, txt, ensure_pre_newline=True, ensure_post_newline=True, require_exist=False):
	if len(fn)>100 and not len(txt)>100:
		txt,fn=fn,txt
	if len(lines(fn))>0 and len(lines(txt))==1:
		txt,fn=fn,txt

	if require_exist and not fexists(fn):
		raise Exception(f'file "{fn}" gotta exist if require_exist=True')

	if (dn:=os.path.dirname(fn)) and not fexists(dn):
		mkdir(dn)

	fn = fix_path(fn)

	if ensure_pre_newline and (not last_line(fn).endswith('\n')) and (not txt.startswith('\n')):
		txt = '\n' + txt
	if ensure_post_newline and (not txt.endswith('\n')):
		txt = txt + '\n'

	mode = 'a' if require_exist else 'a+'
	with open(fn, mode) as f:
		f.write(txt)

def touch(fn):
	if not fexists(fn):
		if (dn:=os.path.dirname(fn)) and dn!='.':
			os.makedirs(os.path.dirname(fn), exist_ok=True)
		with open(fn, 'w+') as f:
			f.write('')

def write(fn, txt, ensure_newline=True, create_dirs=True):
	if len(fn)>100 and not len(txt)>100:
		txt,fn=fn,txt
	if len(lines(fn))>0 and len(lines(txt))==1:
		txt,fn=fn,txt

	if (dn:=os.path.dirname(fn)) and not fexists(dn):
		mkdir(dn)

	fn = fix_path(fn)

	if ensure_newline and (len(txt) == 0 or txt[-1] != '\n'):
		txt += '\n'
	with open(fn, 'w+') as f:
		f.write(txt)

def split(s, delim='', empties=True):
	st = s.strip()
	spl = st.split(delim) if delim else st.split()
	return [l.strip() for l in spl if l or empties]

class lineslice:
	def __init__(self, s):
		self.s = str(s)
		self.l = lines(self.s)

	def __getitem__(self, idx):
		if type(idx)==slice:
			return '\n'.join(self.l[idx])
		else:
			return self.l[idx]

def head(x, n, invert=False):
	if type(x) == list:
		x = [str(e) for e in x]
	else:
		x = lines(str(x))

	if invert:
		return '\n'.join(x[n:])
	else:
		return '\n'.join(x[:n])

def tail(x, n, invert=False):
	if type(x) == list:
		x = [str(e) for e in x]
	else:
		x = lines(str(x))

	if invert:
		return '\n'.join(x[-n:])
	else:
		return '\n'.join(x[:-n])


def lines(s, intuit_file=True, stream=False, strip=True):
	if type(s) != str:
		s = str(s)

	def _itr(base):
		for line in base:
			if (not strip) or (line:=line.strip()):
				yield line

	if intuit_file and os.path.exists(s):
		with open(s, 'r') as f:
			it = _itr(f.readlines())
	else:
		it = _itr((s.strip() if strip else s).split('\n'))

	return it if stream else list(it)


def stream_lines(s, intuit_file=True, strip=True):
	return lines(
		s,
		intuit_file=intuit_file,
		stream=True,
		strip=strip,
	)


def join(l, delim=' '):
	return delim.join([str(x) for x in l])

def ljoin(l):
	return join(l, delim='\n')

def gen2lst(gen):
	ret = []
	for e in gen:
		ret.append(e)
	return ret

def gen2str(gen):
	ret = ''
	for e in gen:
		ret += e
	return ret

def cull(lst):
	return [e for e in lst if (type(e) == bool or e)]

def only_nums(s, also=[]):
	return ''.join([c for c in s if c in nums or c in also])

def only_alpha(s, also=[]):
	return ''.join([c for c in s if c in alpha or c in also])

def alphanums(s, also=[]):
	return ''.join([c for c in s if c in alpha or c in nums or c in also])

def nth(n):
	def _nth(itr):
		if hasattr(itr, '__getitem__'):
			return itr[n]
		else:
			for i, e in enumerate(itr):
				if i == n:
					return e
	return _nth

def md5(s, encoding='utf-8', b=False):
	h = hashlib.md5(s.encode(encoding)).hexdigest()
	return int(h, 16) if b else h

def fmd5(s, encoding='utf-8'):
	return hashlib.md5(bread(s)).hexdigest()

def detect_single_csv_row(x):
	if isinstance(x, dict):
		return render_csv([x], no_headers=True)
	if isinstance(x, list) and len(x)>0 and isinstance(x[0], dict):
		return render_csv(x)
	if any(isinstance(x, y) for y in (list, tuple, set, type({}.keys()))):
		return render_csv(list(x), no_headers=True)

	return None

def csv_row(row):
	if (rv:=detect_single_csv_row(row)) is not None:
		return rv

	return next(_csv.reader(io.StringIO(row)))

def render_csv(row_dicts, no_headers=False):
	assert(len(row_dicts) > 0)

	if isinstance(row_dicts, dict):
		row_dicts = [row_dicts]
	if any(not isinstance(y, dict) for y in row_dicts):
		buf = ''
		for i, x in enumerate(row_dicts):
			if i>0:
				buf += ','
			if ',' in str(x):
				buf += f'"{x}"'
			else:
				buf += str(x)
		return buf

	def _render_field(x):
		x = str(x)
		if ',' in x:
			return '"' + x.replace('"', '""') + '"'
		else:
			return x

	buf = ''
	if not no_headers:
		buf += ','.join(map(_render_field, list(row_dicts[0].keys())))
	for i, row in enumerate(row_dicts):
		if (hasattr(row_dicts, '__len__') and len(row_dicts)>=1) and (i==0 or not no_headers):
			buf += '\n'
		buf += ','.join(map(_render_field, list(row.values())))

	return buf

def csv(fn, delim=None, convert=True, empty='', stream=False, **kwargs):
	assert(not ('dicts' in kwargs and 'header' in kwargs))
	header = True
	if 'dicts' in kwargs:
		header = kwargs['dicts']
	elif 'header' in kwargs:
		header = kwargs['header']


	if (rv:=detect_single_csv_row(fn)) is not None:
		return rv

	fn = fix_path(fn)

	# if len(fn) > 100 and not os.path.exists(fn):
	if not os.path.exists(fn) and (delim or ',') in fn:
		write(fn, fn:=f'/tmp/{uuid()}')
		if delim is None:
			delim = ','
	if delim is None:
		with open(fn, 'r') as f:
			try:
				delim = _csv.Sniffer().sniff(f.read(1024), delimiters=',|\t').delimiter
			except Exception as e:
				if 'Could not determine delimiter' in str(e):
					delim = ','

	if wc_l(fn) == 1: # TODO: faster wc_l in general
		return next(_csv.reader(lines(fn), delimiter=delim))

	def _itr():
		with open(fn, 'r') as f:
			r = _csv.reader(f, delimiter=delim)

			if header:
				cols = next(r)

			rows = []
			for row in r:
				if convert:
					nr = []
					for e in row:
						try:
							if (m:=re.findall('^([0-9]+)$', e)) and m[0]==e:
								nr.append(int(e))
							else:
								if int(e)==float(e):
									nr.append(int(e))
								else:
									nr.append(float(e))
						except ValueError:
							nr.append(e)
					row = nr
				row = [(e if e!='' else empty) for e in row]

				if header:
					yield {cols[i]: row[i] for i in range(len(row))}
				else:
					yield row


	return list(_itr()) if not stream else _itr()
	

	
	

def items(dct):
	return [x for x in dct.items()]

def kv(dct):
	return [x for x in dct.keys()], [x for x in dct.values()]

def pickle(fn, obj):
	if type(obj) == str:
		if (type(fn)!=str) or (len(fn)-len(obj)>200):
			obj,fn = fn,obj

	fn = fix_path(fn)

	with open(fn, 'wb+') as f:
		pkl.dump(obj, f)

def unpickle(fn):
	with open(fn, 'rb') as f:
		return pkl.load(f)


def cache_key(f, args, kwargs):
	return f.__qualname__ + md5(' '.join([
		str('.'.join([f.__module__, f.__qualname__])),
		' '.join([str(a) for a in args]),
		' '.join([f'{k}={v}' for k, v in kwargs.items()]),
	]))


here_cache_cache = dict()
def cache(f):
	@wraps(f)
	def wrapper(*args, cache_base=None, **kwargs):
		global here_cache_cache
		if cache_base is None:
			if f.__qualname__ not in here_cache_cache:
				here_cache_cache[f.__qualname__] = here(up=1) # this was 2 at one point & didn't work in repl or file. now it's 1 & works in both. but just be aware ig
			cache_base = here_cache_cache[f.__qualname__]
		for a in [*args]+list(kwargs.values()):
			if ' object at ' in str(a):
				print(f"Warning: can't hash argument \"{str(a)}\" to cache <{str(f)}> call")
				return f(*args, **kwargs)

		key = cache_key(f, args, kwargs)
		path = os.path.join(cache_base, f'cache/{key}')
		if os.path.exists(path):
			return unpickle(path)
		else:
			cdir = os.path.join(cache_base, 'cache')

			# Actually calculate the result
			res = f(*args, **kwargs)

			os.makedirs(cdir, exist_ok=True)
			pickle(path, res)
			return res

	return wrapper


def wc_l(fn, empties=True):
	# TODO: faster
	fn = fix_path(fn)

	count = 0
	with open(fn, 'r') as f:
		for l in f.readlines():
			if (not empties) or l.strip():
				count += 1
	return count


def keys(d):
	return list(d.keys())

def vals(d):
	return list(d.values())

def items(d):
	return list(d.items())

def yn(msg, default_yes=False):
	global _ll_global_console

	ynstr = '([dark_sea_green3]y[/dark_sea_green3]/[light_coral]N[/light_coral])'
	if default_yes:
		ynstr = ynstr.replace('y','Y').replace('n','N')
	try:
		resp = _ll_global_console.input(f'{msg} {ynstr}: ').strip().lower()
	except KeyboardInterrupt:
		print('')
		quit(1)
	if not resp:
		return default_yes
	return resp in ['yes', 'y']

def num(s, discard=False):
	if type(s) in (int, float):
		return s
	elif type(s) == str:
		try:
			return int(s)
		except:
			try:
				return float(s)
			except:
				pass
		return s
	elif type(s) == list:
		return [num(e) for e in s if type(e) in (int, float) or (not discard)]
	elif type(s) == tuple:
		return tuple(num(list(s)))

def resplit(pat, s, intuit_f=True, multiline=False):
	# Try to determine which is the
	# pattern and which is the string,
	# and also load files as strings
	if intuit_f:
		p_ex = os.path.exists(pat)
		s_ex = os.path.exists(s)
		if p_ex != s_ex:
			if p_ex and not s_ex:
				pat,s=s,pat
			s = read(s)
		elif len(pat)+len(s)>10 and len(s)<len(pat):
			pat,s = s,pat
		elif len(set(s).intersection(set('[]^*?.'))) >= 3:
			pat,s=s,pat

	if multiline:
		return re.split(pat, s, re.MULTILINE)
	else:
		return re.split(pat, s)

def map(f, x):
	if callable(x) and not callable(f):
		f,x=x,f

	# We're gonna do our best. If either string
	# represents a callable method of the other
	# object, we'll do that, 
	if not callable(x) and not callable(f):
		if type(x)==str or type(f)==str:
			if type(x)==type(f)==str: # Both are strings
				try:
					if callable(getattr(x, f)): # Try x.f first
						return map(getattr(x, f), x)
				except AttributeError:
					pass
				if callable(getattr(f, x)): # Then try f.x
					return map(getattr(f, x), x)
			else: # One is a string
				if type(x)==str:
					f,x=x,f
				if type(x) in (list,tuple):
					# We have a string "getter" and a list of stuff

					if len(x) == 0:
						return [] if type(x)==list else tuple()

					# Originally, I had the getattr result being *called*
					# here. idk why. wouldn't we just wanna get it?
					# I'm sure something will break later...
					# n = [getattr(e, f)() for e in x]
					def _flex_get(e, f):
						if isinstance(e, dict):
							return e[f]
						else:
							return getattr(e, f)
					n = [_flex_get(e, f) for e in x]
					if type(x) == tuple:
						n = tuple(n)
					return n
				else:
					if callable(getattr(x, f)):
						return map(getattr(x, f), x)

	if type(x) == list:
		return [map(f, e) for e in x]
	elif type(x) == tuple:
		return tuple([map(f, e) for e in x])
	else:
		return f(x)

@_cm
def attempt(handler=lambda e:0, warn=False):
	if warn:
		handler = lambda e: richprint(f'[light_coral]{str(e)}[/light_coral]')
	try:
		yield
	except Exception as e:
		handler(e)

def dbg(a):
	print(a)
	return a

def post(*a, **kw):
	if 'method' in kw:
		del kw['method']
	return html(*a, method=requests.post, **kw)

def html(*a, tries=1, **kw):
	for _ in range(tries):
		try:
			if (resp:=_html(*a, **kw)) is None:
				time.sleep(2)
				continue
			return resp
		except:
			time.sleep(2)

def _html(url, fake_user=False, b=False, method=requests.get, **kwargs):
	if fake_user:
		kwargs.update({
			'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
			'accept-language': 'en-US,en;q=0.9',
			'cache-control': 'max-age=0',
			'priority': 'u=0, i',
			'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
			'sec-ch-ua-mobile': '?0',
			'sec-ch-ua-platform': '"macOS"',
			'sec-fetch-dest': 'document',
			'sec-fetch-mode': 'navigate',
			'sec-fetch-site': 'none',
			'sec-fetch-user': '?1',
			'upgrade-insecure-requests': '1',
			'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
		})
	if 'payload' in kwargs and 'json' not in kwargs:
		kwargs['json'] = kwargs['payload']
		del kwargs['payload']

	try:
		res = method(url, **kwargs)
		if b:
			return res.content
		else:
			return res.text
	except requests.exceptions.MissingSchema:
		try:
			res = method('https://'+url, **kwargs)
			return res.content if b else res.text
		except:
			res = method('http://'+url, **kwargs)
			return res.content if b else res.text
http = html
webpage = html
web = html
site = html
page = html
url = html

def soup(url, **kwargs):
	return Soup(html(url, **kwargs), 'html.parser')


def json(url):
	if isinstance(url, dict):
		return _json.dumps(url)
	else:
		if (url:=url.strip()).startswith('{'):
			txt = url
		elif os.path.exists(url):
			txt = read(url)
		else:
			try:
				txt = html(url)
			except requests.exceptions.ConnectionError:
				raise Exception(f"idk how to interpret '{url}' as a JSON file, sorry")

		return _json.loads(txt)


def here(p='', up=0, abs=True):
	assert(up>=0)
	h = os.path.dirname(inspect.stack()[1+up].filename)
	res = os.path.join(h, p) if p else h
	if abs:
		res = os.path.abspath(res)
	return res

def import_py(path, up=0):
	assert(path.endswith('.py'))

	if not path.startswith('/'):
		path = here(path, up=up+1)
	
	return importlib.machinery.SourceFileLoader(os.path.basename(path)[:-3], path).load_module()

def main_file():
	try:
		return __main__.__file__
	except AttributeError:
		return None

def is_repl():
	return main_file() is None

def import_from(path, *syms, **aliases):
	if len(syms)==1 and type(syms[0])==list:
		syms = syms[0]
	if len(syms)==1 and type(syms[0])==str and fexists(syms[0]) and not fexists(path):
		path, syms[0] = syms[0], path

	smod = import_py(path, up=1)
	dmod = sys.modules[inspect.currentframe().f_back.f_globals['__name__']]
	for a in _olddir(smod):
		if a in aliases:
			setattr(dmod, aliases[a], getattr(smod, a))
			continue
		if syms and a not in syms:
			continue
		if (not syms) and (a.startswith('__') or a.endswith('__')):
			continue
		setattr(dmod, a, getattr(smod, a))

def import_all(path):
	return import_from(path, syms=[])

def count(l):
	d = defaultdict(int)
	for e in l:
		d[e] += 1
	return d
def counts(l):
	return count(l)

def run(cmd, stderr=False):
	out, err = subprocess.Popen(
		['bash', '-c', cmd],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	).communicate()

	out, err = out.decode(), err.decode()

	return (out, err) if stderr else out

def abs(path):
	return os.path.abspath(str(path))

def fix_path(path):
	if path.startswith('~'):
		path = path.replace('~', os.environ.get('HOME'), 1)

	return path

def ls(path='.', abs=False, rel=False, t=True):
	if abs and rel:
		print('\n[grey50]ll.ls got both abs and rel; choosing abs[grey50]\n')

	if path.startswith('~'):
		path = path.replace('~', os.environ.get('HOME'), 1)

	if abs:
		path = os.path.abspath(path)
	elif rel:
		path = os.path.relpath(path, os.path.abspath(os.path.dirname(
			inspect.getframeinfo(inspect.stack()[1][0]).filename)))

	files = os.listdir(path)
	if abs or rel:
		files = [os.path.join(path, f) for f in files]

	if t:
		files = sorted(files, key=lambda f: os.path.getctime(os.path.join(path, f)))

	return files

bn = basename = os.path.basename
dn = dirname = os.path.dirname

def dot(k):
	return lambda x: x[k] if isinstance(x, dict) else getattr(x, k)

def dot_eq(k, v):
	return lambda x: v == (x[k] if isinstance(x, dict) else getattr(x, k))

def fexists(p):
	if p.startswith('~'):
		p = p.replace('~', os.environ['HOME'], 1)
	return os.path.exists(p)

def pjoin(*p):
	p = os.path.join(*p)
	if p.startswith('~'):
		p = p.replace('~', os.environ['HOME'], 1)
	return p

def exists_here(p):
	return fexists(pjoin(here(up=1), p))

def secure_set(k, v, svc='ll', warn=True):
	if v is None:
		raise Exception("Don't set None; it'd be annoying to write sentinel logic to accommodate that")

	fmt = f'[wheat1]{k}[/wheat1]@[bright_blue]{svc}[/bright_blue]'

	if (oldpw:=keyring.get_password(svc, k)) is not None and oldpw != v:
		if warn and not yn(f'Overwrite password for {fmt}?'):
			print(f'[bold light_coral]ll.secure_set:[/bold light_coral] aborting')
			return
	keyring.set_password(svc, k, v)

def secure_get(k, svc='ll', prompt=True):
	fmt = f'[wheat1]{k}[/wheat1]@[bright_blue]{svc}[/bright_blue]'

	def _getpw():
		print(f'\nEnter new value for {fmt}: ', end='')
		pw1 = getpass.getpass('').strip()
		print(f'\nEnter it again: ', end='')
		pw2 = getpass.getpass('').strip()
		print('')

		return pw1, pw2

	pw = keyring.get_password(svc, k)
	if pw is None and prompt:
		pw1, pw2 = _getpw()
		while pw1 != pw2:
			print(f"[light_coral]Error:[/light_coral] passwords weren't the same!\n")
			if not yn('Try again?'):
				return
			pw1, pw2 = _getpw()
		keyring.set_password(svc, k, pw1)
	pw = keyring.get_password(svc, k)

	return pw

def andify(l, quote='', oxford=True):
	l = list(l)

	ss = [f'{quote}{e}{quote}"' for e in l]

	match len(ss):
		case 0:
			return ''
		case 1:
			return ss[0]
		case 2:
			return f'{ss[0]} and {ss[1]}'
		case _:
			return \
				', '.join(ss[:-1])	+ \
				','*oxford					+ \
				f' and {ss[-1]}'
			
	return ', '.join(ss[:-1]) 

def only_digits(s):
	return ''.join(c for c in s if c in '0123456789')

def track(i, total=None, title=None):
	kwargs = {}
	if total is not None:
		kwargs['total'] = total
	if title is not None:
		kwargs['title'] = title
	
	return _track(i, total=total, description=title)


class gen:
	def __init__(self, it):
		self._it = it
		self._buf = []


	def __getitem__(self, idx):
		# TODO: slices? although, negative idcs would
		# require filling the buf, so...

		if len(self._buf) < idx:
			for _ in range((idx+1)-len(self._buf)):
				self._buf.append(next(self._it))

		return self._buf[idx]


def filter(f, l, stream=False):
	if callable(l) and not callable(f):
		f,l=l,f

	def _it():
		for e in l:
			if f(e):
				yield e

	if stream:
		return _it()
	elif isinstance(l, str):
		return ''.join(l)
	else:
		for t in (tuple, set):
			if isinstance(l, t):
				return t(_it())
		# Finally:
		return list(_it())


def first(f, l):
	for e in filter(f, l, stream=True):
		return e

def agg(l, key=lambda x: x, val=lambda x: x):
	# TODO: does this swap logic make sense?
	if callable(l) and not callable(key):
		l, key = key, l
	if callable(l) and not callable(val):
		l, val = val, l

	dd = defaultdict(list)

	for e in l:
		dd[key(e)].append(val(e))

	return dd


def mtime(fn):
	return datetime.utcfromtimestamp(os.stat(fix_path(fn)).st_mtime)


def dt(x):
	if isinstance(x, int) or isinstance(x, float):
		return dt(datetime.utcfromtimestamp(x))

	if isinstance(x, str):
		return datetime.strptime(x, '%Y-%m-%d')
	else:
		return x.strftime('%Y-%m-%d')


_ll_global_dotenv_found = False
def env(var, loc=None, refresh_global=False, crash=False):
	assert(loc is None or refresh_global is False)

	global _ll_global_dotenv_found

	if loc is not None and loc.startswith('~'):
		loc.replace('~', os.environ['HOME'])

	if loc is not None:
		assert(fexists(loc))
		load_dotenv(loc)
	elif refresh_global or not _ll_global_dotenv_found:
		if (not os.path.exists(de:=os.path.join(os.path.dirname(main_file()), '.env'))) or is_repl():
			de = find_dotenv()
		load_dotenv(de)
		_ll_global_dotenv_found = True

	return os.environ[var] if crash else os.environ.get(var)

def lower(s):
	return s.lower()

def upper(s):
	return s.upper()

def img(url):
	return from_url(url)

def isa(t):
	return lambda x: isinstance(x, t)

is_a = isa

def cat(*ls):
	buf = []
	for l in ls:
		buf.extend(l)
	return buf

def error(*a, kill=True, kill_9=False, **kw):
	print('')
	if (pn:=os.path.basename(sys.argv[0])):
		print(f'[bold light_coral]{pn}:[/bold light_coral] error: ', end='')
	else:
		print('[bold light_coral]error:[/bold light_coral] ', end='')
	print(*a, **kw)
	print('')
	if kill or kill_9:
		if kill_9:
			os._exit(1)
		else:
			sys.exit(1)

def err(*a, **kw):
	return error(*a, **kw)

def pwd():
	return os.getcwd()

def lzip(l):
	return [(l[i], l[i+1]) for i in range(len(l)-1)]

def has_stdin(timeout=1):
	return bool(select.select([sys.stdin], [], [], timeout)[0])

def stdin(lines=False):
	return sys.stdin.readlines() if lines else sys.stdin.read()

def swap(a, b, or_conds):
	return (b, a) if any(or_conds) else (a, b)

def _syntax(txt, synt):
	txt, synt = swap(txt, synt, [
		'\n' in synt and '\n' not in txt,
		len(synt)>20 and len(txt)<=20
	])
	if '\n' in synt and '\n' not in txt:
		synt, txt = txt, synt

	if synt == 'json':
		txt = _json.dumps(_json.loads(txt), indent=2)

	return richprint(Syntax(txt, synt))

def syntax(txt, synt):
	try:
		return _syntax(txt, synt)
	except:
		print(txt)

def synt(*a, **kw):
	return syntax(*a, **kw)

def input(file=False, join_args=True, preferred_arg=1):
	if has_stdin():
		return stdin()
	elif len(sys.argv) <= 1:
		err('no input is available, either on stdin or as a filename CL arg')
	else:
		try:
			fn = sys.argv[preferred_arg]
		except:
			fn = sys.argv[1]

		if file:
			if not fexists(fn):
				err(f"file read specifically requested, but can't find file [grey70]{fn}[/grey70]")

			return read(fn)

		else:
			if len(sys.argv) == 2 and fexists(fn:=sys.argv[1]):
				warn(f"file reading mode isn't on, but you passed the single file [grey70]{fn}[/grey70], so we're reading it anyway")

			if join_args:
				return ' '.join(sys.argv[1:])
			else:
				try:
					return sys.argv[preferred_arg]
				except:
					return sys.argv[1]

def cl_input(*a, **kw):
	if 'file' in kw:
		kw['file'] = False
	return input(*a, **kw)

def cli_input(*a, **kw):
	if 'file' in kw:
		kw['file'] = False
	return cl_input(*a, **kw)

class sentinel:
	pass

def sent(x):
	return isinstance(x, sentinel)


def ass(cond, err_msg=sentinel()):
	cond, err_msg = swap(cond, err_msg, [
		isinstance(cond, str) and
		not isinstance(err_msg, str) and
		not sent(err_msg)
	])

	if not cond:
		if not sent(err_msg):
			err(err_msg)
		else:
			raise Exception('Assertion failed')


def uniq_fn(fn):
	cursor = fn
	while fexists(cursor):
		cursor = f'real_{cursor}'

	return cursor


def mv(fn1, fn2, force=False, ignore=False):
	assert(not (force and ignore))
	# ass(is_file(fn1), err_msg=f"[grey70]{fn1}[/grey70] is not a file")
	if not fexists(fn1):
		err(f"file [grey70]{fn1}[/grey70] not found")
	if is_dir(fn1):
		err(f"file [grey70]{fn1}[/grey70] is a directory")

	if fexists(fn2) and not force:
		if ignore:
			return
		err(f"need to pass [grey70]force=[/grey70][green]True[/green] to overwrite file [grey70]{fn2}[/grey70]")

	return shutil.move(fn1, fn2)


def cp(fn1, fn2, force=False):
	ass(is_file(fn1), err_msg=f"[grey70]{fn1}[/grey70] is not a file")
	if not force and fexists(fn2):
		err(f"[grey70]{fn2}[/grey70] already exists; try calling with [grey70]force=True[/grey70] if you don't care")

	mkdir(dirname(fn2))

	return shutil.copy2(fn1, fn2)

def copy(*a, **kw):
	return cp(*a, **kw)

def escape(txt):
	return urllib.parse.quote(txt)

def unescape(txt):
	return urllib.parse.unquote(txt)

def lev(s1, s2):
	return Levenshtein.distance(s1, s2)

def strip(s):
	return s.strip()

def freqs(l):
	fs = defaultdict(int)
	for e in l:
		fs[e] += 1
	return fs
freq = freqs


def thread(lam, *a, daemon=True, join=False, **kw):
	t = threading.Thread(target=lam, daemon=daemon, args=a, kwargs=kw)
	t.start()
	if join:
		return t.join()
	else:
		return None


def makedirs(*a, **kw):
	return os.makedirs(*a, **kw)


@_cm
def tmp_dir(name=None, persist=False):
	if name is not None:
		dst = ospj('/tmp', name)
		if fexists(dst):
			if isdir(dst):
				raise Exception(f"directory {dst} already exists")
			else:
				# Unnecessary scolding
				raise Exception(f"{dst} already exists, and it's a file, not even a directory")
	else:
		dst = ospj('/tmp', uuid())

	os.makedirs(dst)

	yield dst

	if not persist:
		shutil.rmtree(tmp_dir)
tmpdir = tmp_dir


def add_newline_if_missing(fn):
	if not fexists(fn):
		raise Exception(f"Target {fn} does not exist")
	if last_line(fn)[-1] != '\n':
		with open(fn, 'a') as f:
			f.write('\n')


def sel_dl(url, dst_dir=None, dst_name=None, b=False, clobber=False, ignore=False, ensure_newline=True, tries=10, wait=1):
	# Input checks
	if clobber and ignore:
		raise Exception(f"You can't pass both clobber=True and ignore=True, bro")
	if (dst_dir is not None) and (dst_name is not None):
		_dst = ospj(dst_dir, dst_name)
		if b:
			raise Exception(f"You asked for download destination {_dst}, but you also passed b=True, which means you want the file in-memory in bytes. So which is it?")
		if fexists(_dst):
			if not clobber:
				if ignore:
					return
				else:
					raise Exception(f"Destination name {_dst} already exists" + ' (and is a directory, for that matter)'*isdir(dst) + "; pass clobber=True if you're OK with that, or ignore=True to do nothing instead")
			elif clobber and isdir(_dst):
				raise Exception(f"Destination name {_dst} already exists as a directory")
	elif (dst_dir is not None) and b:
		raise Exception(f"You asked for download destination directory {dst}, but you also passed b=True, which means you want the file in-memory in bytes. So which is it?")
	elif (dst_name is not None) and b:
		raise Exception(f"You asked for download destination name {dst}, but you also passed b=True, which means you want the file in-memory in bytes. So which is it?")

	# Function to do the actual downloading
	# (we may or may not call it in a context manager, so it's separated out)
	def _dl(dst_dir, dst_name=None):
		with Sel.tmp(headless=True, linger=False, download_dir=dst_dir) as sel:
			before = ls(dst_dir)
			sel.load_new_window(url) # to prevent hanging

			for _ in range(tries):
				while len(after:=ls(dst_dir)) == len(before):
					sleep(wait)
			if len(after) <= len(before):
				raise Exception(f"Couldn't download {url}")

			fn = ospj(dst_dir, after[-1])

			if dst_name is not None:
				# This could theoretically just be force=True, since, if the file
				# exists, we're only here if clobber=True, but I think it's better
				# to be redundant because I'm not very smart
				mv(fn, (nfn:=ospj(dst_dir, dst_name)), force=clobber)
				fn = nfn

			return fn

	# Download the file
	if dst_dir is None:
		with tmpdir() as dst_dir:
			fn = _dl(dst_dir, dst_name)
	else:
		fn = _dl(dst_dir, dst_name)

	if ensure_newline:
		add_newline_if_missing(fn)

	if dst_name is None:
		# Return the downloaded file in memory & delete the file
		return read(fn, b=b)
	else:
		# Return the downloaded file's path
		return fn
