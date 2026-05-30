import __main__
import argparse
import atexit
import csv as _csv
import getpass
import hashlib
import importlib
import inspect
import io
import pyjson5 as _json
import keyring
import Levenshtein
import matplotlib.pyplot as plt
import os
import pathlib
import pickle as pkl
import random
import re
import readline
import requests
import select
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback as _traceback
import undetected_chromedriver as uc
import urllib

from base64 import b64encode, b64decode
from bs4 import BeautifulSoup as Soup
from collections import defaultdict, namedtuple
from contextlib import contextmanager as _cm, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from datetime import timedelta as td
from dotenv import find_dotenv, load_dotenv
from functools import wraps
from matplotlib import colors as mcolors
from rich import print as richprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, track as _track
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from term_image.image import from_url
from uuid import uuid4

from rich.color import ANSI_COLOR_NAMES
from rich._palettes import EIGHT_BIT_PALETTE
from rich.terminal_theme import DEFAULT_TERMINAL_THEME


# ospj = os.path.join
def ospj(*a, **kw):
	a = list(a)
	for i, e in enumerate(a):
		if not isinstance(e, str):
			a[i] = str(e)
	return os.path.join(*a,**kw)

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

def is_image_fn(fn):
	return fn.split('.')[-1].lower() in ('jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp')

def err_print(*a, **kw):
	if 'stderr' in kw:
		del kw['stderr']
	return print(*a, stderr=True, **kw)

def print(*a, synt=None, stderr=False, **kw):
	if synt is not None:
		return syntax(*a, synt=synt, **kw)

	if len(a) == 1 and isinstance(a[0], str):
		if (t:=a[0].strip()).startswith('{') and t.endswith('}'):
			try:
				txt = _json.dumps(_json.loads(a[0]), indent=2)
				return richprint(txt)
			except _json.JSONDecodeError as e:
				pass
		elif t.startswith('http') and is_image_fn(a[0]):
			ext = a[0].split('.')[-1]
			write(http(t, b=True), f'/tmp/img.{ext}')
			print(run(f'kitty icat /tmp/img.{ext}').strip())
			return
		elif '/' in a[0] and fexists(a[0]) and is_image_fn(a[0]):
			print(run(f'kitty icat {a[0]}').strip())
			return

	if not isatty():
		return oldprint(*a, **kw)

	try:
		_print(*a, **kw)
	except Exception as e:
		# _oldprint(e)
		if isinstance(e, KeyboardInterrupt):
			raise e
		try:
			richprint(*a, **kw)
		except:
			oldprint(*a, **kw)

_ll_global_console = Console()
_ll_global_error_console = Console(stderr=True)
def _print(*a, stderr=False, **kw):
	global _ll_global_console
	global _ll_global_error_console
	console = _ll_global_console if not stderr else _ll_global_error_console

	dkt = type({}.keys())
	dvt = type({}.values())
	dit = type({}.items())
	for i, e in enumerate(a):
		if any(isinstance(e, t) for t in (dkt, dvt, dit)):
			a = list(a)
			a[i] = list(e)

	# from term_image.image import BaseImage
	if any('KittyImage' in str(type(x)) for x in a):
		return oldprint(*a, **kw)
	if any((len(bs:=_bytes(str(x).strip(), encoding='utf-8'))>0 and bs[0]==27) for x in a):
		return oldprint(*a, **kw)
	if len(a)==1:
		fl = str(a[0]).strip().split('\n')[0].lower()
		if '<!doctype' in fl or '<html' in fl:
			# return richprint(Syntax(Soup(a[0], 'html.parser').prettify(), 'html'))
			return console.print(Syntax(Soup(a[0], 'html.parser').prettify(), 'html'))

	# richprint(*a, **kw)
	for i, e in enumerate(a):
		if isinstance(e, str):
			_a = list(a)
			for k, v in {
				' green]': ' dark_sea_green3]',
				' red]': ' light_coral]',
				'[green ': '[dark_sea_green3 ',
				'[red ': '[light_coral ',
				'[green]': '[dark_sea_green3]',
				'[/green]': '[/dark_sea_green3]',
				'[red]': '[light_coral]',
				'[/red]': '[/light_coral]',
			}.items():
				_a[i] = _a[i].replace(k, v)
			a = tuple(_a)
	console.print(*a, **kw)
	
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
alpha_lower = 'abcdefghijklmnopqrstuvwxyz'
alpha_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

nums = '0123456789'
digits = nums

def sleep(*a, **kw):
	return time.sleep(*a, **kw)

def is_file(fn):
	fn = fix_path(fn)
	if not fexists(fn):
		err(f"couldn't find [grey70]{fn}[/grey70]")
	return os.path.isfile(fn)
isfile = is_file

def is_dir(fn):
	fn = fix_path(fn)
	if not fexists(fn):
		err(f"couldn't find [grey70]{fn}[/grey70]")
	return os.path.isdir(fn)
isdir = is_dir

def uuid(dash=True):
	u = str(uuid4())
	if not dash:
		u = u.replace('-', '')
	return u

def regf(regex, multiline=True, all=True):
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

def splitf(regex):
	def _(s):
		chunks = []
		buf = ''
		while s:
			match = regf(regex, all=False)(s)
			if match and s.index(match)==0:
				if buf:
					chunks.append(buf)
					buf = ''
				chunks.append(match)
				s = s[len(match):]
			else:
				buf += s[0]
				s = s[1:]
		if buf:
			chunks.append(buf)
			buf = ''

		return chunks
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

def dedupe_itr(l, key=lambda x:x):
	seen = set()
	for e in l:
		if key(e) in seen:
			continue
		seen.add(key(e))
		yield e

def dedupe(l, key=lambda x:x):
	r = dedupe_itr(l, key=key)
	if type(l) == list:
		r = list(r)
	return r

def rule(title=None, pre_space=1, post_space=1):
	pre = []
	for i in range(pre_space):
		if i > 0:
			pre.append('\n')
		else:
			pre.append('')
	post = []
	for i in range(post_space):
		if i > 0:
			post.append('\n')
		else:
			post.append('')

	a = pre + [Rule(title=title, style='grey30')] + post
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


def first_line(fn):
	return next(stream_lines(fn))


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
	if fexists(txt) and not fexists(fn):
		txt,fn=fn,txt
	elif fexists(fn) and not fexists(txt):
		pass
	elif len(fn)>100 and not len(txt)>100:
		txt,fn=fn,txt
	elif len(lines(fn))>1 and len(lines(txt))==1:
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

def write(fn, txt, ensure_newline=True, create_dirs=True, swap=True):
	if swap:
		if isinstance(txt, dict):
			txt = _json.dumps(txt, indent=2)
		if isinstance(fn, dict):
			fn = _json.dumps(fn, indent=2)
			txt,fn=fn,txt
		if isinstance(txt, list) and all(isinstance(x, dict) for x in txt):
			txt = _json.dumps(txt, indent=2)
		if isinstance(fn, list) and all(isinstance(x, dict) for x in fn):
			fn = _json.dumps(fn, indent=2)

		if len(fn)>100 and not len(txt)>100:
			txt,fn=fn,txt
		elif '/' in txt and '/' in fn and fexists(abs(dirname(txt))) and (not fexists(abs(dirname(fn)))):
			fn,txt=txt,fn
		elif (not fexists(fn)) and (len(lines(fn))>1 and len(lines(txt))==1):
			txt,fn=fn,txt
		elif '/' in txt and '/' in fn and fexists(dirname(txt)) and not fexists(dirname(fn)):
			txt,fn=fn,txt
	

	if (dn:=os.path.dirname(fn)) and not fexists(dn):
		mkdir(dn)

	fn = fix_path(fn)

	if ensure_newline and isinstance(txt, str) and (len(txt) == 0 or txt[-1] != '\n'):
		txt += '\n'
	mode = 'w+'
	if isinstance(txt, bytes):
		mode = 'wb+'
	with open(fn, mode) as f:
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

def nhead(x, n):
	def _it():
		with open(x, 'r') as f:
			for i, l in enumerate(f.readlines()):
				if i < n:
					continue
				yield l

	return ''.join(list(_it())).strip()

def lines(s, intuit_file=True, stream=False, strip=True):
	if type(s) != str:
		s = str(s)

	def _itr(base):
		for line in base:
			if (not strip) or (line:=line.strip()):
				yield line

	if intuit_file and os.path.exists(fix_path(s)):
		def _it():
			with open(fix_path(s), 'r') as f:
				while True:
					try:
						yield next(f)
					except StopIteration:
						break
		it = _itr(_it())
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
	content = s.encode(encoding) if not fexists(fix_path(s)) else bread(s)
	h = hashlib.md5(content).hexdigest()
	return int(h, 16) if b else h

def md5_int(s, encoding='utf-8'):
	return md5(s, encoding=encoding, b=True)
int_md5 = md5_int

def fmd5(s, encoding='utf-8'):
	return hashlib.md5(bread(s)).hexdigest()

def detect_single_csv_row(x, delim=','):
	if isinstance(x, type({}.keys())):
		return detect_single_csv_row(list(x))
	if isinstance(x, type({}.values())):
		return detect_single_csv_row(list(x))
	if isinstance(x, dict):
		return render_csv([x], no_headers=True, delim=delim)
	if isinstance(x, list) and len(x)>0 and isinstance(x[0], dict):
		return render_csv(x, delim=delim)
	if any(isinstance(x, y) for y in (list, tuple, set, type({}.keys()))):
		return render_csv(list(x), no_headers=True, delim=delim)

	return None

def csv_row(row, delim=','):
	if (rv:=detect_single_csv_row(row, delim=delim)) is not None:
		return rv

	if len(lines(row.strip())) == 1:
		return next(_csv.reader([row.strip()], delimiter=delim))
	else:
		return next(_csv.reader(io.StringIO(row), delimiter=delim))

def render_csv(row_dicts, no_headers=False, delim=','):
	assert(len(row_dicts) > 0)

	if isinstance(row_dicts, dict):
		row_dicts = [row_dicts]
	if any(not isinstance(y, dict) for y in row_dicts):
		buf = ''
		for i, x in enumerate(row_dicts):
			if i>0:
				buf += delim
			if delim in str(x):
				buf += f'"{x}"'
			else:
				buf += str(x)
		return buf

	def _render_field(x):
		if isinstance(x, datetime):
			return x.strftime('%Y-%m-%d %H:%M:%S')

		x = str(x)
		if delim in x:
			return '"' + x.replace('"', '""') + '"'
		else:
			return x

	buf = ''
	if not no_headers:
		buf += delim.join(map(_render_field, list(row_dicts[0].keys())))
	for i, row in enumerate(row_dicts):
		if (hasattr(row_dicts, '__len__') and len(row_dicts)>=1) and (i==0 or not no_headers):
			buf += '\n'
		buf += delim.join(map(_render_field, list(row.values())))

	return buf

def csv(fn, delim=None, convert=True, empty='', stream=False, **kwargs):
	if isinstance(fn, str) and (len(lines(fn))==1) and (len(fn.split(delim))>1) and (not fexists(fn)):
		return csv_row(fn, delim=delim or ',')

	if isinstance(fn, list) and len(fn) > 0 and all(isinstance(e, dict) for e in fn):
		keys = sorted(list(fn[0].keys()))
		for row in fn[1:]:
			if sorted(list(row.keys())) != keys:
				raise Exception(f"rows of input have different sets of keys")
		return render_csv(fn, delim=delim or ',')

	assert(not ('dicts' in kwargs and 'header' in kwargs))
	header = True
	if 'dicts' in kwargs:
		header = kwargs['dicts']
	elif 'header' in kwargs:
		header = kwargs['header']


	if (rv:=detect_single_csv_row(fn, delim=delim or ',')) is not None:
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

	# I can't remember why I even added this...
	# if wc_l(fn) == 1: # TODO: faster wc_l in general
		# return next(_csv.reader(lines(fn), delimiter=delim))

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
							nr.append(datetime.strptime(e, '%Y-%m-%d %H:%M:%S'))
							continue
						except ValueError:
							pass

						try:
							nr.append(int(e))
							'''
							if (m:=re.findall('^([0-9]+)$', e)) and m[0]==e:
								nr.append(int(e))
							else:
								if int(e)==float(e):
									nr.append(int(e))
								else:
									nr.append(float(e))
							'''
						except ValueError:
							try:
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
	cache_str = ' '.join([
		str('.'.join([f.__module__, f.__qualname__])),
		' '.join([str(a) for a in args]),
		' '.join([f'{k}={v}' for k, v in kwargs.items()]),
	])

	return f.__qualname__ + md5(cache_str)


here_cache_cache = dict()
def cache(stale=None, cache_base=None):
	def inner_cache(f, stale=None, cache_base=None):
		@wraps(f)
		def wrapper(*args, stale=None, cache_base=None, **kwargs):
			global here_cache_cache
			if cache_base is None:
				if f.__qualname__ not in here_cache_cache:
					here_cache_cache[f.__qualname__] = here(up=2) 
				cache_base = here_cache_cache[f.__qualname__]
			for a in [*args]+list(kwargs.values()):
				if (' object at ' in str(a)) and (not '__hash__' in a.__class__.__dict__):
					print(f"Warning: can't hash argument \"{str(a)}\" to cache <{str(f)}> call")
					return f(*args, **kwargs)

			key = cache_key(f, args, kwargs)
			path = os.path.join(cache_base, f'cache/{key}')

			if os.path.exists(os.path.abspath(path)) and ((stale is None) or (age(path)<stale)):
				return unpickle(path)
			else:
				cdir = os.path.join(cache_base, 'cache')

				# Actually calculate the result
				res = f(*args, **kwargs)

				os.makedirs(cdir, exist_ok=True)
				pickle(path, res)
				return res

		return lambda *a,**kw: wrapper(*a, stale=stale, cache_base=cache_base, **kw)
	return lambda _f: inner_cache(_f, stale=stale, cache_base=cache_base)


def wc_l(fn, empties=True):
	return int(run(f'cat {fn} | wc -l').strip())

	# TODO: faster
	fn = fix_path(fn)

	count = 0
	with open(fn, 'r') as f:
		try:
			for l in f.readlines():
				if (not empties) or l.strip():
					count += 1
		except UnicodeDecodeError:
			print(fn)
			return None
	return count


def wc_c(fn):
	return os.stat(fix_path(fn)).st_size


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
				continue
			return resp
		except Exception as e:
			time.sleep(2)

def _html(url, cookies={}, user_agent=None, fake_user=False, b=False, method=requests.get, **kwargs):
	if user_agent is not None:
		kwargs.update({'headers': {
			'User-Agent': user_agent,
		}})
	elif fake_user:
		kwargs.update({'headers': {
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
		}})
	if 'payload' in kwargs and 'json' not in kwargs:
		kwargs['json'] = kwargs['payload']
		del kwargs['payload']

	def _method(*a, **kw):
		if not cookies:
			# __builtins__['input'](str(a))
			# __builtins__['input'](str(kw))
			resp = method(*a, **kw)
			# _oldprint(resp, resp.text)
			return resp
		else:
			with requests.Session() as s:
				s.cookies.update(cookies)
				print(s.cookies)
				quit()
				return method(*a, **kw)

	try:
		res = _method(url, **kwargs)
		if b:
			return res.content
		else:
			return res.text
	except requests.exceptions.MissingSchema:
		try:
			res = _method('https://'+url, **kwargs)
			return res.content if b else res.text
		except:
			res = _method('http://'+url, **kwargs)
			return res.content if b else res.text
http = html
webpage = html
web = html
site = html
page = html
url = html

def soup(url, **kwargs):
	if url.strip().endswith('</html>'):
		return Soup(url, 'html.parser')
	return Soup(html(url, **kwargs), 'html.parser')


def sjson(txt):
	return _json.loads(txt)


def json(url):
	if isinstance(url, dict) or isinstance(url, list):
		return _json.dumps(url, indent=2)
	else:
		if (url:=url.strip()).startswith('{') or url.startswith('['):
			txt = url
		elif os.path.exists(url):
			txt = read(url)
		else:
			try:
				txt = lget(url)
			except requests.exceptions.ConnectionError:
				raise Exception(f"idk how to interpret '{url}' as a JSON file, sorry")

		return _json.loads(txt)


def here(p='', up=0, abs=True):
	if p.startswith('/'):
		return p
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
	path = fix_path(path)

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

def run(cmd, live=False, out=True, err=False, code=False, dir=None):
	dir = fix_path(dir)

	if live:
		out = False
		err = False

	if dir is not None:
		if not fexists(dir):
			raise Exception(f"{dir} doesn't exist")
		elif not isdir(dir):
			raise Exception(f"{dir} isn't a directory")
		else:
			cmd = f'cd {os.path.abspath(dir)} && {cmd}'

	kw = {}
	if not live:
		kw['stdout'] = subprocess.PIPE
		kw['stderr'] = subprocess.PIPE
	p = subprocess.Popen(
		['bash', '-c', cmd],
		**kw
	)

	_out, _err = p.communicate()
	if not live:
		_out, _err = _out.decode(), _err.decode()
	_code = p.returncode
	if live:
		return _code


	ret = []
	if out:
		ret += [_out]
	if err:
		ret += [_err]
	if code:
		ret += [_code]

	ret = tuple(ret)
	if len(ret) == 1:
		ret = ret[0]

	return ret

def bash(cmd, from_dir=None, crash=False, stderr=False, strip=True):
	out, err = run(cmd, from_dir=from_dir, stderr=True, strip=strip) # [sic]
	if (not out) and err:
		raise Exception(f"bash command '{cmd}' had no output w/ error text: {err}")
	elif err and crash:
		raise Exception(f"bash command '{cmd}' had error output: {err}")
	return (out, err) if stderr else out
def bashcrash(*a, **kw):
	if 'crash' in kw:
		del kw['crash']
	return bash(*a, crash=True, **kw)

def abs(path):
	return os.path.abspath(str(path))

def fix_path(path):
	if path is None:
		return None

	if path.startswith('~'):
		path = path.replace('~', os.environ.get('HOME'), 1)

	return path

def ls(path='.', abs=None, rel=None, t=True, pat=None):
	if abs and rel:
		print('\n[grey50]ll.ls got both abs and rel; choosing abs[grey50]\n')

	if '*' in (pp:=path.split('/')[-1]):
		pat = pp.replace('*', '.*')
		path = os.path.dirname(path)
		if path == '':
			path = '.'

	if pat is not None:
		pat = ('^' if (not pat.startswith('^')) else '') + pat + ('$' if (not pat.endswith('$')) else '')

	if path.startswith('~'):
		path = path.replace('~', os.environ.get('HOME'), 1)

	if path.startswith('/'):
		if abs is None:
			abs = True
	elif '/' in path:
		if rel is None:
			rel = True

	if abs:
		path = os.path.abspath(path)
	elif rel:
		path = os.path.relpath(path, os.path.abspath(os.path.dirname(
			inspect.getframeinfo(inspect.stack()[1][0]).filename)))

	files = os.listdir(path)
	if abs or rel:
		files = [os.path.join(path, f) for f in files]

	if pat is not None:
		files = [f for f in files if regf(pat)(f)]

	if t:
		files = sorted(files, key=lambda f: os.path.getctime(os.path.join(path, f) if not (abs or rel) else f))

	return files


bn = basename = os.path.basename
dn = dirname = os.path.dirname


def crawl(p='.', stream=False, filt=None, abs=False):
	def _ok(x):
		return (filt is None) or (re.search(filt, x))

	if isfile(p):
		if _ok(p):
			def _sit():
				if abs:
					yield os.path.abspath(p)
				else:
					yield rempre(p, os.path.abspath(os.getcwd()) + '/')
			return _sit() if stream else list(_sit())
		else:
			raise Exception(f"You passed a single file ('{p}'), but it didn't match the filter ('{filt}')")

	def _it():
		for fn in ls(p, abs=True):
			if isdir(fn):
				for e in crawl(p=fn):
					if _ok(e):
						yield e
			else:
				if _ok(fn):
					yield fn
	def _it2():
		for e in _it():
			if abs:
				yield p
			else:
				yield rempre(e, os.path.abspath(os.getcwd()) + '/')

	return _it2() if stream else list(_it2())


def dot(k):
	return lambda x: x[k] if isinstance(x, dict) else getattr(x, k)

def dotcall(k):
	return lambda x: x[k]() if isinstance(x, dict) else getattr(x, k)()

def dot_eq(k, v):
	return lambda x: v == (x[k] if isinstance(x, dict) else getattr(x, k))

def fexists(p):
	if (not isinstance(p, str)) and (not isinstance(p, pathlib.Path)):
		return False
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

	ss = [f'{quote}{e}{quote}' for e in l]

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

def track(i, total=None, title='', console=None, init=0, transient=False):
	global _ll_global_console
	if console is None:
		console = _ll_global_console
	if hasattr(i, '__len__'):
		total = len(i)
	elif hasattr(i, '__length_hint__'):
		total = i.__length_hint__()

	kwargs = {}
	if total is not None:
		kwargs['total'] = total
	if title is not None:
		kwargs['title'] = title
	
	# TODO: better initial progress than total-init
	return _track(i, total=total-init, description=title, console=console, transient=transient)


def flat_track(*a, **kw):
	return list(track(*a, **kw))


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


def first(l, cond=bool):
	for e in filter(cond, l, stream=True):
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

def now():
	return datetime.now()

def safe_timedelta(*a, **kw):
	if len(a)==1 and isinstance(a[0], timedelta):
		return a[0]
	else:
		return timedelta(*a, **kw)

def days(n=1):
	return safe_timedelta(days=n)
day = days

def hours(n):
	return safe_timedelta(hours=n)
hour = hours

def minutes(n):
	return safe_timedelta(minutes=n)
minute = minutes

def seconds(n):
	return safe_timedelta(seconds=n)
second = seconds

def ctime(fn):
	return datetime.fromtimestamp(os.stat(fix_path(fn)).st_ctime)

def mtime(fn):
	return datetime.fromtimestamp(os.stat(fix_path(fn)).st_mtime)

def touched(fn):
	return max(mtime(fn), ctime(fn))

def age(fn):
	return now()-touched(fn)

def dt(x):
	if isinstance(x, int) or isinstance(x, float):
		return dt(datetime.utcfromtimestamp(x))

	# For dates like 2026-05-19T20:00:05.9017741Z (in the tcgcsv data),
	# Python doesn't like the 7 digits at the end there; it only wants 6
	if isinstance(x, str):
		x = re.sub(
			r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.)(\d{6})\d+(Z)$',
			r'\1\2\3',
			x,
		)

	for fstr in (
		'%Y-%m-%dT%H:%M:%S.%fZ',
		'%Y-%m-%dT%H:%M:%S',
		'%Y-%m-%d %H:%M:%S',
		'%Y/%m/%d %H:%M:%S',
		'%Y-%m-%d %H:%M',
		'%Y/%m/%d %H:%M',
		'%Y-%m-%d',
		'%Y/%m/%d',
	):
		try:
			if isinstance(x, str):
				return datetime.strptime(x, fstr)
			else:
				return x.strftime(fstr)
		except ValueError:
			continue

	raise ValueError(f"No fstrs we know about can parse '{x}'")


_ll_global_dotenv_found = False
def env(var, loc=None, refresh_global=False, crash=False):
	global _ll_global_dotenv_found

	assert(loc is None or refresh_global is False)

	for _base in (here(), os.getcwd()):
		if fexists(ospj(_base, '.env')):
			load_dotenv(ospj(_base, '.env'))
			_ll_global_dotenv_found = True
			return os.environ[var] if crash else os.environ.get(var)


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
	err_print('')
	if (pn:=os.path.basename(sys.argv[0])):
		err_print(f'[bold light_coral]{pn}:[/bold light_coral] error: ', end='')
	else:
		err_print('[bold light_coral]error:[/bold light_coral] ', end='')
	err_print(*a, **kw)
	err_print('')
	if kill or kill_9:
		if kill_9:
			os._exit(1)
		else:
			sys.exit(1)

def err(*a, **kw):
	return error(*a, **kw)

def warn(*a, **kw):
	print(f'[bold orange3]warning:[/bold orange3] ', end='')
	print(*a, **kw)
	print('')

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
class Sentinel:
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
	fn1 = fix_path(fn1)
	fn2 = fix_path(fn2)

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


def cp(fn1, fn2, force=False, exist_ok=False):
	fn1 = fix_path(fn1)
	fn2 = fix_path(fn2)

	ass(is_file(fn1), err_msg=f"[grey70]{fn1}[/grey70] is not a file")
	if fexists(fn2):
		if not (force or exist_ok):
			err(f"[grey70]{fn2}[/grey70] already exists; try calling with [grey70]force=True[/grey70] or [grey70]exist_ok=True[/grey70] if you don't care")
		elif exist_ok:
			return

	mkdir(dirname(fn2))

	return shutil.copy2(fn1, fn2)

def copy(*a, **kw):
	return cp(*a, **kw)

def escape(txt):
	return urllib.parse.quote(txt)
quote = escape

def unescape(txt):
	return urllib.parse.unquote(txt)
unquote = unescape

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
def tmp_file(content, b=False):
	with open(fn:=f'/tmp/{uuid(dash=False)}', 'w'+'b'*b+'+') as f:
		f.write(content)
	yield fn
	os.remove(fn)
tmpfile=tmp_file


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
		shutil.rmtree(dst)
tmpdir = tmp_dir


def add_newline_if_missing(fn):
	if not fexists(fn):
		raise Exception(f"Target {fn} does not exist")
	if len(las:=last_line(fn))>0 and las[-1] != '\n':
		with open(fn, 'a') as f:
			f.write('\n')


def sel(url, headers=None):
	with Sel.tmp(url, headers=None, headless=True, linger=False) as sel:
		return sel.src()


def sel_dl(
	url,
	dst_dir=None,
	dst_name=None,
	b=False,
	clobber=False,
	ignore=False,
	ensure_newline=True,
	tries=10,
	wait=1,
	max_wait=30,
	backoff_after=None,
	headless=True,
	cookies=None,
	headers=None,
):
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
		with Sel.tmp(headless=headless, linger=False, download_dir=dst_dir, cookies=cookies, headers=headers) as sel:
			before = ls(dst_dir)
			sel.load_new_window(url) # to prevent hanging

			wait_secs = wait
			pre_backoff_ticks = 0
			for _ in range(tries):
				if len(after:=ls(dst_dir)) > len(before):
					break
				sleep(wait_secs)
				pre_backoff_ticks += 1
				if backoff_after is not None:
					if pre_backoff_ticks >= backoff_after:
						wait_secs = min(max_wait, wait_secs*2)
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

	def _handle_dl(fn):
		if ensure_newline:
			add_newline_if_missing(fn)
		if dst_name is None:
			# Return the downloaded file in memory & delete the file
			return read(fn, b=b)
		else:
			# Return the downloaded file's path
			return fn

	# Download the file
	if dst_dir is None:
		with tmpdir() as dst_dir:
			fn = _dl(dst_dir, dst_name)
			return _handle_dl(fn)
	else:
		fn = _dl(dst_dir, dst_name)
		return _handle_dl(fn)




@_cm
def silent(out=True, err=True):
	with open(os.devnull, 'w') as f:
		with redirect_stderr(f):
			yield

silence = silent


# from selenium import webdriver
with silent():
	from seleniumwire import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chromium.options import ChromiumOptions
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
class Sel:
	def __init__(self, url=None, headless=True, linger=False, download_dir=None, cookies=None, headers=None, delay=0, user_agent=None):
		self.headless = headless
		self.last_loaded_url = None
		self.delay = delay
		self.user_agent = user_agent

		self.download_dir = download_dir
		if self.download_dir is None:
			# self.download_dir = os.path.join(os.environ['HOME'], 'Downloads')
			self.download_dir = f'/tmp/{uuid()}'
			os.makedirs(self.download_dir)

		self.options = Options()
		self.options.set_preference('permissions.default.stylesheet', 2)
		self.options.set_preference('permissions.default.image', 2)
		self.options.set_preference('browser.download.folderList', 2)
		self.options.set_preference('browser.download.manager.showWhenStarting', False)
		self.options.set_preference('browser.download.dir', os.path.abspath(self.download_dir))
		self.options.set_preference('devtools.jsonview.enabled', False)
		self.options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')
		if self.headless:
			self.options.add_argument('--headless')
		if self.user_agent is not None:
			self.options.set_preference('general.useragent.override', self.user_agent)
		self.driver = webdriver.Firefox(options=self.options)

		'''
		self.options = ChromiumOptions()
		if self.user_agent is not None:
			self.options.add_argument(f'--user-agent={self.user_agent}')
		uc.TARGET_VERSION = 146
		self.options.browser_executable_path = '/Users/lane/Downloads/chr/Google Chrome for Testing'
		self.driver = uc.Chrome(version_main=146, options=self.options, headless=self.headless, use_subprocess=False)
		'''



		self._interceptor = None
		self.headers = headers
		if isinstance(cookies, dict):
			self.cookies = cookies
		else:
			self.cookies = Sel.parse_cookies(cookies, dictate=False)

		if self.headers is not None:
			def interceptor(request):
				for k, v in self.headers.items():
					request.headers[k] = v
			if self._interceptor is None:
				self._interceptor = interceptor
			else:
				_x = self._interceptor
				self._interceptor = (lambda _y: lambda *a,**kw: interceptor(_y(*a,**kw)))(_x)

		if self.cookies is not None:
			def interceptor(request):
				request.headers['Cookie'] = self.cookies
			if self._interceptor is None:
				self._interceptor = interceptor
			else:
				_x = self._interceptor
				self._interceptor = (lambda _y: lambda *a,**kw: interceptor(_y(*a,**kw)))(_x)

		if self._interceptor is not None:
			self.driver.request_interceptor = self._interceptor



		self._closed = False
		if not linger:
			atexit.register(self.close)

		if url is not None:
			self.load(url)


	@staticmethod
	def parse_cookies(cookies, dictate=True):

		if isinstance(cookies, str):
			if fexists(cookies):
				cookies = read(cookies)
			if not dictate:
				return cookies

			d = {}
			cstrs = map(strip, cookies.split(';'))
			for cs in cstrs:
				spl = cs.split('=')
				d[spl[0]] = '='.join(spl[1:])

			return d

		elif isinstance(cookies, dict):
			return cookies
		else:
			return None


	@staticmethod
	@_cm
	def tmp(*a, **kw):
		sel = None
		try:
			sel = Sel(*a, **kw)
			yield sel
		finally:
			if (sel is not None) and ('linger' not in kw or not kw['linger']):
				sel.close()


	def load(self, url):
		self.driver.get(url)
		self.last_loaded_url = url


	def load_new_window(self, url):
		self.driver.execute_script(f'window.open("{url}","_blank");')


	def xpath(self, tag='*', **kw):
		for k in ('txt', 'text', 'content'):
			if k in kw:
				kw['contains'] = kw[k]
				del kw[k]
		contains = kw.get('contains')
		if 'contains' in kw:
			del kw['contains']

		attrs = [f'[@{k}="{v}"]' for k, v in kw.items()]
		if contains is not None:
			attrs.append(f'[contains(text(), "{contains}")]')

		return f'//{tag}' + ''.join(attrs)


	def el(self, wait=30, xpath=None, **kw):
		if xpath is not None:
			return WebDriverWait(self.driver, wait).until(
				EC.presence_of_element_located((By.XPATH, xpath)))
		else:
			return WebDriverWait(self.driver, wait).until(
				EC.presence_of_element_located((By.XPATH, self.xpath(**kw))))


	def els(self, *a, poll=1, wait=30, min_num=1, strict=True, **kw):
		found = []

		waited = 0
		while True:
			# If we're done or have waited too long
			if (len(found) >= min_num) or waited >= wait:
				break
			# Wait & account for the polling time
			WebDriverWait(self.driver, poll).until(
				EC.presence_of_element_located((By.XPATH, self.xpath(**kw))))
			waited += poll
			# Add all found elements
			for e in self.driver.find_elements(By.XPATH, self.xpath(**kw)):
				found.append(e)

		if strict and (len(found) < min_num):
			raise Exception(f"You asked for a minimum of {min_num} elements, and strict=True, and we only found {len(found)}")

		return found


	def click(self, **kw):
		if self.delay:
			time.sleep(random.random() * self.delay)
		self.el(**kw).click()


	def click_at(self, **kw):
		if self.delay:
			time.sleep(random.random() * self.delay)
		ActionChains(self.driver).move_to_element_with_offset(self.el(**kw), 5, 5).click().perform()


	def type(self, txt, **kw):
		if self.delay:
			time.sleep(random.random() * self.delay)
		self.el(**kw).send_keys(txt)


	def type_at(self, txt, **kw):
		if self.delay:
			time.sleep(random.random() * self.delay)
		ActionChains(self.driver).move_to_element(self.el(**kw)).click().send_keys(txt).perform()


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


	def dl(
		self,
		url,
		dst_dir=None,
		dst_name=None,
		b=False,
		clobber=False,
		ignore=False,
		ensure_newline=True,
		tries=10,
		wait=1,
		max_wait=30,
		backoff_after=None,
		cookies=None,
		headers=None,
	):
		# TODO: fix
		assert(dst_dir is None) 
		dst_dir = self.download_dir

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
			# with Sel.tmp(headless=headless, linger=False, download_dir=dst_dir, cookies=cookies, headers=headers) as sel:
			before = ls(dst_dir)
			self.load_new_window(url) # to prevent hanging

			wait_secs = wait
			pre_backoff_ticks = 0
			for _ in range(tries):
				if len(after:=ls(dst_dir)) > len(before):
					break
				sleep(wait_secs)
				pre_backoff_ticks += 1
				if backoff_after is not None:
					if pre_backoff_ticks >= backoff_after:
						wait_secs = min(max_wait, wait_secs*2)
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

		def _handle_dl(fn):
			if ensure_newline:
				add_newline_if_missing(fn)
			if dst_name is None:
				# Return the downloaded file in memory & delete the file
				return read(fn, b=b)
			else:
				# Return the downloaded file's path
				return fn

		# Download the file
		if dst_dir is None:
			with tmpdir() as dst_dir:
				fn = _dl(dst_dir, dst_name)
				return _handle_dl(fn)
		else:
			fn = _dl(dst_dir, dst_name)
			return _handle_dl(fn)


def norm(s, ok=alpha+nums+'_'):
	cmap = {c: c for c in ok}
	cmap.update({c.upper(): c for c in ok if c.upper() not in cmap})
	cmap.update({c.lower(): c for c in ok if c.lower() not in cmap})

	if ' ' not in ok:
		cmap.update({' ': '_'})

	if '-' not in ok:
		cmap.update({'-': '_'})
	elif '_' not in ok:
		cmap.update({'_': '-'})

	buf = ''
	for c in s:
		if c in cmap:
			buf += cmap[c]
	return buf


'''
class Arg:
	def __init__(self, name, type=None, default=Sentinel, required=None, optional=None):
		dash = name.startswith('-')
		name = norm(name, ok=alpha+nums+'-')

		# Make sure these are consistent (I sometimes forget which to use)
		if (optional is not None) and (required is not None):
			if optional == required:
				raise Exception(f"You passed optional={optional} and required={required}!")
		elif required is None:
			required = not optional
		elif optional is None:
			optional = not required
		elif (required is None) and (optional is None):
			required, optional = False, True
			if default != Sentinel:
				default = None

		# Sanity check
		if required and (default != Sentinel):
			raise Exception(f"How can '{name}' be required if you also gave a default value?")

		# Infer type from default
		_type = __builtins__['type']
		if (type is None) and (default != Sentinel):
			type = _type(default)
'''









def arg(*a, **kw):
	ap = argparse.ArgumentParser()
	ap.add_argument(*a, **kw)
	args, _ = ap.parse_known_args()

	short, long = None, None
	if a[0].startswith('-'):
		if a[0].startswith('--'):
			long = a[0]
		else:
			short = a[0]
	if len(a) > 1 and a[1].startswith('-'):
		if a[1].startswith('--'):
			long = a[1]
		else:
			short = a[1]

	arg = (long or short) or a[0]
	while arg[0] == '-':
		arg = arg[1:]

	return getattr(args, arg.lower().strip().replace('-', '_'))


def words(s):
	return [w.strip().lower() for w in s.strip().split()]


def words_in(a, b):
	if not (isinstance(a, str) and isinstance(b, str)):
		return False

	wsa, wsb = words(a), words(b)
	return len([wa for wa in wsa if wa in wsb])


def camelCase(s):
	if '_' not in s:
		return s
	chunks = s.split('_')
	return chunks[0].lower() + ''.join([c[0].upper()+c[1:].lower() for c in chunks[1:]])
camel = camelCase

def UpperCamel(s):
	if len(s) == '':
		return ''

	uc = camelCase(s)
	return uc[0].upper() + uc[1:]
upper_camel = UpperCamel
uppercamel = UpperCamel

def uncamel(s):
	chunks = []
	buf = ''
	for i, c in enumerate(s):
		if i==0:
			buf += c
			continue
		if c==c.upper():
			chunks.append(buf)
			buf = c
		else:
			buf += c
	if buf:
		chunks.append(buf)
		buf = ''

	return '_'.join([c.lower() for c in chunks])


def equal(a, b):
	return a == b
equals = equal


def rich(txt):
	return ((c:=Console(record=True, file=io.StringIO())).print(txt, end=''), c.export_text(styles=True))[-1]


def options(
	a,
	msg='Please choose one: ',
	msg_col='bold orange3',
	col1='gold3',
	col2='plum3',
	padding=1,
	idx=False,
):
	for _ in range(padding):
		print('')
	print(f"[{msg_col}]{msg}[/{msg_col}]")
	id2opt = {}
	for i, opt in enumerate(a):
		col = col1 if i%2 else col2
		print(f"\t[grey70]\[[/grey70]{i+1}[grey70]][/grey70] [{col}]{opt}[/{col}]")

	res = None
	while True:
		x = __builtins__['input'](rich('\n[grey70]Choice: [/grey70]'))
		if x.isnumeric() and (int(x) in range(1, len(a)+1)):
			res = a[int(x)-1]
			break

	for _ in range(padding):
		print('')

	return (int(x)-1) if idx else res


def no_nones(l):
	return [e for e in l if e is not None]
nonones = no_nones

def rempre(s, pre):
	if pre.startswith(s):
		s,pre=pre,s
	if s.startswith(pre):
		return s[len(pre):]
	return s

def remsuf(s, suf):
	if suf.endswith(s):
		s,suf=suf,s
	if s.endswith(suf):
		return s[:-len(suf)]
	return s


class lldd:
	def __init__(self, factory=list):
		self.factory = factory
		self.children = dict()
		# self.v = Sentinel
		self._val = Sentinel


	@property
	def val(self):
		if self._val == Sentinel:
			self._val = self.factory()
		return self._val

	@val.setter
	def val(self, v):
		self._val = v


	def __getattr__(self, attr):
		try:
			return self.__getattribute__(attr)
		except AttributeError:
			return self.val.__getattribute__(attr)


	def __iadd__(self, val):
		self._val = self.val + val
		return self._val


	def __getitem__(self, idx):
		if idx not in self.children:
			self.children[idx] = lldd(factory=self.factory)
		return self.children[idx]


	def __setitem__(self, idx, val):
		if idx not in self.children:
			self.children[idx] = lldd(factory=self.factory)
		self.children[idx].val = val


	# def __getattr__(self, attr):
		# return getattr(self.val, attr)


	# def __setattr__(self, attr, val):
		# self.__setattribute__(attr, val)
		# return setattr(self.val, attr, val)


	def __len__(self):
		return len(self.children)


	def keys(self):
		return list(self.children.keys())


	def vals(self):
		return list(self.children.vals())


	def items(self):
		return list(self.children.items())


	def __iter__(self):
		return self.items()


	def get(self, idx):
		return self.children.get(idx)


	def dict(self):
		caller = inspect.stack()[1]
		if len(self.children) > 0:
			_d = {}
			for k, v in self.children.items():
				_d[k] = v.dict()
			return _d
		else:
			recursion = (caller.filename == __file__ and caller.function == 'dict')
			return self.val if recursion else {}


# Errata:
# 	1. In reality, Patricia does not trie very hard
def patricia(trie, prefcomb=lambda a,b:f'{a} {b}' if a else b, _pref=''):
	if isinstance(trie, lldd):
		trie = trie.dict()

	if (not isinstance(trie, dict)) or len(trie)==0:
		return trie

	elif len(trie) == 1:
		tk = list(trie.keys())[0]
		# _pref = prefcomb(_pref, tk)
		return patricia(trie[tk], _pref=prefcomb(_pref,tk), prefcomb=prefcomb)
		# return {prefcomb(_pref, k): patricia(v, _pref=prefcomb(_pref, k), prefcomb=prefcomb)
			# for k, v in trie.items()}

	return {prefcomb(_pref,k): patricia(v, _pref=prefcomb(_pref,k), prefcomb=prefcomb) for k, v in trie.items()}



def trie(seqs=None):
	if seqs is None:
		seqs = []

	seqs = [list(x) for x in seqs]
	d = lldd(int)

	for seq in seqs:
		cursor = d
		for e in seq:
			cursor = cursor[e]

	return d


def replaces(s, repls):
	for k, v in repls.items():
		s = s.replace(k, v)
	return s


def safe_int(s, none=False):
	try:
		return int(s)
	except:
		return s if not none else None

def capitalize(s, words=True):
	if len(s)==0:
		return ''

	if words and ' ' in s:
		ws = s.split(' ')
		return ' '.join([capitalize(w, words=False) for w in ws])
	else:
		return s[0].upper() + s[1:].lower()


def ceil(x):
	return x if int(x)==x else int(x)+1

def floor(x):
	return int(x)

def round(x):
	return int(x+0.5)


def centering(l, r):
	# This is for estimating card centering,
	# so we'll round pessimistically
	l,r=l*100/(l+r),r*100/(l+r)
	l,r=min(l,r),max(l,r)
	l,r=floor(l),ceil(r)
	col = 'red' if l<45 else 'green'
	print(f'[{col}]{l}[/{col}] / [{col}]{r}[/{col}]')
	print(f'\t{int(l*100/50)}% (need 90%)')


def recdict(dd):
	return {
		k: (recdict(v) if isinstance(v, defaultdict) else v)
			for k, v in dd.items()
	}


_ll_global_memcache = {}
def memcache(f):
	global _ll_global_memcache
	def wrapper(*a, **kw):
		if (key:=cache_key(f,a,kw)) not in _ll_global_memcache:
			_ll_global_memcache[key] = f(*a, **kw)
		return _ll_global_memcache[key]
	return wrapper


class pbar:
	def __init__(self, title='', total=None, transient=False, console=None, dummy=False):
		global _ll_global_console

		self._dummy = dummy
		if self._dummy:
			return

		self._title = title
		self._total = total
		self._cur = 0
		self._prog = Progress(transient=transient, console=console or _ll_global_console)
		self._task = self._prog.add_task(self._title, total=self._total)
		self._started = False
		self._stopped = False

	def __iadd__(self, q):
		if self._dummy or self._stopped:
			return self
		if not self._started:
			self.start()
		self._cur += q
		self._prog.update(self._task, advance=q)
		if self._cur >= self._total:
			self.stop()

		return self

	def set_title(self, title):
		if self._dummy or self._stopped:
			return
		self._title = title
		self._prog.update(self._task, description=title)

	def start(self):
		if self._dummy or self._stopped or self._started:
			return
		self._started = True
		self._prog.start()

	def stop(self):
		if self._dummy or self._stopped:
			return
		self._stopped = True
		if self._cur < self._total:
			self.__iadd__(self._total-self._cur)
		self._prog.stop()

	def close(self):
		self.stop()


def chunks(it, n=1, stream=True):
	def _chit():
		buf = []
		for i, e in enumerate(it):
			buf.append(e)
			if len(buf) >= n:
				yield buf
				buf = []
		if buf:
			yield buf

	ret = _chit()
	if (t:=is_coll(it)):
		ret = t(ret)
	elif not stream:
		ret = list(ret)
	return ret
chunk = chunks


def indent(s, n=1):
	return '\n'.join([(('\t'*n)+line) for line in lines(s)])


def curl2url(curl):
	if fexists(curl):
		curl = read(curl)
	if not curl.startswith('curl'):
		err(f"input doesn't start with the word [grey70]curl[/grey70]")
	curl = curl.replace('\\', '')

	return regf("curl '(.*)'")(curl)


def curl2headers(curl):
	if fexists(curl):
		curl = read(curl)
	if not curl.startswith('curl'):
		err(f"input doesn't start with the word [grey70]curl[/grey70]")
	curl = curl.replace('\\', '')

	headers = dict(regf("-H '(.*): (.*)'", all=True)(curl))

	return headers


def curl2cookies(curl):
	if fexists(curl):
		curl = read(curl)
	cookies = {}
	for line in lines(curl):
		if line.startswith('-b'):
			content = "'".join(line.split("'")[1:-1])
			kvs = content.split('; ')
			for kv in kvs:
				spl = kv.split('=')
				k = spl[0]
				v = '='.join(spl[1:])
				cookies[k] = unquote(v)
	return cookies


def curl2any(curl, tag=None):
	if tag is None:
		raise Exception(f"You can't pass tag=None")

	if fexists(curl):
		curl = read(curl)
	ret = []
	for line in lines(curl):
		if line.startswith(f'-{tag}') or line.startswith(f'--{tag}'):
			content = "'".join(line.split("'")[1:-1])
			ret.append(content)

	return ret


def curl(cmd):
	if fexists(cmd):
		cmd = read(cmd)
	cmd = '\n'.join(l.replace('\\', '').strip() for l in cmd.split('\n') if l.replace('\\', '').strip())

	url = curl2url(cmd)
	headers = curl2headers(cmd)
	# cookies = curl2cookies(cmd)
	cookies = '; '.join([l.split('-b')[-1].strip() for l in lines(cmd) if '-b ' in l])

	print(sel_dl(url, headers=headers, cookies=cookies, headless=False))


@_cm
def loc(d):
	pwd = os.getcwd()
	try:
		os.chdir(d)
		yield
	finally:
		os.chdir(pwd)


def clear():
	print(run('clear'))


def retry(tries=3, try_wait=3, v=False, ctrl_c=True):
	def _1(f):
		@wraps(f)
		def _2(*a, **kw):
			for i in range(tries):
				try:
					return f(*a, **kw)
				except Exception as e:
					if i==tries-1 or isinstance(e, KeyboardInterrupt):
						raise e
					else:
						if v:
							print(e)
						time.sleep(try_wait)
		return _2
	return _1


class file:
	def __init__(self, path):
		path = fix_path(path)
		self.path = path if path.startswith('/') else os.path.join(os.getcwd(), path)


	def write(self, txt, b=False, nl=False, safe=False, create=False):
		if safe and os.path.exists(self.path):
			raise Exception(f"safe=True and exists({self.path})")
		if create:
			os.makedirs(os.path.dirname(self.path), exist_ok=True)

		if nl and txt and txt[-1] != '\n':
			txt += '\n'

		mode = 'wb+' if b else 'w+'
		with open(self.path, mode) as f:
			f.write(txt)


	def from_url(self, url, pbar=False, b=False, **kwargs):
		self.write('', **kwargs) # to proc warnings/errors before downloading
		kwargs['safe'] = False # bc we just checked by writing
		self.write(lget(url, b=b, prog=pbar), b=b, **kwargs)


	def append(self, txt, b=False, nl=False):
		if nl:
			if txt.split('\n')[-1].strip():
				txt += '\n'
			if os.path.exists(self.path) and (ex:=self.read()) and ex[-1] != '\n':
				txt = f'\n{txt}'

		mode = 'ab+' if b else 'a+'
		with open(self.path, mode) as f:
			f.write(txt)


	def read(self, b=False, fmt=None):
		mode = 'rb' if b else 'r'
		with open(self.path, mode) as f:
			txt = f.read()

		match fmt:
			case 'json':
				return _json.loads(txt)
			case 'csv':
				return csv(txt)
			case _:
				return txt


	def lines(self, stream=False, strip=True):
		def _itr():
			with open(self.path, 'r') as f:
				while True:
					try:
						line = next(f)
					except StopIteration:
						break
					if strip and not (line:=line.strip()):
							continue
					yield line

		return _itr() if stream else list(_itr())


def lget(url, b=False, prog=False, stream=False):
	def _itr():
		resp = requests.get(url, stream=True)

		total = int(resp.headers.get('content-length', 0))
		blk = 1024

		p = pbar(total=total, dummy=(not prog))
		for chunk in resp.iter_content(blk):
			if total==0 and (total:=resp.headers.get('content-length', 0)):
				p = pbar(total=total, dummy=(not prog))
			if total > 0:
				p += len(chunk)
			yield chunk
		if total > 0:
			p.stop()

	if stream:
		return _itr()

	_resp = []
	for _chunk in _itr():
		_resp.extend(_chunk)
	_resp = bytes(_resp)

	return _resp if b else _resp.decode()


def be(x, t):
	if isinstance(x, type) and not isinstance(t, type):
		x,t=t,x
	return isinstance(x, t)


def wrap(s, width=None, console=None):
	if width is None:
		width = os.get_terminal_size().columns - 1

	if console is None:
		global _ll_global_console
		console = _ll_global_console

	return '\n'.join(map(str, list(Text(s).wrap(width=width, console=console))))


def is_coll(x):
	for t in (list, tuple, set):
		if be(x, t):
			return t
	return False


def sort(l, n=False, key=lambda x:x, reverse=False):
	_l, l = l, list(l) # back up the orig j.i.c.

	_proc = lambda r: ((lambda z: type(_l)(z) if is_coll(_l) else z)(r[::-1] if reverse else r))

	if len(l) == 0:
		return collify(l) # in case it wouldn't be a copy

	if not n:
		return _proc(sorted(l, key=key))

	# From here on out, we're sorting numerical
	# components first, then strings
	_num = lambda x: be(x, float) or be(x, int) or (be(x, str) and x.replace('.','').isnumeric())

	if is_coll(key(l[0])):
		# Sorting by composite keys

		# Group by length
		'''
			Shape:

			{
				1: [
					(orig_array_idx, (sort_key_col_1,)),
					(orig_array_idx, (sort_key_col_1,)),
				],
				2: [
					(orig_array_idx, (sort_key_col_1, sort_key_col_2)),
					(orig_array_idx, (sort_key_col_1, sort_key_col_2)),
				],
			}
		'''
		keys = map(key, l)
		lengroups = dict(sorted(agg(list(enumerate(keys)), key=lambda x: len(x[1])).items(), key=nth(0), reverse=reverse))

		final_idcs = []
		# for each length of sort key tuple
		for ln, tups in lengroups.items():
			# remember: tups are metatups of (orig_idx, tup)

			# for each item in the sort key tuple (reverse-wise)
			for i in range(ln-1, -1, -1):
				# separate nums and non-nums
				# note: j here is the tup's index *within this length group*;
				# the idx within the *original input* (since it's a key) is inertly
				# preserved within each entry
				# TODO: for loop to avoid second pass for nons?
				nums = [(j, float(x[1][i])) for j, x in enumerate(tups) if _num(x[1][i])]
				nons = [(j, x[1][i]) for j, x in enumerate(tups) if not _num(x[1][i])]

				# Sort separately
				nums = sorted(nums, key=nth(1), reverse=reverse)
				nons = sorted(nons, key=nth(1), reverse=reverse)

				# Recombine and project back onto tups
				comb = (nons+nums) if reverse else (nums+nons)
				tups = [tups[idx] for idx in [x[0] for x in (nums+nons)]]

			# this length group has now been sorted by each field,
			# so we'll add the original indices, in the new order,
			# to the final idx list
			final_idcs.extend([x[0] for x in tups])

		# now just reorder the input array and we're done!
		return _proc([l[idx] for idx in final_idcs])
	
	else:
		# OK, here is much easier: we're just sorting by
		# scalar keys
		nums = [(i, float(x)) for i, x in enumerate(l) if _num(x)]
		nons = [(i, x) for i, x in enumerate(l) if not _num(x)]
		idcs = [x[0] for x in sorted(nums,key=nth(1))+sorted(nons,key=nth(1))]
		return _proc([l[idx] for idx in idcs])

	# that wasn't so bad was it
	# 😭


def nsort(*a, **kw):
	return sort(*a, **{**kw, 'n': True})


def clean(l):
	if be(l, str):
		return l.strip()
	elif (t:=is_coll(l)):
		return t(e for e in l if clean(e))
	else:
		try:
			return [e for e in l if clean(e)]
		except TypeError:
			return l


def hex2rgb(x):
	x = x.replace('#', '').lower()
	x = [x[i:i+2] for i in range(0,len(x),2)]
	return tuple(int(e, 16) for e in x)


def rgb2hex(*a):
	if len(a) == 1:
		r,g,b = a[0]
	else:
		r,g,b = a

	return f'#{r:02x}{g:02x}{b:02x}'
	


def triplet_to_hex(t):
	return 

_ll_rich_colors = None
def colors():
	global _ll_rich_colors

	if _ll_rich_colors is None:
		_ll_rich_colors = {}
		for name, num in ANSI_COLOR_NAMES.items():
			t = (DEFAULT_TERMINAL_THEME.ansi_colors if num<16 else EIGHT_BIT_PALETTE)[num]
			_ll_rich_colors[name] = f'#{t.red:02x}{t.green:02x}{t.blue:02x}'
		# _ll_rich_colors = {k:_ll_rich_colors[k] for k in sorted(_ll_rich_colors.keys())}

	return _ll_rich_colors


def color(c):
	return colors()[c]


def topsort(l, is_child=lambda c,p:False):
	buf = l[::]
	take = []
	while buf:
		for i, e1 in enumerate(buf):
			for j, e2 in enumerate(buf):
				if i==j:
					continue
				if is_child(e2, e1):
					break
			# e1 has no children
			take.append(e1)
			buf.remove(e1)
			break

	return take


@_cm
def traceback():
	try:
		yield
	except Exception as e:
		print(_traceback.format_exc())


def hint(i, n):
	if be(i, int) and not be(n, int):
		i,n=n,i
	t = type(i)
	try:
		class _(t):
			def __init__(self, i, n):
				super().__init__(i)
				self.n = n
			def __length_hint__(self):
				return self.n
		return _(i, n)
	except:
		class _:
			def __init__(self, o, n):
				self.o = iter(o)
				self.n = n
			def __length_hint__(self):
				return n
			def __next__(self):
				return next(self.o)
			def __iter__(self):
				return self
		return _(i, n)


_ll_scrylen = None
def scryfall():
	fn = here('scryfall.json')

	global _ll_scrylen
	if _ll_scrylen is None:
		_ll_scrylen = wc_l(fn)

	def _it():
		for line in track(lines(fn, stream=True), total=_ll_scrylen):
			nfs = {'foil': 'Foil', 'etched': 'Etched', 'nonfoil': ''}

			if not line.startswith('{'):
				continue
			if line.endswith(','):
				line = line[:-1]

			yield json(line)

	return hint(_it(), _ll_scrylen)


def coalesce(*a):
	if len(a) == 1 and is_coll(a[0]):
		a = a[0]

	for e in a:
		if e is not None:
			return e

	return None


def sqlite(dbfn, autocommit=True):
	conn = sqlite3.connect(dbfn)
	def _():
		try:
			if autocommit:
				conn.commit()
		except:
			pass
		try:
			conn.close()
		except:
			pass
	atexit.register(_)
	conn.row_factory = lambda cur, row: {
		col[0]: row[i]
			for i, col in enumerate(cur.description)
	}

	return lambda q, params=None: conn.execute(*([q]+([params] if params else []))).fetchall()
