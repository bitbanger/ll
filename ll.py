import __main__
import csv as _csv
import getpass
import hashlib
import importlib
import inspect
import io
import json as _json
import keyring
import matplotlib.pyplot as plt
import os
import pickle as pkl
import re
import requests
import subprocess
import sys
import traceback

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

_olddir = __builtins__['dir']

_oldprint = __builtins__['print']
def oldprint(*a, **kw):
	global _oldprint
	return _oldprint(*a, **kw)

def print(*a, **kw):
	if not sys.stdout.isatty():
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

def _print(*a, **kw):
	# from term_image.image import BaseImage
	if any('KittyImage' in str(type(x)) for x in a):
		return oldprint(*a, **kw)
	if any((len(bs:=bytes(x.strip(), encoding='utf-8'))>0 and bs[0]==27) for x in a):
		return oldprint(*a, **kw)
	if len(a)==1:
		fl = str(a[0]).strip().split('\n')[0].lower()
		if '<!doctype' in fl or '<html' in fl:
			return richprint(Syntax(Soup(a[0], 'html.parser').prettify(), 'html'))

	richprint(*a, **kw)
	
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

def uuid():
	return str(uuid4())

def regf(regex):
	def _(s):
		res = re.search(regex, s)
		if res is None:
			return None
		return res.group(1)
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

def write(fn, txt, ensure_newline=True):
	if len(fn)>100 and not len(txt)>100:
		txt,fn=fn,txt

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

def csv_row(row):
	return next(_csv.reader(io.StringIO(row)))

def render_csv(row_dicts):
	assert(len(row_dicts) > 0)

	def _render_field(x):
		x = str(x)
		if ',' in x:
			return '"' + x.replace('"', '""') + '"'
		else:
			return x

	buf = ''
	buf += ','.join(map(_render_field, list(row_dicts[0].keys())))
	for row in row_dicts:
		buf += '\n'
		buf += ','.join(map(_render_field, list(row.values())))

	return buf

def csv(fn, delim=None, header=True, convert=True, empty=''):
	if isinstance(fn, list) and len(fn)>0 and isinstance(fn[0], dict):
		return render_csv(fn)

	fn = fix_path(fn)

	if len(fn) > 100 and not os.path.exists(fn):
		write(fn, fn:=f'/tmp/{uuid()}')
	if delim is None:
		with open(fn, 'r') as f:
			try:
				delim = _csv.Sniffer().sniff(f.read(1024), delimiters=',|\t').delimiter
			except Exception as e:
				if 'Could not determine delimiter' in str(e):
					delim = ','

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
				rows.append({cols[i]: row[i] for i in range(len(row))})
			else:
				rows.append(row)


	return rows
	

	
	

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
	ynstr = '([dark_sea_green3]y[/dark_sea_green3]/[light_coral]N[/light_coral])'
	if default_yes:
		ynstr = ynstr.replace('y','Y').replace('n','N')
	resp = Console().input(f'{msg} {ynstr}: ').strip().lower()
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
					if len(x) == 0:
						return [] if type(x)==list else tuple()
					n = [getattr(e, f)() for e in x]
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

def html(url):
	try:
		return requests.get(url).text
	except requests.exceptions.MissingSchema:
		try:
			return requests.get('https://'+url).text
		except:
			return requests.get('http://'+url).text
http = html
webpage = html
web = html
site = html
page = html
url = html

def soup(url):
	return Soup(html(url), 'html.parser')


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

def ls(path='.', abs=False, rel=False):
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

	return files

basename = os.path.basename
dirname = os.path.dirname

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

def digits(s):
	return ''.join(c for c in s if c in '0123456789')

def track(i, total=None):
	if total is not None:
		return _track(i, total=total)
	else:
		return _track(i)


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

	return _it() if stream else list(_it())


def first(f, l):
	for e in filter(f, l, stream=True):
		return e

def agg(l, extractor):
	if callable(l) and not callable(extractor):
		l, extractor = extractor, l

	dd = defaultdict(list)

	for e in l:
		dd[extractor(e)].append(e)

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
