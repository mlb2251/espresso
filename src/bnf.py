
import re


def flatten(items):
    assert isinstance(items,list)
    result = []
    for x in items:
        if isinstance(x,list):
            result.extend(flatten(x))
        elif isinstance(x,ITEM):
            x.flatten()
            result.append(x)
        else:
            result.append(x)
    return result

class ITEM:
    def flatten(self):
        for val in vars(self):
            if isinstance(val,list):
                flatten(list)
            elif isinstance(x,ITEM):
                x.flatten()

class PRODUCTION(ITEM):
    def __init__(self,items,rhs):
        assert isinstance(items,list)
        assert isinstance(rhs,str)
        self.items = items # [ITEM]
        self.rhs = rhs # str or None
class OR(ITEM):
    def __init__(self,left,right):
        assert isinstance(left,list)
        assert isinstance(right,list)
        self.left = left # [ITEM]
        self.right = right # [ITEM]
class ASN(ITEM):
    def __init__(self,name):
        assert isinstance(name,str)
        self.name = name # str
        self.items = None
class BNF(ITEM):
    def __init__(self,name,args=""):
        assert isinstance(name,str)
        assert isinstance(right,list)
        self.name = name
        self.args = args
class MACRO(ITEM):
    def __init__(self,name,args):
        assert isinstance(name,str)
        assert isinstance(right,list)
        if len(self.args) > 0:
            assert isinstance(self.args[0],list)
        self.name = name
        self.args = args
    def flatten(self):
        self.args = [flatten(a) for a in self.args]
class SYM(ITEM):
    def __init__(self,body):
        assert isinstance(body,str)
        self.body = body
class KW(ITEM):
    def __init__(self,str):
        assert isinstance(body,str)
        self.body = body
class MAYBE(ITEM):
    def __init__(self,list):
        assert isinstance(items,list)
        self.items = items

re_macro = re.compile(r'([A-Z0-9_]+)\(')
re_asn = re.compile('([a-z0-9_]+):')
re_ws = re.compile('\s+')
re_fname = re.compile('([a-z_]\w*)')
re_kw = re.compile(r'([\'"])(\w*)\1')
re_sym = re.compile(r'([\'"])(\W*)\1')
regexes = [re_macro, re_asn, re_ws, re_fname, re_kw, re_sym]

re_quote = re.compile(r'([\'"])((\\\1)|[^\1])\1')


## TODO check for dangling assignment
## btw a nicer ASN would use a method that parses exactly one unit

class BNFParser:
    def __init__(self,input):
        self.input = input
        self.charno = 0
    def step(self,chars=1):
        self.input = self.input[chars:]
        self.charno += chars
    def parseline(self):
        production = self.parse_production('')
        assert isinstance(production,PRODUCTION)
        production.flatten()
    def parse_production(self,closer):
        assert closer in (')',']','')
        result = []
        args = [] # not used usually
        while True:
            char = remaining[:1] # a single char or empty str
            self.step()
            if char == '':
                if closer != '':
                    raise SyntaxError(f'unmatched parens/brackets at end of string')
                return PRODUCTION(result,None)
            if char == '[':
                result.append(MAYBE(self.parse_production(']')))
            elif char == ']':
                if closer != ']':
                    raise SyntaxError('mismatched parens/brackets')
                return result
            elif char == '(':
                result.append(self.parse_production(')'))
                return result # append list so that "*" can act on it etc, at end doflattening
            elif char == ',':
                if closer != 'args':
                    raise SyntaxError(f'comma can only appear when parsing arguments')
                args.append(result)
                result = []
            elif char == ')':
                if closer not in [')','args']:
                    raise SyntaxError(f'mismatched parens/brackets, expected {closer}')
                if closer == 'args':
                    args.append(result)
                    return args
                return result
            elif char == '|':
                return OR(result,self.parse_production(closer)) # thus a|b|c is a|(b|(c))
            elif char+input[:1] == '->':
                self.step()
                if closer != '':
                    raise SyntaxError(f'unmatched paren/brackets when starting "->"')
                return PRODUCTION(result,self.verbatim_python(''))
            elif char == ':':
                raise SyntaxError('improper ":" placement. Note: there can be no space between the var name and the colon')
            else:
                res = self.try_regex(char+self.input)
                if res is not None:
                    result.append(res)
            if len(result) >= 2 and isinstance(result[-2],ASN):
               result[-2].items = result.pop()

    def verbatim_python(self,closer):
        paren_depth = 0
        result = ""
        while True:
            char = self.input[:1]
            self.step()
            if char == '':
                if closer != '':
                    raise SyntaxError(f'unmatched ')
            elif char == '(':
                paren_depth += 1
                result += '('
            elif char == ')':
                paren_depth -= 1
                if paren_depth == -1:
                    return result
                result += ')'
            else:
                m = re_quote.match(char+self.input)
                if m is not None:
                    result += m.group()
                else:
                    result += char # generic case
        return result

    def try_regex(self,input):
        for regex in regexes:
            m = regex.match(input)
            if m is None:
                continue
            self.step(m.end())
            if regex is re_macro:
                return MACRO(m.group(1),self.parse_production('args'))
            elif regex is re_asn:
                name = m.group(1)
                return ASN(name)
            elif regex is re_ws:
                return None
            elif regex is re_fname:
                if len(self.input) == 0 or self.input[0] != '(':
                    return BNF(m.group(0),'')
                return BNF(m.group(0),self.verbatim_python(')'))
            elif regex is re_kw:
                return KW(m.group(0))
            elif regex is re_sym:
                return SYM(m.group(0))
            assert False # unreachable
        return None








re.compile('\w+:')

INDENT = '    '

class Builder:
    def __init__(self):
        self.text = ''
        self.idt = '' # indent
    def indent(self):
        self.idt += INDENT
    def dedent(self):
        assert self.idt != ''
        self.idt = self.idt[:-len(INDENT)]
    def build(self,line):
        self.text += f'{self.idt}{line}\n'


def run(text):
    lines = iter(text.split('\n'))
    while True:
        line = next(lines)
        if len(line.strip()) == 0:
            continue # empty line
        if line.strip()[0] == '#':
            continue # comment line
        assert '::=' in line
        assert line[0] not in [' ','\t']



class BNFParser:
    def __init__(self,line):
        self.line = line
        self.original = line
    def ws(self):
    def match(self,regex):
        regex.match(self.line).



