
# the initial parser. This is used to parse bnf.ast and bnf.psr to generate a new gen II parser to replace this one :)
import re
import mlb


def flatten(y):
    if isinstance(y,ITEM):
        x = y.flatten()
        assert x is not None
        return x
    if not isinstance(y,list):
        return y

    # y is a list
    result = []
    for x in y:
        if isinstance(x,list):
            result.extend(flatten(x))
        else:
            result.append(flatten(x))
    return result

class ITEM:
    def flatten(self):
        for k,v in self.__dict__.items():
            setattr(self,k,flatten(v))
        return self

class PRODUCTION:
    def __init__(self,items,rhs):
        assert isinstance(items,list)
        assert isinstance(rhs,str) or rhs is None
        self.items = flatten(items) # [ITEM]
        self.rhs = rhs # str or None
    def __repr__(self):
        return f'{self.items} \n    {mlb.mk_red("->")} {self.rhs}'
class OR(ITEM):
    def __init__(self,left,right):
        assert isinstance(left,list)
        assert isinstance(right,list)
        self.left = left # [ITEM]
        self.right = right # [ITEM]
    def __repr__(self):
        return f'({self.left}|({self.right}))'

class ASN(ITEM):
    def __init__(self,name):
        assert isinstance(name,str)
        self.name = name # str
        self.items = None
    def flatten(self):
        if self.items is None:
            raise SyntaxError("Assignment to nothing!") # hijacking the fact that this is just called at the very end
        return self
    def __repr__(self):
        return f'({self.name}={self.items})'
class BNF(ITEM):
    def __init__(self,name,args=""):
        assert isinstance(name,str)
        assert isinstance(args,str)
        self.name = name
        self.args = args
    def __repr__(self):
        return f'{self.name}({self.args})'
class MACRO(ITEM):
    def __init__(self,name,args):
        assert isinstance(name,str)
        assert isinstance(args,list)
        if len(args) > 0:
            assert isinstance(args[0],list)
        self.name = name
        self.args = args
    def flatten(self): # so that self.args itself doesnt get flattened
        self.args = [flatten(a) for a in self.args]
        return self
    def __repr__(self):
        return f'{self.name}({self.args})'
class SYM(ITEM):
    def __init__(self,body):
        assert isinstance(body,str)
        self.body = body
    def __repr__(self):
        return f'{self.body}'
class KW(ITEM):
    def __init__(self,body):
        assert isinstance(body,str)
        self.body = body
    def __repr__(self):
        return f'{self.body}'
class MAYBE(ITEM):
    def __init__(self,items):
        assert isinstance(items,list)
        self.items = items
    def __repr__(self):
        return f'MAYBE({self.items})'
class KLEENESTAR(ITEM):
    def __init__(self,items):
        assert isinstance(items,list)
        self.items = items
    def __repr__(self):
        return f'({self.items})*'
class KLEENEPLUS(ITEM):
    def __init__(self,items):
        assert isinstance(items,list)
        self.items = items
    def __repr__(self):
        return f'({self.items})+'

re_macro = re.compile(r'(_*[A-Z][A-Z0-9_]*)\(')
re_asn = re.compile('([a-z0-9_]+):')
re_ws = re.compile('\s+')
re_fname = re.compile('(_*[a-z][a-z0-9_]*)')
re_kw = re.compile(r'([\'"])(\w*)\1')
re_sym = re.compile(r'([\'"])([^\'"a-zA-Z_]*?)\1')
regexes = [re_macro, re_asn, re_ws, re_fname, re_kw, re_sym]

re_quote = re.compile(r'([\'"])((\\\1)|[^\1])\1')


## TODO check for dangling assignment
## btw a nicer ASN would use a method that parses exactly one unit


#class QuickIter:
#    def __init__(self,iterable):
#        self.it = iter(iterable)
#    def __iter__(self,


class BNF_Fn:
    def __init__(self,name,argnames,rules,deco=None):
        assert isinstance(name,str)
        assert isinstance(rules,list)
        assert isinstance(argnames,list)
        assert len(rules) > 0
        assert isinstance(rules[0],PRODUCTION)
        assert isinstance(deco,str) or deco is None
        self.name = name
        self.rules = rules
        self.argnames = argnames
        self.deco = deco
    def __repr__(self):
        body = '\n'.join([repr(rule) for rule in self.rules])
        dstr = '@'+self.deco+' ' if self.deco is not None else ''
        return f'{dstr}{self.name}({self.argnames})\n{body}'

class Pdef_Fn:
    def __init__(self,lines):
        assert isinstance(lines,list)
        self.lines = lines
    def __repr__(self):
        body = '\n'.join(self.lines)
        return f"Pdef(\n{body}\n)"

class Macro_Fn:
    def __init__(self,name,argnames,rules):
        assert isinstance(name,str)
        assert isinstance(rules,list)
        assert isinstance(argnames,list)
        assert len(rules) > 0
        assert isinstance(rules[0],PRODUCTION)
        self.name = name
        self.rules = rules
        self.argnames = argnames
    def __repr__(self):
        body = '\n'.join([repr(rule) for rule in self.rules])
        return f'{self.name}({self.argnames})\n{body}'

class BNFParser:
    def __init__(self):
        self.input = None
        self.charno = 0 # for error printing
        self.lineno = 0 # for error printing
        self.prev_in = None
        self.prev_cno = None
    def step(self,chars=1):
        self.prev_in = self.input
        self.prev_cno = self.charno
        self.input = self.input[chars:]
        self.charno += chars
        print(self.input)
    def unstep(self):
        self.input = self.prev_in
        self.charno = self.prev_cno
    def parse_file(self,text):
        """
        Parses a full file of pdef and production rules, and returns (bnf_fns,pdef_fns,macro_fns) which are both lists.
        """
        it = iter(text.split('\n'))
        self.lineno = 0
        def nextline():
            """
            Returns the next line with the comment stripped off of it. Skips over empty or lines that only have a comment on them. Returns None at EOF
            """
            while True:
                self.lineno += 1
                try:
                    l = next(it)
                except StopIteration:
                    return None
                if '#' in l: # strip comment
                    l = l[:l.index('#')]
                if l.strip() == '':
                    continue
                return l

        bnf_fns = []
        pdef_fns = []
        macro_fns = []

        self.input = nextline()
        while True:
            deco = None
            if self.input is None:
                break

            # '@'
            if self.input.startswith('@'):
                self.step()
                m = re_fname.match(self.input)
                if m is None:
                    raise SyntaxError("expected name after `@`")
                deco = m.group()
                self.step(m.end())
                if self.input.strip() != '':
                    raise SyntaxError("Trailing characters")
                self.input = nextline()
            if deco is not None and '::=' not in self.input:
                raise SyntaxError("decorator should be followed by BNF fn on next line")

            # '::='
            # NOT elif
            if '::=' in self.input: # Parse a BNF fn
                bnf_name,argnames,ftype = self.parse_bnf_head()
                if ftype == 'macro' and deco is not None:
                    raise SyntaxError("Cant use decorator with Macro")
                bnf_rules = []
                indent = None
                while True: # parse the production rules
                    self.input = nextline()
                    if self.input is None:
                        if len(bnf_rules) == 0:
                            raise SyntaxError("unexpected EOF")
                        break
                    m = re_ws.match(self.input)
                    if m is None: # ie no whitespace
                        if len(bnf_rules) == 0:
                            raise SyntaxError("expected indent")
                        break
                    if indent is None: # first prod rule sets indent level
                        indent = m.group()
                    else:
                        if m.group() != indent:
                            raise SyntaxError("wrong indent level")
                    self.step(m.end()) # step past the whitespace
                    bnf_rules.append(self.parse_production(''))
                if ftype == 'bnf':
                    bnf_fns.append(BNF_Fn(bnf_name,argnames,bnf_rules,deco=deco))
                elif ftype == 'macro':
                    macro_fns.append(Macro_Fn(bnf_name,argnames,bnf_rules))
                else:
                    assert False # unreachable


            # 'pdef'. We parse until the first non-comment non-empty line that has zero indent level
            elif self.input.startswith('pdef'):
                self.step() # skip the 'p' in 'pdef'
                lines = [self.input]
                while True:
                    self.input = nextline()
                    if re_ws.match(self.input) is None: # no leading ws
                        break
                    lines.append(self.input)
                pdef_fns.append(Pdef_Fn(lines))
            else:
                raise SyntaxError(f"Unrecognized {self.input}")
        return bnf_fns,pdef_fns,macro_fns

    def parse_bnf_head(self):
        """
        Parse lines like "literal ::="
        Returns name,argnames,ftype
        argnames is a list of macro args or bnf inheritances (empty list if none)
        ftype is 'bnf' or 'macro'
        """
        if len(self.input.lstrip()) != len(self.input): # check for indent (forbidden)
            raise SyntaxError("Illegal indent on a BNF heading")
        fn = re_fname.match(self.input)
        macro = re_macro.match(self.input)
        if fn is None and macro is None:
            raise SyntaxError("Unable to parse BNF heading")
        if fn is not None and macro is not None:
            raise SyntaxError("fn name interpretable as either BNF or Macro")

        ftype = 'bnf' if fn is not None else 'macro'

        if ftype == 'bnf':
            self.step(fn.end())
            name = fn.group()
        else:
            self.step(macro.end())
            name = macro.group(1)

        # argless bnf
        if ftype == 'bnf' and self.input[:1] != '(':
            if self.input.strip() != '::=':
                raise SyntaxError('Unable to parse BNF heading')
            return name,[],ftype
        if ftype == 'bnf':
            self.step()

        argnames = [] # macro args or bnf inheritance
        while True: # parse parameters
            arg = re_fname.match(self.input)
            if arg is None:
                raise SyntaxError("error parsing parameters to macro or inheritance to bnf header")
            self.step(arg.end())
            argnames.append(arg.group())
            if self.input[:1] == ',': # next param
                self.step()
            while self.input[:1] == ' ': # skip spaces
                self.step()
            if self.input[:1] == ')':
                self.step()
                break # end of params
        if self.input.strip() != '::=':
            raise SyntaxError('Unable to parse BNF heading')
        return name,argnames,ftype # true for BNF_Fn

    def parse_production(self,closer,macro_args=False):
        #mlb.blue(f'parsing till {closer} macro_args:{macro_args}')
        """
        Parse a single production rule like '"(" v:starred_expression ")" -> v'
        """
        assert closer in (')',']','')
        result = []
        if macro_args:
            args = [] # not used usually

        while True:
            char = self.input[:1] # a single char or empty str
            self.step() # consume that char

            # EOL
            if char == '':
                if closer != '':
                    raise SyntaxError(f'unmatched parens/brackets at end of string')
                return PRODUCTION(result,None)
            # "->"
            elif char+self.input[:1] == '->':
                self.step()
                if closer != '':
                    raise SyntaxError(f'unmatched paren/brackets when starting "->"')
                return PRODUCTION(result,self.verbatim_python(''))
            # "[]"
            elif char == '[':
                result.append(MAYBE(self.parse_production(']')))
            elif char == ']':
                if closer != ']':
                    raise SyntaxError('mismatched parens/brackets')
                return result
            # "()"
            elif char == '(':
                result.append(self.parse_production(')'))
            elif char == ')':
                if closer != ')':
                    raise SyntaxError(f'mismatched parens/brackets, expected {closer}')
                if macro_args:
                    #mlb.green('closing paren for macro args')
                    args.append(result)
                    return args
                #mlb.green('closing paren')
                return result
            # ","
            elif char == ',':
                if not macro_args:
                    raise SyntaxError(f'comma can only appear when parsing arguments')
                args.append(result)
                result = []
            # "|"
            elif char == '|': # does a full `return` instead of append since it's the weakest binding thing there is
                # a|b|c is a|(b|(c))
                res = self.parse_production(closer)
                if closer == '':
                    assert isinstance(res,PRODUCTION)
                    res.items = [OR(result,res.items)]
                    return res # return a PRODUCTION
                return OR(result,res)
            # "*"
            elif char == '*':
                result.append(KLEENESTAR([result.pop()]))
            # "+"
            elif char == '+':
                result.append(KLEENEPLUS([result.pop()]))
            # err case
            elif char == ':':
                raise SyntaxError('improper ":" placement. Note: there can be no space between the var name and the colon')
            # regex or other
            else:
                self.unstep()
                res = self.try_regex()
                if res == 'no match':
                    raise SyntaxError(f"{self.input}") # unrecognized
                if res is not None:
                    result.append(res)

            # do assignment if ASN was parsed before this
            if len(result) >= 2 and isinstance(result[-2],ASN) and result[-2].items is None:
                x = result.pop()
                result[-1].items = x

    def verbatim_python(self,closer):
        paren_depth = 0
        result = ""
        while True:
            char = self.input[:1]
            self.step()
            if char == '':
                if closer != '':
                    raise SyntaxError(f'unmatched ')
                return result
            elif char == '(':
                #print('openparen')
                #print(paren_depth)
                paren_depth += 1
                result += '('
            elif char == ')':
                #print('closeparen')
                #print(paren_depth)
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

    def try_regex(self):
        for regex in regexes:
            m = regex.match(self.input)
            if m is None:
                continue
            self.step(m.end())
            if regex is re_macro:
                return MACRO(m.group(1),self.parse_production(')',macro_args=True))
            elif regex is re_asn:
                name = m.group(1)
                return ASN(name)
            elif regex is re_ws:
                return None
            elif regex is re_fname:
                if len(self.input) == 0 or self.input[0] != '(':
                    return BNF(m.group(0),'')
                self.step() # step past the paren
                return BNF(m.group(0),self.verbatim_python(')'))
            elif regex is re_kw:
                return KW(m.group(2))
            elif regex is re_sym:
                return SYM(m.group(2))
            assert False # unreachable
        return 'no match'

b = BNFParser()
with open('python.psr') as f:
    text = f.read()
with mlb.debug():
    res = b.parse_file(text)
breakpoint()


