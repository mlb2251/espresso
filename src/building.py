
# b.build(expr) works for AST nodes. non python nodes have their translate() methods called, which may do some building of its own! eg forward declarations


rapid tree traversal to find all instances of a node for example, etc
useful for forward definitions of values, all of which can be stored in some master object __x.whatever so they dont conflict with namespace.


b.gen(fparseablestring)

# you write a `build` for each Expr or Stmt, rather than a monolithic match stmt. There's total sugar similar to AST in appearance that desugars to this setup.
# then b.stmt() automatically tries all Stmts and b.expr automatically tries all exprs

# macros should be resolved first by direct substitution before anything (mangling to ensure)


#`whatever` becomes b.gen(ff'whatever') which is basically p.parse(f'whatever')
# build is the same as b.build()

b.build(pynode) will build a python node
b.build(customnode) will build a pynode and possibly some extra pynodes
# there is no b.translate because it would have side effects by nature eg forward declarations

# PythonParser(Parser).parse(): test.py -> pyAST
# PythonBuilder(Builder).build(): pyAST -> out.py # inherit from Builder

# CBuilder(Builder).build(): cAST -> out.c # inherit from Builer

# BNFParser(Parser).parse(): test.my -> myAST
# BNFBuilder(PythonBuilder).build(): myAST -> out.py # inherit from PythonBuilder

# if you're building in a language you generally wanna inherit their builder.
# b.build(node) -> bool, with side effect of generating text
# b.gen(fstr,locals) basically just does build(parse(str))
# b.def(name=None) is a PythonBuilder contextmanager

# super().gen('return True')
# b.gen(f'lower_name ::= \n\t /whatever/')
# hmmm you wanna do super().gen way more than b.gen. Do you ever even wanna do b.gen(f)? I guess probably for some things.



with b.anon_def() as fn:
    pass





## build() returns a boolean of success (ie not too important)
## build() has b.gen and b.def side effects. The root is all about bnfcall which does gen('p.whatever()')
## here we're trying ot build python code

undefined = object()

class Constructed: # anything that can be built into using a Builder with a Context pointing into this
    pass

class Container:
    pass # used at runtime at the top of the program. associated with a file

class Function(Constructed): # represents a function that is being built
    def __init__(self,name):
        self.name = name
    def append_ctx():
        pass
    def prepend_ctx():
        pass
    def str():
        return self.name # important for fstrings
## TODO str should actually by like f"{self.vars}.{self.name}"

class Text:
    def __init__(self):
        self.lines = []
    def append(self,line):
        assert '\n' not in line
        self.lines.append(line)
    def extend(self,lines):
        for line in lines:
            assert '\n' not in line
        self.lines.extend(lines)
    def join(self):
        return '\n'.join(self.lines)

class Context: # represents a location that a builder is building at
    def __init__(self,text,mode):
        self.text = text
        self.mode = mode
        # self.file ?? to access AnonymousContainer
    def write(self,text):
        if mode == 'append':
            self.text.append(self.text.pop()+f'{text}')
        else:
            raise NotImplementedError
    def write_line(self,text):
        if mode == 'append':
            self.text.append(f'{text}')
        else:
            raise NotImplementedError

class File(Constructed): # represents a file
    pass
class Class(Constructed): # represents a class
    pass

# BNFBuilder is trying to build a Parser class with a method for each BnfFn


def Builder:
    def __init__(b,ctx):
        b.contexts = [ctx]
        b.vars = Variables()
    def enter_ctx(b,ctx):
        b.contexts.append(ctx)
    def exit_ctx(b):
        b.contexts.pop()
        assert len(b.contexts) > 0

    @contextmanager
    def def(b,name=None):
        fn = Function()
        ctx = fn.append_ctx
        b.enter_ctx(ctx)
        try:
            yield None
        finally:
            b.exit_ctx()
            return fn

    @property
    def ctx(b):
        return b.contexts[-1]

class Variables:
    pass

class BNFBuilder(Builder):
    def __init__(b,ctx):
        super().__init__(ctx)
        b.pyparse = PythonParser()
        b.pybuild = PythonBuilder()
    def gen(b,text): # parse as python
        ast = b.pyparse.parse(text)
        assert b.build(ast) is True
    def build(b,node):
        if b.pybuild.build(node):
            return True # successfully built node with parent
        elif isinstance(node,Or):
            left,right = node.tuple
            with b.anon_def() as left_fn:
                b.build(left)
            with b.anon_def() as right_fn:
                b.build(right)
            b.gen(f"{b.vars}.logical_or({right_fn},{left_fn})")
        elif isinstance(node,PdefFn):
            text, = node.tuple
            b.gen(f"{text}")
        elif isinstance(node,Maybe):
            items, = node.tuple
            with b.def() as fn:
                b.build(items)
            b.gen(f"{b.vars}.maybe({fn})")
        elif isinstance(node,KleenePlus):
            items, = node.tuple
            with b.def() as fn:
                b.build(items)
            b.gen(f"{b.vars}.kleene_plus({fn})")
        elif isinstance(node,KleeneStar):
            items, = node.tuple
            with b.def() as fn:
                b.build(items)
            b.gen(f"{b.vars}.kleene_star({fn})")
        elif isinstance(node,Seq):
            stmts, = node.tuple
            if len(stmts) == 1:
                b.build(stmts[0]) # simply build the expression
            else:
                with b.def() as fn:
                    for stmt in stmts:
                        b.build(stmt)
                b.gen(f"{fn}()")
        elif isinstance(node,BnfCall):
            name,args = node.tuple
            b.gen(f"p.{name}({args})")
        else:
            return False # unable to build node
        return True # successfully built node


BnfCall(name,args):
    return `p.{name}({args})`


KleeneStar(expr):
    with b.def() as fn:
        b.build(expr)
    return `{loop}({fn})` # able to reference `loop` thats defined in this file

KleenePlus(expr):
    with b.def() as fn:
        b.build(expr)
    with b.def() as fn2:
        b.build(```
        res = len({loop}({fn}))
        if len(res) == 0:
            raise SyntaxError
        return res
        ```)
    return `{fn2}()` # able to reference `loop` thats defined in this file

def kleene_star(fn):
    res = loop(fn)
    return len(res)

def kleene_plus(fn):
    res = loop(fn)
    if len(res) == 0:
        raise SyntaxError
    return len(res)

def maybe(fn):
    try:
        return fn()
    except SyntaxError:
        pass
    return None

def logical_or(*fns):
    for fn in fns:
        try:
            return fn()
        except SyntaxError:
            pass
    raise SyntaxError

def loop(fn):
    result = []
    while True:
        try:
            result.append(fn())
        except SyntaxError:
            return result

Asn(name,expr):
    val = b.translate(expr) # val = translate expr in future
    with b.def() as fn: # fn(): in future
        b.build(`p.ctx['{name}'] = {val}`) # build `...` in future
    b.build(`{fn}()`) # build `...` in future



