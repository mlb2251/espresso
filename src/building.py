
# b.desugar(expr) works for AST nodes. non python nodes have their translate() methods called, which may do some desugaring of its own! eg forward declarations


rapid tree traversal to find all instances of a node for example, etc
useful for forward definitions of values, all of which can be stored in some master object __x.whatever so they dont conflict with namespace.


b.gen(fparseablestring)

# you write a `desugar` for each Expr or Stmt, rather than a monolithic match stmt. There's total sugar similar to AST in appearance that desugars to this setup.
# then b.stmt() automatically tries all Stmts and b.expr automatically tries all exprs

# macros should be resolved first by direct substitution before anything (mangling to ensure)


#`whatever` becomes b.gen(ff'whatever') which is basically p.parse(f'whatever')
# desugar is the same as b.desugar()

b.desugar(pynode) will desugar a python node
b.desugar(customnode) will desugar a pynode and possibly some extra pynodes
# there is no b.translate because it would have side effects by nature eg forward declarations

# PythonParser(Parser).parse(): test.py -> pyAST
# PythonDesugarer(Desugarer).desugar(): pyAST -> out.py # inherit from Desugarer

# CDesugarer(Desugarer).desugar(): cAST -> out.c # inherit from Builer

# BNFParser(Parser).parse(): test.my -> myAST
# BNFDesugarer(PythonDesugarer).desugar(): myAST -> out.py # inherit from PythonDesugarer

# if you're desugaring in a language you generally wanna inherit their Desugarer.
# b.desugar(node) -> bool, with side effect of generating text
# b.gen(fstr,locals) basically just does desugar(parse(str))
# b.def(name=None) is a PythonDesugarer contextmanager

# super().gen('return True')
# b.gen(f'lower_name ::= \n\t /whatever/')
# hmmm you wanna do super().gen way more than b.gen. Do you ever even wanna do b.gen(f)? I guess probably for some things.



with b.anon_def() as fn:
    pass





## desugar() returns a boolean of success (ie not too important)
## desugar() has b.gen and b.def side effects. The root is all about bnfcall which does gen('p.whatever()')
## here we're trying ot desugar python code

undefined = object()

class Constructed: # anything that can be built into using a Desugarer with a Context pointing into this
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

class Context: # represents a location that a Desugarer is desugaring at
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

# BNFDesugarer is trying to desugar a Parser class with a method for each BnfFn




def Desugarer:
    def __init__(b,ctx):
        b.contexts = [ctx]
        b.vars = Variables()
    def enter_ctx(b,ctx):
        b.contexts.append(ctx)
    def exit_ctx(b):
        ret = b.contexts.pop()
        assert len(b.contexts) > 0
        return ret

    @contextmanager
    def context(self,ctx):
        self.enter_ctx(ctx)
        try:
            yield None
        finally:
            assert self.exit_ctx() is ctx

    @contextmanager
    def anon_def(self):
        fn = pyast.FuncDef(fname='__tmp_fn_name')
        return self.context(fn.body)

    @property
    def ctx(b):
        return b.contexts[-1]

class Variables:
    pass


we want to be able to enter contexts (via contextmanagers) for

# the current "context" is always a Suite and idx within that suite (?)
# b.desugar() has side effects on the current context, usually by appending stmts

when you enter build(node), self.ctx.suite == node.suite()

with b.anon_def() as fn:
    b.desugar(body)
with b.context() as fn:
    b.desugar(body)



# with self.context(self.suite):
# with self.context(

# self.desugar() desugars to 

p = BNFParser()
ast = p.parse('heres some input')
d = BNFDesugarer()
d.desugar(ast) # side effectful
ast.to_py('test.py')



class BNFDesugarer(Desugarer):
    def __init__(self):
        # TODO parse_head(text) should parse just the head of a compound fn so you can do things like p.parse_head(f'class Foo({bar})')
        self.parser_cls = pyast.ClassDef(cname='GeneratedParser',inheritance=pyast.Args(['Parser']))
        super().__init__()
        with self.context(self.global_context):
            self.append(parser_cls)
            self.vars = pyast.ClassDef(cname='Variables'))
            self.append(self.vars)
            self.append(pyast.ClassDef(cname='Variables'))
            self.append(self.parse("UNDEF = object()"))
    def desugar(b,node):
        if b.pydesugar.desugar(node):
            return True # successfully built node with parent
        elif isinstance(node,Or):
            left,right = node.tuple
            with b.anon_def() as left_fn:
                b.desugar(left)
            with b.anon_def() as right_fn:
                b.desugar(right)
            return 
            b.gen(f"{b.vars}.logical_or({right_fn},{left_fn})")
        elif isinstance(node,PdefFn):
            text, = node.tuple
            fdef = pyast.parse(text,pyast.FuncDef)
            self.parser_cls.body.append(fdef)
            ### or equivalently:
            ##with self.context(self.parser_cls):
            ##    self.append(fdef)
        elif isinstance(node,BnfFn):
            name,argnames,productions,decos = node.tuple
            fdef = FuncDef(fname=name)
            productions = []
            for production in productions:
                items,rhs = production.tuple
                with self.anon_def() as fn:
                    self.build(items)
                    if rhs is None:

                    
        elif isinstance(node,Maybe):
            items, = node.tuple
            with b.def() as fn:
                b.desugar(items)
            b.gen(f"{b.vars}.maybe({fn})")
        elif isinstance(node,KleenePlus):
            items, = node.tuple
            with b.def() as fn:
                b.desugar(items)
            b.gen(f"{b.vars}.kleene_plus({fn})")
        elif isinstance(node,KleeneStar):
            items, = node.tuple
            with b.def() as fn:
                b.desugar(items)
            b.gen(f"{b.vars}.kleene_star({fn})")
        elif isinstance(node,Seq):
            """
            build(Seq) -> Expr which may be a function call which may return UNDEF (eg this happens for a sequence longer than length 1)
            """
            stmts, = node.tuple
            if len(stmts) == 1:
                return b.desugar(stmts[0]) # simply desugar the expression
            with b.anon_def() as fn:
                for stmt in stmts:
                    b.desugar(stmt)
                b.append(pyast.parse("return {UNDEF}",pyast.Return))
            return b.parse(f"{fn.path}()",pyast.Call)
        elif isinstance(node,BnfCall):
            name,args = node.tuple
            b.gen(f"p.{name}({args})")
        else:
            return False # unable to desugar node
        return True # successfully built node


BnfCall(name,args):
    return `p.{name}({args})`


KleeneStar(expr):
    with b.def() as fn:
        b.desugar(expr)
    return `{loop}({fn})` # able to reference `loop` thats defined in this file

KleenePlus(expr):
    with b.def() as fn:
        b.desugar(expr)
    with b.def() as fn2:
        b.desugar(```
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


class Maybe(Brancher):
    def assign(vars_dict):
        pass


Asn(name,expr):
    val = b.translate(expr) # val = translate expr in future
    with b.def() as fn: # fn(): in future
        b.desugar(`p.ctx['{name}'] = {val}`) # desugar `...` in future
    b.desugar(`{fn}()`) # desugar `...` in future



