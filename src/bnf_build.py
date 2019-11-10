

#TODO jump down to if isinstance(node,BnfFn): where i do apply_rhs and fix all taht stuff up. We're gonna need to return stuff in case of no rhs etc. Write the apply_rhs function and all that.


# I think non backtracking star is fine for kleene in BNF

# as_stmt converts Exprs to exprstmts and does nothing to Stmts
# get_ref(str) is like `!str` but for outside. It returns

# state has the current program and all extra associated variables


class BNFBuilder(Builder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.seq_fns = []
    def new_fn(self,suite):
        self.seq_fns.

    def build(b,node):
        # All Atoms return Python Expr nodes which yield whatever value theyll give when using Asn with them. Some have side effects of defining functions which are done through build() in place. Most don't actually call build() since they just construct expression nodes and return them. Seq with >1 item and Asn can't be assignment targets. Seq defines a function then returns the function name as a Var node which can then be called.
        if isinstance(node,Atom):
            if isinstance(node,BnfCall):
                return `p.$(node.name)($(node.args))`
            if isinstance(node,Or):
                fnames = [build(node.left),build(node.right)]
                return `dict.update(or_seqs($fnames))` # note Asn(Or) never happens (it gets rearranged) so its ok for Or to not return anything
            if isinstance(node,Maybe):
                fname = build(node.seq)
                return `maybe_seq($fname,dict)`
            if isinstance(node,Quoted):
                val = node.val
                return `p.keyword($val)`
            if isinstance(node,Regex):
                # control chars:
                # d=re.DOTALL
                # m=re.MULTILINE
                # single digit numbers = capture groups to return
                flags = 0
                groups = []
                for char in node.control:
                    if char == 'd':
                        flags |= re.DOTALL
                    elif char == 'm':
                        flags |= re.MULTILINE
                    elif char.isdigit():
                        groups.append(int(char))
                if groups == []:
                    groups = [0] # only the full-string group, group(0)
                reg = re.compile(node.re,flags=flags)
                return `get_regex_groups(p.match(reg),groups)` # TODO write p.match
            if isinstance(node,KleenePlus):
                fn = build(node.seq) # fn:Var
                return `kleene_plus($fn,dict)`
            if isinstance(node,KleeneStar):
                fn = build(node.seq) # fn:Var
                return `kleene_star($fn,dict)`
            if isinstance(node,Asn):
                if len(node.seq.items) != 1:
                    raise SyntaxError(f"Can't assign to sequence of length not equal to 1. Length: {len(node.seq.items)}")
                val = node.seq.items[0]
                name = node.name
                if isinstance(val,Asn):
                    raise SyntaxError("Can't do Asn(Asn(...))")
                if isinstance(val,Or): # rearrange v:(a|b) to (v:a|v:b)
                    return build(Or(Asn(name,val.left),Asn(name,val.right)))

                rhs = build(val) # build into a pynode
                return `dict[$name] = $rhs`
            if isinstance(node,Seq):
                suite = Suite([as_stmt(build(item)) for item in node.items])
                ```
                def !seq():
                    dict = SADict()
                    $suite
                    return dict
                ```
                return `!seq` # return the function (not fn call) as a Var

            if isinstance(node,MacroCall):
                # MacroCalls are executed at compile time -- as in right now!
                # Important/confusing: MacroFns used BnfCalls as placeholders for their macro arguments, so when executing a macrocall we just replace all the BnfCall nodes that use our argnames with the Seqs passed in as args for those argnames.
                # TODO v imp get_macro_fn_copy needs to clone the macrofn
                macro_fn = b.get_macro_fn_copy(node.name) # TODO b.get_macro_fn

                # zip together argnames with args
                if len(node.args) != len(macro_fn.argnames):
                    raise SyntaxError("wrong number of arguments for macro call")
                args = {name:val for name,val in zip(macro_fn.argnames,node.args)}

                def helper(node):
                    if not isinstance(node,BnfCall):
                        return node # do nothing if not bnfcall
                    if node.name not in args:
                        return node # do nothing if this is a legit bnfcall
                    return args[node.name] # replace the bnfcall with the value of the macro arg

                modded = tree_apply(macro_fn,helper)
                # now just change the class to be a bnffn instead of a macrofn:
                new_bnffn = BnfFn(modded.name, modded.argnames, modded.rules, modded.decos)

                b.build(new_bnffn) # TODO not sure
                return ``

        if isinstance(node,Fn):
            if isinstance(node,BnfFn):
                name = node.name
                fns = []
                for prod in node.rules:
                    # build(prod.seq) returns a fn that returns a dict
                    # then apply_rhs(fn,rhs) returns a fn that calls fn() then uses that dict to fill in values in the rhs expression which comes after "->" in bnf rules.
                    fns.append(apply_rhs(build(prod.seq),rhs)) # TODO write apply_rhs
                ```
                def $name():
                    return p.logical_or($fns)
                ```
                b.add_parser_method(name,fn)
            if isinstance(node,PdefFn):
            if isinstance(node,MacroFn):
                pass # we dont actually have to build MacroFns!!!! Tree form is all we need for MacroCall
    def add_parser_method(self,name,fn):
        if hasattr(self.parser,node.name):
            raise SyntaxError(f"Name '{node.name}' already defined by Parser class")
        setattr(self.parser,node.name,fn)


def tree_apply(node,fn): # apply a fn to every node in the tree, recursively. Note it applies to children before parents (easy to reverse if desired)
    for k,v in vars(node):
        if isinstance(v,Node):
            setattr(node,k,tree_apply(v))
    return fn(node)


class SADict(dict):
    def __setitem__(self,key,val):
        if key in self:
            raise KeyError(f"Can only assign once to a single assignment dict. Key: {key}")

def get_regex_groups(match,groups):
    if len(groups) == 1:
        return match.group(groups[0])
    else:
        return [match.group(i) for i in groups]

def kleene_plus(fn,dict):
    ret = kleene_star(fn)
    if ret == 0:
        raise SyntaxError
    return ret

def kleene_star(fn,dict):
    list_of_dicts = []
    n = 0
    while True:
        try:
            list_of_dicts.append(fn())
            n += 1
        except SyntaxError:
            break

    # get a list of what keys show up
    keys = set()
    for d in list_of_dicts:
        keys |= d.keys() # "|" is set union

    ret = {k:[] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            ret[k].append(d.get(k,None)) # fill in with None if missing

    dict.update(ret)
    return n

def maybe_seq(fn,dict): # returns a bool and has side effects on a dict
    try:
        dict.update(fn())
        return True
    except SyntaxError:
        return False

def or_seqs(*fns): # returns a dict (note side effects would require rollback so this is a much easier solution!)
    for fn in fns:
        try:
            return fn() # returns a dict
        except SyntaxError:
            pass
    raise SyntaxError("All branches of Or failed")


def while_stmt():
    def rule1():
        # "while" cond:expression ":" body:suite ["else" ":" else_body:suite]
        # -> While(cond,body,else_body)
        def seq1():
            dict = SADict()
            p.keyword("while")
            dict['cond'] = p.expression()
            p.keyword(":")
            dict['body'] = p.suite()
            dict.update(maybe(seq2))
            return dict
        def seq2():
            dict = SADict()
            p.keyword("else")
            p.keyword(":")
            dict['else_body'] = p.suite()
            return dict
        return While(dict['cond'],dict['body'],dict['else_body'])









