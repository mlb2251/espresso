# CODEGEN
# OVERVIEW:
# class Tok: an enum of the possible tokes, eg Tok.WHITESPACE
# class Token: a parsed token. has some associated data
# class AtomCompound, etc: These Atoms are AST components. Each has a .gentext() method that generates the actual final text that the atom should become in the compiled code
# parse() is the main function here. It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
from collections import namedtuple
from keyword import kwlist
keywords = set(kwlist)


#TODO fig out missing colon errors ahead! You prob have to do it during the atomize step while handling an IndentationError

#TODO dont spend any more time on LocTagged


#TODO: currently comment lines are included in StmtLists, and trailing comments on the ends of lines are also included in the lines. In both cases `AComment` is used. I suppose this is all good because it's useful to have comments in the AST for future projects.
#TODO: newline escaping with a \ as the final character of a line


#TODO should prob have py{} and (py ) to mirror the sh ones.
#TODO should also have sh''' ... ''' syntax for truly verbatim sh
#TODO hmm think all this thru before doing it. Prob want just one syntax really.


#TODO be careful to use .verbatim on the actual QUOTE Tok that opened a verbatim quote block bc that .verbatim includes whatever trailing whitespace it had, which is an important part of the actual quote.


# TODO in order to get debug info on line numbers for an unfinished atom just call .finish() then the .loc will be populated. So you prob wanna do this kind of thing during exception handlers for useful debug info

# kill all lines with comments (dedent level does NOT ever effect anything). Note that line numbers are still important.
# Tokenize by regex -> [Tok] *modify to include line numbers
# Atomize by a linear pass -> [Atoms] *include line numbers
#   "",'',(),{},[]
#   Additionally colons ':' and commas ','
#   Colons that are NOT at the end of a line group till the end of the line.
#   Colons that ARE at the end of the line group till the next line with nonwhitespace dedented to the right amt or even more dedented
#   Commas are the only divider within a paren group and nothing beats them in precedence
#   Flag all keywords at the same time, and save pointers to all instances of each of them in a dict
#   Throw out all newlines within (), {}, []. Now every line is a statement.
#   Colon groups are lists of statements, and they are the ONLY list of statements besides the master list
#   Semicolons merely become newlines with same indent level


# Atomization pass: takes a token list and outputs an atom tree, which is basically like a very coarse AST that just has broad structural info



def tokinfo(tok):
    return f"'{tok.verbatim}' at {tok.loc.start.line}:{tok.loc.start.char}"

def atomize(tokenized_lines, interpreter):
    if atomize.saved_stack is not None:
        stack = atomize.saved_stack
        atomize.saved_stack = None
    else:
        stack = [AMasterList()]

    def pop():
        stack[-1].finish()
        stack[-2].add(stack.pop())

    for linedata in tokenized_lines:
        if linedata.linetkns == []:
            continue # ignore lines that are purely comments or empty
        # Handle any dedenting
        while stack[-1].linestart_trypop(linedata) is True:
            pop()
        for tok in linedata.linetkns:
            #u.gray(tok)
            if tok.typ == stack[-1].closer: # `closer` must be defined by each atom
                pop()
            elif tok.typ in stack[-1].error_on:
                raise SyntaxError(f"Invalid token {tokinfo(tok)} when parsing contents of atom {stack[-1]}")
            elif tok.typ not in stack[-1].allowed_children: # `allowed_children` must be defined by each atom
                stack[-1].add(tok) # if we aren't allowed to transition on a token we just add the raw token
            elif tok.typ in [QUOTE1,QUOTE2]:
                stack.append(AQuote(tok, isinstance(stack[-1],ASH)))
            elif tok.typ in [SH_LBRACE,SH_LPAREN,SH]:
                stack.append(ASH(tok))
            elif tok.typ == LBRACE:
                stack.append(ABrace(tok))
            elif tok.typ == LPAREN:
                stack.append(AParen(tok))
            elif tok.typ == LBRACKET:
                stack.append(ABracket(tok))
            elif tok.typ == COLON:
                stack.append(AColonList(tok,linedata,stack[-1].pop_curr_stmt()))
            #elif tok.typ == COMMA:
            #    stack.append(ACommaList())
            elif tok.typ == PYPAREN:
                stack.append(ADollarParen(tok))
            #elif tok.typ == HASH:
                #stack.append(AComment())
        # End of line. Handle ASH newline pops and any postprocessing on lines like inserting \n tokens if needed
        while stack[-1].lineend_trypop(linedata) is True: # any necessary end of line processing
            pop()
    # Now we autodedent plenty
    while len(stack) > 1 and isinstance(stack[-1],AStmtList):
        pop()
    if len(stack) > 1:
        if interpreter:
            atomize.saved_stack = stack
            print(stack)
            return INCOMPLETE
        else:
            raise SyntaxError(f"Unclosed {stack[-1].name} expression. Full stack:\n{stack}")
    stack[0].finish()
    return stack[0]
atomize.saved_stack = None


# Keywords: ^def ^class ^if ^with ^while ^return ^import


# we value features over speed!


#def empty(toklist):
#    if len(toklist) == 0 or (len(toklist) == 1 and getattr(toklist[0],'typ',None) == WHITESPACE):
#        return True # note the None in getattr makes it false even if this is an Atom instead of a Tok bc the .typ field is missing
#    return False

class Atom(LocTagged):
    def __init__(self,start_tok):
        super().__init__()
        self.body = []
        self.start_tok = start_tok # may be None e.g. for AMasterList
        # pretty name for class
        classstr = str(self.__class__)
        self.name = classstr[classstr.index('A'):classstr.index('>')-1]
    def add(self,tok):
        assert not self.finished # we don't want any additions to a finished atom, that will mess up line numbers and such
        self.body.append(tok)
    def linestart_trypop(self,linedata):
        return False # does nothing for most atoms
    def lineend_trypop(self,linedata):
        self.add(linedata.newline_tok)
        return False
    def finish(self):
        """
        Figuring out our start and end locations in the original text. Start location is based on the token that created us for Atoms that aren't AStmtLists, for AColonList it's the start of the head, for AMasterList it's line 1 char 1.
        """

        # loc.start
        if isinstance(self,AStmtList):
            self.finish_stmt() # finish stmt, this is our last chance
            if isinstance(self,AColonList):
                self.loc_start(self.head[0])
            if isinstance(self,AMasterList):
                self.loc_start(Loc(1,1))
        else:
            self.loc_start(self.start_tok)

        # loc.end
        if len(self.body) > 0:
            self.loc_end(self.body[-1])
        else: # empty body (rare case)
            if isinstance(self,AColonList):
                self.loc_end(self.start_tok) # the ':'
            if isinstance(self,AMasterList):
                self.loc_end(Loc(1,2))
            else: # non AStmtList
                self.loc_end(self.start_tok)

        self.finished = True # must come after self.finish_stmt()
    def __repr__(self):
        body = ' '.join([repr(elem) for elem in self.body])
        return u.mk_b(f"{self.name}(") + f"{body}" + u.mk_b(")")

class AStmtList(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.curr_stmt = []
        self.closer = None # only gets closed by the linestart_trypop function
    def finish_stmt(self):
        assert not self.finished
        if len(self.curr_stmt) > 0: # we ignore blank lines and pure comment lines
            self.body.append(self.curr_stmt)
            self.curr_stmt = []
    def add(self,tok):
        if getattr(tok,'typ',None) is SEMICOLON: # this will drop to 'else' case for non Toks like LPAREN too thanks to the None
            self.finish_stmt()
        else:
            self.curr_stmt.append(tok)
    # so that AColonList can steal its lhs from us if it wants
    def pop_curr_stmt(self):
        tmp = self.curr_stmt
        self.curr_stmt = []
        return tmp
    def lineend_trypop(self,linedata):
        self.finish_stmt()
        return False
    def __repr__(self):
        return self.repr_depth(depth=0)
    def repr_depth(self,depth=0):
        head_indent = '  '*depth
        block_indent = '  '*(depth+1)
        res = f'{head_indent}{self.name}'
        if hasattr(self,'head'):
            res += '(' + ' '.join([repr(elem) for elem in self.head]) + ')'
        res += ':'
        for line in self.body:
            res += f'\n'
            if len(line) == 1 and isinstance(line[0],AStmtList):
                res += line[0].repr_depth(depth=depth+1) # print stmt list
            else:
                res += f'{block_indent}' # only need for nonstmtlists bc stmtlists handle indents themselves
                for elem in line: # print a line full of non-stmtlist elements
                    assert not isinstance(elem,AStmtList), "AStmtList should only appear as the only element on a line if it appears"
                    res += repr(elem) + ' '
                res = res[:-1] # kill last space
        return res

class AMasterList(AStmtList):
    def __init__(self):
        super().__init__(None) # There is no token that created AMasterList
        self.allowed_children = COMMON_ALLOWED_CHILDREN | {COLON} # set union
        self.error_on = COMMON_ERROR_ON
    def linestart_trypop(self,linedata):
        if linedata.leading_whitespace != '':
            raise IndentationError(f"unexpected indent: should have zero indent")
        return False # never pops

SEMICOLONS_ONLY = -1

class AColonList(AStmtList):
    def __init__(self,tok,linedata,head):
        super().__init__()
        self.prev_leading_whitespace = linedata.leading_whitespace # this is the indent on the line ending with a ':', so it's not actually the indent level of the block
        self.leading_whitespace = None
        self.head = head
        self.allowed_children = COMMON_ALLOWED_CHILDREN | {COLON}
        self.error_on = COMMON_ERROR_ON
    def linestart_trypop(self,linedata):
        if self.leading_whitespace is SEMICOLONS_ONLY:
            return True # pop! even if a comment is approaching, just pop

        # This will fire on every new line until the first real line of code is hit
        if self.leading_whitespace is None:
            if len(linedata.leading_whitespace) <= len(self.prev_leading_whitespace):
                raise IndentationError(f"unexpected indent: not enough indent / empty colon block")
            if len(self.body) > 0: # this means something of the form `def foo(): return 1` where there was a statment on the same line as the colon and no newline.
                self.leading_whitespace = SEMICOLONS_ONLY # in this case only semicolons can be used to separate statements and there is NO valid indent level
            self.leading_whitespace = linedata.leading_whitespace # yay, we've determined our indent level
            return False # no pop

        if linedata.leading_whitespace == self.leading_whitespace:
            return False # no pop

        if len(linedata.leading_whitespace) > len(self.leading_whitespace):
            # note that it would be misleading to say 'too much indent' because this actually fires even when the indent level is just some invalid level between two others, as described below.
            raise IndentationError(f"unexpected indent: not an existing indent level")
        return True # pop
        # if the new line is below us we can safely dedent
        # note that if the new line is an invalid level BETWEEN us and the lower valid level, then when the lower valid level (which is necessarily our direct parent since statment lists are only children of other statement lists) receieves this linestart_trypop, it will raise an indentation error. The only reason im doing this is bc i dont wanna pass the list of all current valid indents around, though that would be cleaner so you should do that.



#class AComment(Atom): # useful for future projects
#    def __init__(self):
#        super().__init__()
#        self.closer = NEWLINE # this is just an annotation, NEWLINE is never fed in. We get popped by .lineend_trypop()
#        self.allowed_children = []
#        self.error_on = []
#        def lineend_trypop(self,linedata):
#            return True

class AQuote(Atom):
    def __init__(self, tok, in_sh):
        super().__init__(tok)
        self.closer = tok.typ
        self.in_sh = in_sh # may be useful later e.g. fstrings and more
        self.allowed_children = []
        self.error_on = []

class AParen(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RPAREN
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ADollarParen(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RPAREN
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ABracket(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RBRACKET
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON
class ABrace(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.closer = RBRACE
        self.allowed_children = COMMON_ALLOWED_CHILDREN
        self.error_on = COMMON_ERROR_ON

class ASH(Atom):
    def __init__(self,tok):
        super().__init__(tok)
        self.opener = tok.typ
        self.depth = 0
        if typ == SH:
            self.eventual_closer = NEWLINE # note NEWLINE is just an annotation, in reality .lineend_trypop() will pop it
        elif typ == SH_LBRACE:
            self.eventual_closer = RBRACE
        elif typ == SH_LPAREN:
            self.eventual_closer = RPAREN
        self.closer = self.eventual_closer
        self.allowed_children = [PYPAREN,QUOTE1,QUOTE2]
        self.error_on = []
    def add(self,tok):
        super().add(tok)
        if self.opener is SH: # no depth / nesting for full line sh statements
            return
        if tok == self.opener:
            self.depth += 1
            self.closer = None # disable atomize() from popping us until we reassign this to something other than None
        elif tok == self.eventual_closer:
            self.depth -= 1
            if self.depth == 0:
                self.closer = self.eventual_closer
        assert self.depth >= 0
    def lineend_trypop(self,linedata):
        if self.closer == NEWLINE:
            return True
        return False
#class ACommaList(Atom):
#    def __init__(self):
#        self.allowed_children = COMMON_ALLOWED_CHILDREN


# fundamentally {} (and some other syntaxes) are used for expr lists and ':' are used for stmt lists


# check if the elem at index `idx` in statment `stmt` exists AND is a token AND is a keyword AND is the specific keyword `kw`
def iskeyword(stmt,idx,kw):
    if len(stmt) <= idx: # idx doesnt exist in stmt
        return False
    x = stmt[idx]
    if not isinstance(x,Tok): # must be a Tok in order to be a keyword
        return False
    if x.typ != KEYWORD:
        return False
    if x.data != kw:
        return False
    return True

def istoken(stmt,idx,typ_or_list):
    if len(stmt) <= idx: # idx doesnt exist in stmt
        return False
    x = stmt[idx]
    if not isinstance(x,Tok): # must be a Tok in order to be a keyword
        return False
    if isinstance(typ_or_list,list):
        if x.typ not in typ_or_list:
            return False
    else:
        if x.typ != typ:
            return False
    return True

# returns None if no leading keyword
def get_leading_keyword(elems):
    if len(elems) == 0:
        return None
    x = elems[0]
    if not isinstance(x,Tok):
        return None
    if x.typ != KEYWORD:
        return None
    return x.data



def stmt(elems):
    is_compound = (len(elems) == 1 and isinstance(elems[0],AColonList))
    # compound statement
    if is_compound:
        compound = elems[0] # compound is an AColonList
        kw = get_leading_keyword(compound.head)
        if kw is not None:
            if kw == 'def':
                return FuncDef(compound)
            if kw == 'if':
                return If(compound)
            if kw == 'for':
                return For(compound)
            if kw == 'while':
                return While(compound)
            if kw == 'class':
                return ClassDef(compound)
            if kw == 'with':
                return With(compound)
            if kw == 'try':
                return Try(compound)
            if kw == 'except':
                return Except(compound)
            if kw == 'elif':
                return Elif(compound)
            if kw == 'else':
                if get_leading_keyword(compound.head[1:]) == 'if':
                    return Elif(compound)
                return Else(compound)

            raise SyntaxError(f"Improper placement for '{kw}' keyword")
        raise SyntaxError(f"Unrecognized start to a colon block")
    # simple statement
    kw = get_leading_keyword(elems)
    if kw is not None:
        if kw == 'return':
            return Return(elems)
        if kw == 'pass':
            return Pass(elems)
        if kw == 'raise':
            return Raise(elems)
        if kw in ['import','from']:
            return Import(elems)
        if kw == 'break':
            return Break(elems)
        if kw == 'continue':
            return Continue(elems)
        if kw == 'delete':
            return Delete(elems)
        if kw == 'assert':
            return Assert(elems)
        if kw == 'global':
            return Global(elems)
        if kw == 'nonlocal':
            return Nonlocal(elems)
        raise SyntaxError(f"Improper placement for '{kw}' keyword")
    # simple statements that don't start with keywords
    e, rest = expr(elems)
    # ExprStmt
    if rest == []:
        return ExprStmt(e)
    # Asn
    if istoken(rest,0,EQ):
        return Asn(e,rest)
    # AugAsn
    if istoken(rest,0,BINOPS) and istoken(rest,1,EQ):
        return AugAsn(e,rest)
    raise SyntaxError(f"Unrecognized statment where first token is not a keyword and after parsing the first section as an expression the first token of the remainder is not EQ nor in BINOPS followed by EQ so it can't be Asn nor AugAsn.\nFirst expr parsed:{e}\nRest:{rest}")


def expr_assert_empty(*args,**kwargs):
    *ignore,elems = expr(*args,**kwargs)
    empty(elems)
    return e

"""
Returns the first valid expression that starts at elems[0] and is not the subexpression of a larger expression in `elems`, and also returns the remains of `elems`

note: don't worry if you see stuff like ABracket being translated into a dict always. That's only what happens when we ask for an expression and it START with {}. You can still make your own nondict syntax using {} as long as you don't start a generic expr with the '{' symbol. If you do want to override the kinds of things that can happen when an expr starts with '{' you are also welcome to do that, just modify the code to look harder at it and decide between Dict() and whatever your AST node is.

"""


# returns a Set or Dict as appropriate
def braced_expr(abrace):
    raise NotImplementedError



class Expr(Node):
    @staticmethod
    def identify(elems):
        raise NotImplementedError
    @staticmethod
    def build(elems):
        raise NotImplementedError

exprnodes []

def expr(elems):
    identified = []
    for node in exprnodes:
        if node.identify(elems):
            identified.append(node)





def expr(elems,usage=None,leftop=None):
    elem = elems[0]
    e = None
    # ATOMS
    if isinstance(elem,Atom):
        if isinstance(elem,AQuote):
            e = Str(elem)
        elif isinstance(elem,AParen):
            if len(elem.body) == 0:
                e = Tuple([],usage=usage)
            else:
                e = expr_assert_empty(elem.body,usage)
        elif isinstance(elem,ABracket):
            e = List(elem,usage=usage)
        elif isinstance(elem,ABrace):
            e = braced_expr(elem)
        elif isinstance(elem,ASH):
            e = Sh(elem)
        elif isinstance(elem,ADollarParen):
            e = EmbeddedPy(elem) # EmbeddedPy can just do expr_assert_empty(elem.body)
        else:
            raise NotImplementedError("{elem}")
    # TOKENS
    else:
        # LITS AND VARS
        if token(elem,[INTEGER,FLOAT,COMPLEX]):
            e = Num(elem)
        elif token(elem,ID):
            e = Var(elem,usage=usage)
        elif keyword(elem,['True','False','None']):
            e = NamedConstant(elem)
        # Anything we can figure out from the token(s) starting an expression
        # (or any further analysis. It just happens that we only need the very first token
        # to determine all of these, and anything that can't be determined by the first token
        # happens to begin with an expr as the leftmost item. If any constructs did not begin
        # with an expr nor keyword nor unique token then we would handle it here by doing
        # further analysis on `elems`
        elif token(elem,'*'):
            e,elems = Starred.init(elems,usage=usage) # note that more than just `elem` is needed here
        elif token(elem,UNOPSL):
            e,elems = UnopL.init(elems) # more than just first elem needed
        elif keyword(elem,'lambda'):
            e,elems = Lambda.init(elems)
        elif keyword(elem,'yield'):
            e,elems = Yield.init(elems)
        else:
            raise NotImplementedError("{elem}")
    """
    To this point we have parsed everything short of left-recursive expression, in particular:
        > Binop, Boolop, Compare, Call, Ternary, Attr, Subscript, ListComp
    However left-recursions have a fatal flaw that makes them easy to capture: they can't recurse forever, and eventually their lefthand expr must be one of the expressions we've already parsed.
    Furthermore, at this point the only way to parse a larger expression is through a left-recursive expression, because we already have used a parser to produce an expression `e` and therefore in order to extend this expression the extended expression must have the Expr `e` as its leftmost component and thus by definition the extended expression is left-recursive.
    """

    raise NotImplementedError # TODO at this point trim `elems` to have dealt with consuming whatever was needed to make `e`


    #TODO fig out how to handle returning `elems` since __init__ can't do it, and to indicate how much of elems has been eaten


    # left-recursive
    while True:
        e_lhs = e # using the old `e` as the lhs
        if len(elems) == 0:
            break
        elem = elems[0]

        # You have a complete subexpression `e_lhs`. You see a comma. This means that you need to build a tuple. This belongs in the `while` loop bc Tuples are effectively left-recursive so they could show up at any point once you have a subexpression.
#        if token(elem,','):
#            if no_tuple_recursion:
#                return e,elems # return, with leading comma included so Tuple.init knows there's more to come.
#            else:
#                e_lhs,elems = Tuple.init(e_lhs,elems) # completely construct tuple. This completes a sub-expression and we can then move into more left-recursive items to extend it as usual.

        # ATOM
        if isinstance(elem,Atom):
            if isinstance(elem,AParen):
                op = Call
            elif isinstance(elem,ABracket):
                op = Subscript
            else:
                raise NotImplementedError("{elem}")
        # TOKEN
        else:
            if token(elem,BINOPS):
                rightop = Binop
            elif token(elem,BOOLOPS):
                rightop = Boolop
            elif token(elem,CMPOPS):
                rightop = Boolop
            elif keyword(elem,'if'):
                rightop = Ternary
            elif token(elem,'.'):
                rightop = Attr
            elif keyword(elem,'for'):
                rightop = Comprehension
            elif token(elem,','):
                rightop = Tuple
            else:
                break # the very important case where the rest of elems is not useful for constructing an extended Expr
        # found an `op`
        rightop._typ = elem # needed by `precedence`
        if precedence(leftop,rightop) is LEFT:
            break # left op is more tight so we return into our caller as a completed subexpr
        else:
            kw = {'usage':usage} if isinstance(rightop,Targetable) else {}
            e = rightop.init(e_lhs,elems,leftop=rightop, **kw)

    return e,elems

LEFT = CONSTANT()
RIGHT = CONSTANT()

# takes an operator to your left and an operator to your right and tells you which one applies to you first (binds more tightly). The None operator always loses. `left` and `right` are Node classes

precedence_tables = [
    ['()','[]','.'], # righthand unops (call, subscript, dot)
    [],
    [],
    [],
]


precedence = [  (UnopR,Call,Subscript,Attr),
                (UnopL


precedence_table = {
        'unopsr': [Call,Subscript,Attr],
        'unopsl': [Unop],
}

precedence_table = [
[Call,Subscript,Attr], # Tightest binding (& left to right within a level
[UNOPSL],
[ ],
[ ],
[ ], # Weakest binding
]


# Binop, Boolop, Compare, Call, Ternary, Attr, Subscript, ListComp
def precedence(left,right):
    raise NotImplementedError



# TODO note that the .usage of things often isn't known when they're created initially, and is rather added during left-recursion when the target becomes part of a larger statement. So it should really be up to the larger statement to update the .usage for its targets. In other cases it is known when you call expr() for example when already inside a larger statement and calling expr to construct a nonleftrecursive smaller expr. Also ExprStmt for example could set things to LOAD, etc.

class Tuple(Node):
    def init(self,e_lhs,elems):
        self.vals = [e_lhs]
        more = True
        while token(elems,','): # `more` means expr() ended on a comma last time
            e,elems = expr(elems,no_tuple_recursion=True)
            self.vals.append(e)
        return self,elems


"""
The issue with Tuples, and the solution:
    The Tuple constructor must necessarily call expr() somewhat recursively in order to parse the expressions that make it up.
    An initially promising idea is to have the Tuple.init(e_lhs,elems) function call expr(elems) to recurse just as Binops, Boolops, and all other left-recursive forms have been doing. Say we're parsing expr("x,y,z") then the first call would be Tuple.init("x",",y,z") which would result in a new expr call expr("y,z"), which would subsequently yield an expr("z") after yet another Tuple.init. It's obvious that expr("y,z") should return a Tuple and expr("z") should return a Var. So logically Tuple.init() could use the result of its expr() call to A) if the result is a tuple then prepend whatever your lhs is onto the Tuple to create a tuple of slightly larger size and B) if the result is not a tuple then for a new 2-tuple from your lhs and the expression result. The issue with this: what if the input was expr("x,y,(a,b)")? Then the expr("a,b") call would return a Tuple whic would get merged with the lhs ("y") to form a 3-tuple y,a,b.
    The solution: add a flag to expr so that it doesn't recurse on tuple-building. Yep it's lame, really the interesting bit was the way that the initially promising idea was wrong.
"""




"""
The following method is guaranteed to work for binary and unary operators of the form:
    | Expr op Expr # binop
    | Expr op      # right unop
    | op Expr      # left unop
Following the example `x + 5*z.y - 3`
1. Read elems until you have the smallest possible valid Expr e that starts from the first elem
    e=`x`. We may be fed an operator `prevop` by our caller. If we aren't then `op` is None.
2. Check if the next elem(s) indicate that e could be a lhs subexpression
    `+` is part of the rule [Expr:==Expr '+' Expr]
    Additionally `+` isn't part of any other syntax that has an lhs Expr immediately to its left
3.1 If False simply return e.
3.2 If True and `op` is lower precedence than `prevop`, return e
3.3 If True and `op` is higher precedence than `prevop` (or prevop is None), call expr() on the remaining expression and your final result is the the result of unify(e,op,expr())
*if at any point you fail, simply return the last valid expression you had that starts at the first elem

Non unop/binop expressions:
    | various literals / vars -> trivial
    | Lists/tuples/sets/dicts -> easy recognize, known exact Expr locs
    | comprehension -> easy recognize, known exact Expr locs
    | ternary -> easy recognize, known exact Expr locs
    | lambda -> easy recognize, known exact Expr locs
    | yield -> easy recognize, known exact Expr locs


"""





    # Tuple
    # Note: An empty pair of parentheses yields an empty tuple object
    # A trailing comma creates a 1-tuple
    # a lack of commas yields the expr (non tuple)


"""

Var List Tuple Starred Attr Subscript should assert that .usage is not None when provided


Non unop/binop expressions:
    | various literals / vars -> trivial
    | Lists/tuples/sets/dicts -> easy recognize, known exact Expr locs
    | comprehension -> easy recognize, known exact Expr locs
    | ternary -> easy recognize, known exact Expr locs
    | lambda -> easy recognize, known exact Expr locs
    | yield -> easy recognize, known exact Expr locs

commas look out for. They turn you into a tuple with implied parens i think. In fact in the case of parens starting an expression i literally just call expr() on aparen.body and leave it up to expr() to return me a tuple if appropriate

"""









#TODO considerations:
    # make the .name field of FuncDef a Var in STORE mode?
    # same with a LOT of other `str`s in here
"""

===Stmts===
FuncDef:
    .name: str
    .args: Args
    .body: [Stmt]
    .decorators: [Var]
If:
    .cond: Expr
    .body: [Stmt]
    .elifs:[Elif]
    .else: Else # empty list if no else
Elif: # only present inside If
    .cond: Expr
    .body: [Stmt]
Else: # only present inside other stmts
    .body: [Stmt]
For: for i
    .target: [Target] # specifically: Var | Tuple | List
    .iter: Expr
    .body: [Stmt]
    .else: [Stmt] # empty if no else
While:
    .cond: Expr
    .body: [Stmt]
    .else: [Stmt] # empty if no else
Return:
    .val: Expr
ClassDef:
    .name: str
    .bases: [Var] # base clases to inherit from
    .keywords: [Keyword] # e.g. the 'metaclass' keyword
    .body: [Stmt]
    .decorators: [Var]
Raise:
    .exception: Expr | None
    .cause: Expr | None # where Expr is another exception.
      # (niche, prints an error message that blames .exception happening
      # on the fact that the prior .cause exception happened)
Import:
    .importitems: [Importitem]
    .from: str | None
    .numdots: int
      # from ..a import b          #.val=[(.var=b,.alias=None)],                  .from='a'  .numdots=1
      # from ... import b as c, d  #.val=[(.var=b,.alias=c),(.var=d,.alias=None)] .from=None .numdots=3
      # import b                   #.val=[(.var=b,.alias=None)],                  .from=None .numdots=0
  Importitem(namedtuple)
      .item: str
      .alias: str | None
Break:
    [no fields]
Pass:
    [no fields]
Continue:
    [no fields]
Delete:
    .targets: [Reference]
Assert:
    .cond: Expr
    .msg: Str
    .fn: Expr  # this is not pythonic, but im adding it :)
Try:
    .body: [Stmt]
    .excepts: [Except]
    .else: [Stmt] # empty if none
    .finally: [Stmt] # empty if none
Except: # only ever present inside a Try node
    .type: Var | None # None is catch-all
    .name: str | None
    .body: [Stmt]
Global:
    .names: [str]
Nonlocal:
    .names: [str]
With:
    .withitems: [withitem]
    .body: [Stmt]
      # with a as b, c as d, e
  Withitem(namedtuple):
      .contextmanager: Expr
      .targets: Reference # specifically Var | Tuple | List
        # the result of evaluating .contextmanager is assigned to .variables or something i think (if they arent None)
ExprStmt:
    .expr: Expr
Asn:
    .targets: [Reference]
    .val: Expr
AugAsn:
    .target: Name | Subscript | Attribute (not Tuple | List)
    .op: BINOP
    .val: Expr
Module:
    .body: [Stmt]


===Exprs===
Num:
    .val: int|float|complex
    .type= int|float|complex # the actual classes int()/etc
Str:
    .val: str
Bytes:
    .val: bytes
List:
    .vals: [Expr]
    .usage: USAGE
Tuple:
    .vals: [Expr]
    .usage: USAGE
Set:
    .vals: [Expr]
Dict:
    .keys: [Expr]
    .vals: [Expr]
Ellipsis:
    [no fields]
      # for the '...' syntax used extremely rarely in indexing
NamedConstant:
    .val= None|True|False
Var:
    .name: str
    .usage: USAGE
Starred:
    .val: Var | some other options
    .usage: USAGE
Unop:
    .op: UNOP
    .val: Expr
Binop:
    .op: BINOP
    .left: Expr
    .right: Expr
Boolop:
    .op: BOOLOP
    .vals: [Expr]
      # `a or b or c` is collapsed into one Boolop
Compare:
    .ops: [CMPOPS]
    .vals:[Expr]
    # e.g. `a > b > c`
Call:
    .func: Name | Attr | some other stuff
    .args: [Expr]
    .keywords: [Keyword]
  Keyword(namedtuple):
      .name: str
      .val: Expr
Ternary:
    .cond: Expr
    .if: Expr
    .else: Expr
Attr:
    .expr: Expr
    .attr: str
    .usage: USAGE
Subscript:
    .expr: Expr
    .slice: Index | Slice | ExtSlice
    .usage: USAGE
  Index(namedtuple):
      .val: Expr
  Slice(namedtuple):
      .start: Expr
      .stop: Expr
      .step: Expr
  ExtSlice(namedtuple):
      .dims: [Slice|Index]
Comprehension:
    .expr: Expr
    .comprehensions: [Comprehension]
      # sadly [ord(c) for line in file for c in line] is valid Python
  Comprehension(namedtuple):
      .target: Name | Tuple | other things
      .iter: Expr # the thing we iterate over
      .conds: [Expr] # can have mult in a single comprehension
Lambda:
    .args: [Arg] # cant take kwargs etc i think
    .body: Expr # i may extend this to allow same-line statements.
Arg(Dependent):
    .name: str
Args(Dependent):
    .args: [Arg]
    .defaults: [Expr] # specifically corresponding to the last len(.defaults) positional args (bc disallowed to put any defaultable args before required args)
    .stararg: Arg # *arg. It's an Arg('') for a raw '*' or an Arg('name') for '*name', and it's None if no star is present at all.
    .doublestararg: Arg  # **kwargs
    .kwonlyargs: [Arg]
    .kwdefaults: [Expr] # for kwonlyargs. if one is None then that kwarg is required. Unclear what non-kwonly defaults are...
      # def f(a, b=1, c=2, *d, e, f=3, **g)
      # a,b,c -> .args
      # 1,2 -> .defaults # note how defaults[i] corresponds to args[i+1] here, but more useful: defaults[-i] corresponds to args[-i]
      # d -> .vararg
      # e,f -> .kwonlyargs
      # 3 -> .kwdefaults # same deal, kwonlyargs[-i] corresponds to kwdefaults[-i]
      # g -> .kwargs
Yield: # yes, this is actually an expression
    .val: Expr  # value to yield or generator to yield from
    .from: bool # whether this is a `yield from` expression


USAGE = LOAD | STORE | DEL


Target = Var | Attr | Subscript | Slice | '*' Target
Reference = Var | Attribute | Subscript


"""

Pos = namedtuple('Pos', 'line char')
# these numbers are inclusive as start* and exclusive as end*
# so endchar is one past the last char
Loc = namedtuple('Loc', 'start end')




class LocTagged:
    def __init__(self):
        self.loc = Pos(Loc(None,None),Loc(None,None))
        self.finished = False
    def loc_start(self,other):
        if isinstance(other,Pos):
            self.loc = Loc(other,self.loc.end)
            return
        assert isinstance(other,LocTagged)
        assert other.finished
        self.loc = Loc(other.loc.start,self.loc.end)
    def loc_end(self,other):
        if isinstance(other,Pos):
            self.loc = Loc(self.loc.start,other)
            return
        assert isinstance(other,LocTagged)
        assert other.finished
        self.loc = Loc(self.loc.start,other.loc.end)
    def finish():
        # must override in subclasses to properly set `.loc` in stone
        # And you must have this at some point in the finish call:
        #   self.finished = True
        raise NotImplementedError

# assert_contiguous was thrown out bc things like Comments make stuff non contiguous
# also it's really not worth spending this much time on location stuff when the vast majority of editing will be done in the AST world.
#    @staticmethod
#    def assert_contiguous(llist):
#        """
#        takes a list of LocTagged items and ensures that there are no location gaps between them, so each one has a loc.start equal to the loc.end of the previous one.
#        """
#        for i in range(len(llist)-1):
#            if llist[i].loc.end != llist[i+1].loc.start:
#                raise LocError(f"Non contiguous LocTagged elements in list: {llist[i].loc.end} != {llist[i+1].loc.start} for \n{llist[i]} \nand \n{llist[i+1]} \nin \n {llist}")
#
#class LocError(Exception): pass


class Node(LocTagged): # abstract class, should never be instatiated
    def __init__(self,partial_stmt):
        super().__init__()
        pass
#    def finish():
#        attrs = list(self.__dict__.keys())
#        for attr in attrs:

class Module(Node):pass

class Expr(Node):pass



# elems is always the first argument
# elems is always the last thing returned (ALWAYS)
def keyword(): pass # no return
def token():
    """
    `query` is an elem or [elem]
        -if [elem] is given then the 0th index elem is used
        -if a single (nonlist) elem is given, then the return type is a bool indicating success/failure. After all, there's nothing useful to pass back to the caller since they didn't give us any list we can modify. This makes for nice if-statements too.
    `tok` is a token CONSTANT() or str
        -if a str is given then it's converted to the right token e.g. '(' -> RPAREN
        (we allow both options since INTEGER for example has no valid str form other than making up an arbirary one which is unnecessary and confusing.)
        -if a list (of CONSTANTs or strs) then matching on one token results in overall success
    `fail` = FAIL | SILENT | None | BOOL
        -Note that in all following descriptions `remains` is the rest of elems after consuming the processed token, and `original` is the original elems input.
        -None (default) - SyntaxError raised on failure. Success returns: remains
        -FAIL - returns (OK,remains) or (FAIL,original)
        -SILENT - returns remains on success and original on failure. No exception raised.
        -BOOL - returns True or False and nothing else
        -*see note in `query` about how passing in a single elem will result in a boolean return type. An error will be thrown on any attempt to specify the fail= option.


    """
    pass



def identifier(): pass
def target(): pass


def exprs(): pass # [elem] -> [Expr]
def stmts(elem_list_list): # [[elem]] -> [Stmt]
    the_stmts = []
    for elems in elem_list_list:
        the_stmt = stmt(elems)
        if the_stmt.dependent and the_stmts[-1].offer_neighbor(the_stmt):
            continue # e.g. Try consuming Except by accepting it as a neighbor
        elif the_stmt.dependent:
            raise SyntaxError(f"dependent statement {the_stmt} not accepted by neighbor {the_stmts[-1]}")
        the_stmts.append(the_stmt)

def empty(): pass # assert that the elem list given is empty, if not throw an error about trailing symbols


# STMTS

class Stmt(Node): # abstract
    def __init__(self):
        self.dependent = False
    def offer_neighbor(self, stmt):
        """
        All statements that want to consume their same-indent-level neighbors should override this, for example Try should override it to add stmt to its self.excepts if stmt is an Except node, and should return True to indicate that the stmt no longer needs to be added to the normal stmt list (bc instead it's inside of Try)
        """
        return False

class CompoundStmt(Stmt): pass # abstract
class SimpleStmt(Stmt): pass # abstract
class SynElem(Node): pass # Syntatic Element. Things like Args that are neither Exprs nor Stmts

## Stmts that take an AColonList as input
class FuncDef(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"def")
        self.name,head = identifier(head)
        self.args,head = Args(head)
        empty(head)
        self.body = stmts(compound.body)

def comma_split(elems): # [elem] -> [[elem]]
    res = []
    prev_comma_idx = -1
    for i in range(elems):
        if isinstance(elems[i],Tok) and elems[i].typ is COMMA:
            res.append(elems[prev_comma_idx+1:i])
            prev_comma_idx = i
    res.append(elems[prev_comma_idx+1:])
    return res


class Arg(SynElem):
    def __init__(self,name):
        self.name = name

class Args(SynElem):
    def __init__(self,aparen):
        super().__init__()

        items = comma_split(aparen.body)
        [_nondefaulted, _defaulted, _stararg, _kwonlynondefaulted, _kwonlydefaulted, _doublestararg] = [x for x in range(6)] # order matters, can't go backwards once you enter one mode
        err_str = { # for error messages
            _nondefaulted:'all non-defaulted arguments', _defaulted:'all defaulted arguments', _stararg:'the *arg argument', _kwonlynondefaulted:'all keyword-only-non-defaulted arguments', _kwonlydefaulted:'all keyword-only-defaulted arguments', _doublestararg:'the **kwarg argument'
                }
        position = _defaulted
        self.args = []
        self.defaults = []
        self.stararg = None
        self.kwonlyargs = []
        self.kwonlydefaults = []
        self.doublestararg = None

        # process each comma separated elemlist
        for i,elems in enumerate(items):
            if len(elems) == 0:
                if i != len(items)-1:
                    raise SyntaxError(f"two commas in a row not allowed in argument list") # TODO point out exactly where it is / suggest fix / setup vim commands for fix

            # kwonlynondefaulted or nondefaulted or stararg(no identifier)
            if len(elems) == 1: # nondefaulted arg or stararg with no identifier
                argname,elems = identifier(elems,fail=FAIL)
                # kwonlynondefaulted or nondefaulted
                if argname is not FAIL: # successfully parsed a nondefaulted arg
                    if position > _kwonlynondefaulted:
                        raise SyntaxError(f"{err_str[_kwonlynondefaulted]} must go before {err_str[position]}")
                    # kwonlynondefaulted
                    if _nondefaulted > position >= _kwonlynondefaulted:
                        position = _kwonlynondefaulted
                        self.kwonlyargs.append(Arg(argname))
                        continue
                    # nondefaulted
                    position = _nondefaulted
                    self.args.append(Arg(argname))
                    continue
                # failed kwonlynondefaulted and nondefaulted so lets try stararg with no ident
                status,elems = token(elems,'*',fail=FAIL)
                # stararg with no identifier. Note that theres no '**' without an identifier
                if status is not FAIL: # successfully parsed a stararg with no identifier
                    if position == _stararg:
                        raise SyntaxError(f"not allowed to have multiple * or *arg arguments")
                    if position > _stararg:
                        raise SyntaxError(f"{err_str[_stararg]} must go before {err_str[position]}")
                    position = _stararg
                    self.stararg = Arg('')
                    continue
                raise SyntaxError(f"unable to parse argument {elems}")
            # stararg with identifier
            if len(elems) == 2: # *arg or **arg
                status,elems = token(elems,'*',fail=FAIL)
                if status is FAIL:
                    status2,elems = token(elems,'**',fail=FAIL)
                if status is FAIL and status2 is FAIL:
                    raise SyntaxError(f"unable to parse argument {elems}")
                if status is not FAIL:
                    _mode = _stararg
                else:
                    _mode = _doublestararg

                argname,elems = identifier(elems,fail=FAIL)
                if status is not FAIL and argname is not FAIL: # successfully parsed a stararg with identifier
                    if position == _mode:
                        raise SyntaxError(f"not allowed to have multiple {'*arg' if _mode == _stararg else '**kwargs'} arguments")
                    if position > _mode:
                        raise SyntaxError(f"{err_str[_stararg]} must go before {err_str[position]}")
                    position = _mode
                    if _mode == _stararg:
                        self.stararg = Arg(argname)
                    else:
                        self.doublestararg = Arg(argname)
                    continue
                raise SyntaxError(f"unable to parse argument {elems}")
            # defaulted and kwonlydefaulted
            if len(elems) > 2:
                argname,elems = identifier(elems,fail=FAIL)
                status,elems = token(elems,'=',fail=FAIL)
                if argname is FAIL or status is FAIL:
                    raise SyntaxError(f"unable to parse argument {elems}")
                val,elems = expr(elems,fail=FAIL)
                if val is FAIL:
                    raise SyntaxError(f"unable to parse expression for default {elems}")
                status = empty(elems,fail=FAIL)
                if status is FAIL:
                    raise SyntaxError(f"trailing tokens in default. The expression was parsed as {val} but there are trailing tokens: {elems}")
                if position > _kwonlydefaulted:
                    raise SyntaxError(f"{err_str[_kwonlydefaulted]} must go before {err_str[position]}")
                # kwonlydefaulted
                if _defaulted > position >= _kwonlydefaulted:
                    position = _kwonlydefaulted
                    self.kwonlyargs.append(Arg(argname))
                    self.kwonlydefaults.append(val)
                    continue
                # defaulted
                position = _defaulted
                self.args.append(Arg(argname))
                self.defaults.append(val)
                continue

        if position == _stararg:
            raise SyntaxError(f"named arguments must follow bare *")
        return # end of __init__ for Args

class If(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"if")
        self.cond,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class CONSTANT(object):
    def __init__(self,name):
        self.name=name
        CONSTANT.instances.append(self)
        for i,instance in enumerate(CONSTANT.instances):
            for instance2 in enumerate(CONSTANT.instances[i+1:):
                assert instance != instance2
                assert instance is not instance2
                assert instance.name != instance2.name
    def __eq__(self,other):
        return self is other
    def __neq__(self,other):
        return self is not other
    def __repr__(self):
        return self.name

CONSTANT.instances = []


FAIL = CONSTANT('FAIL')
SILENT = CONSTANT('SILENT')
OK = CONSTANT('OK')

class Elif(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        status,head = keyword(head,"elif",fail=FAIL)
        if status is FAIL:
            head = keyword(head,"else")
            head = keyword(head,"if")
        self.cond = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class Else(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.dependent = True

        head = keyword(head,"else")
        empty(head)
        self.body = stmts(compound.body)


class For(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"for")
        self.target,head = target(head)
        head = keyword(head,"in")
        self.iter,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class While(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"while")
        self.cond,head = expr(head)
        empty(head)
        self.body = stmts(compound.body)

class ClassDef(CompoundStmt):pass
class With(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.withitems = []

        head = keyword(head,"with")
        items = comma_split(head)
        if items == []:
            raise SyntaxError("empty `with` statement header")
        for item in items:
            contextmanager,item = expr(item)
            keyword(item,'as')
            target,item = target(item)
            empty(item)
            self.withitems.append(Withitem(contextmanager,target))

        self.body = stmts(compound.body)
class Withitem(SynElem):
    def __init__(self,contextmanager,target):
        self.contextmanager = contextmanager
        self.target = target
class Try(CompoundStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"try")
        empty(head)
        self.body = stmts(compound.body)



## Stmts that take an elem list as input
class Return(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"return")
        self.val,head = expr(head)
        empty(head)
class Pass(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"pass")
        empty(head)
class Raise(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.exception = None
        self.cause = None

        head = keyword(head,"raise")
        if not empty(head,fail=BOOL):
            self.exception,head = expr(head)
        if not empty(head,fail=BOOL):
            head = keyword(head,'from')
            notempty(head) # TODO it's important to have guards like this since otherwise an empty expr might just eval to None which Python does not do. This is along the lines of expr_assert_empty except like a expr_preassert_nonempty except with a better name.
            self.cause,head = expr(head)
        empty(head)


# `by` and `till` strings, tokens, or functions
def dot_split(): # [elem] -> [[elem]]
    raise NotImplementedError

# [elem] -> | str         if `till` is None
#           | str,[elem]  if `till` is not None
# as usual `till` can be a string/token/function
def raw_string():
    raise NotImplementedError

class Import(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head
        self.importitems = []
        self.from = None
        """
        | import ident[.ident.ident] [as ident][, ident.ident.ident as ident, ident.ident.ident as ident]
        | from [..]ident[.ident.ident] import x [as ident][, y as ident, ident as ident]
        | from [..]ident[.ident.ident] import *
        * only `from` statments can use numdots != 0 like .mod or ..mod
        """

        # `from`
        if keyword(head,"from",fail=BOOL):
            head = keyword(head,"from")
            self.from,head = raw_string(imp,till='import')

        head = keyword(head,"import")

        # `from x import *`
        if token(head,'*',fail=BOOL):
            if self.from is None:
                raise SyntaxError("`import *` not valid, must do `from [module] import *`")
            self.importitems = ['*']
            return

        imports = split(head,',')
        for imp in imports:
            alias = None
            modstr,imp = raw_string(imp,till='as')
            if not empty(imp,fail=BOOL):
                imp = keyword(imp,'as')
                alias = identifier(imp)
            self.withitems.append(Withitem(modstr,alias))

class Importitem(SynElem):
    def __init__(self,var,alias):
        self.var=var
        self.alias=alias
class Break(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"break")
        empty(head)
class Continue(SimpleStmt):
    def __init__(self,compound):
        super().__init__()
        head = compound.head

        head = keyword(head,"continue")
        empty(head)
class Delete(SimpleStmt):
        super().__init__()
        head = compound.head
        self.targets = []

        head = keyword(head,"delete")
        items = comma_split(head)
        if items == []:
            raise SyntaxError("`delete` statement must have at least one target")
        for item in items:
            target,item = target(head,usage=DEL)
            empty(item)
            self.targets.append(target)

class Assert(SimpleStmt):
class Global(SimpleStmt):
class Nonlocal(SimpleStmt):
class ExprStmt(SimpleStmt):
class Asn(SimpleStmt):
class AugAsn(SimpleStmt):



# EXPRS
class Targetable:
    def __init__(self,elems,**kw):
        assert isinstance(self,Expr)
        if 'usage' not in kw:
            raise InheritanceError("{self} is a Targetable object and must be called with the 'usage' kwarg")
        self.usage = kw.pop('usage')



# TODO All e_lhs inputs to a constructor (left recursive) should call set_usage on e_lhs. We enforce this by having all .targetable things ensure a non-None .usage during a tree traverasal after the AST is made


class Var(Expr,Targetable):pass
class Starred(Expr,Targetable):pass
class List(Expr,Targetable):pass
class Tuple(LeftRecursive,Targetable):pass
class Set(Expr):pass
class Dict(Expr):pass
class Ellipsis(Expr):pass
class Comprehension(LeftRecursive):pass


class Lit(Expr): pass
class Num(Lit):pass
class Str(Lit):pass
class Bytes(Lit):pass
class NamedConstant(Lit):pass

class UnopL:(Expr):pass
class UAdd(UnopL):pass
class USub(UnopL):pass
class Not(UnopL):pass
class Invert(UnopL):pass


class UnopR:(LeftRecursive):pass
class Attr(UnopR,Targetable):pass
class Subscript(UnopR,Targetable):pass
class Call(UnopR):pass


class UnopL(Unop):pass


class LeftRecursive(Expr): pass

class Binop(LeftRecursive):
    @staticmethod #TODO perhaps a metaclass is the right thing for the job here, im not sure
    def build(e_lhs,):

class Add(Binop):pass
class Sub(Binop):pass
class Mul(Binop):pass
class Div(Binop):pass
class FloorDiv(Binop):pass
class Mod(Binop):pass
class Exp(Binop):pass
class ShiftR(Binop):pass
class ShiftL(Binop):pass
class BitAnd(Binop):pass
class BitOr(Binop):pass
class BitXor(Binop):pass


class Boolop(LeftRecursive):pass
class And(Boolop):pass
class Or(Boolop):pass

class Compare(LeftRecursive):pass
class Lt(Compare):pass
class Leq(Compare):pass
class Gt(Compare):pass
class Geq(Compare):pass
class Eq(Compare):pass
class Neq(Compare):pass
class Is(Compare):pass
class Isnot(Compare):pass
class In(Compare):pass
class Notin(Compare):pass



class Ternary(LeftRecursive):pass
class ListComp(Expr):pass
class Lambda(Expr):pass
class Yield(Expr):pass

# namedtuples for use in Nodes
Importitem = namedtuple('Importitem', 'var alias loc')
Withitem = namedtuple('Withitem', 'contextmanager targets loc')
Keyword = namedtuple('Keyword', 'name value loc')
Arg = namedtuple('Arg', 'name loc')
Args = namedtuple('Args', 'args defaults vararg kwarg kwonlyargs kwdefaults loc')
Comprehension = namedtuple('Comprehension', 'target iter conds loc')
Index = namedtuple('Index', 'val loc')
Slice = namedtuple('Slice', 'start stop step loc')
ExtSlice = namedtuple('ExtSlice', 'dims loc')



#def assignable(node): # checks if expression can be assigned to
#    return hasattr(node,'usage') # .usage is used to give the mode as STORE, LOAD, DEL





# proceed from outside inward:
# -first deal with statements that can hold other statements: `def` and `class`
# -next deal with statements
# -next deal with expressions

# `print(x); if True: ...` is forbidden but `print(x); return x` is fine. So you can't have colons later on a line that has a semicolon











## TODO next: make SInitial more pretty. I dont htink overloader is the way to go, we shd start in SInit then trans to Snormal not just wrap Snormal.

## assertion based coding. After all, we're going for slow-but-effective. And assertions can be commented in the very final build. This is the python philosophy - slow and effective, but still fast enough

## would be interesting to rewrite in Rust or Haskell. ofc doesn't have all the features we actually want bc in particular we should be able to override parse_args or whatever dynamically. And list of callable()s would have to be passed here.

from enum import Enum,unique
import re
from util import die,warn,Debug
import util as u
import inspect

# Tok is the lowest level thing around. 
# It's just an enum for the different tokens, with some built in regexes



OFFSET=100
# str_of_token(PERIOD) == 'PERIOD'
## important: if you add a TOKEN to this list you must also add it to the list of constants and REGEX_OF_TOKEN, and it must show up in the SAME position in the former two lists (it can show up somewhere different in REGEX_OF_TOKEN and that's fine)
_str_of_token = ['INTEGER','PERIOD','COMMA','COLON','SEMICOLON','EXCLAM','PYPAREN','SH_LBRACE','SH_LPAREN','SH','LPAREN','RPAREN','LBRACE','RBRACE','LBRACKET','RBRACKET','ADD','SUB','MUL','FLOORDIV','DIV','EXP','MOD','SHIFTRIGHT','SHIFTLEFT','BITAND','BITOR','BITXOR','INVERT','NOT','AND','OR','LEQ','LT','GEQ','GT','EQ','NEQ','IS','ISNOT','IN','NOTIN','ASN','ESCQUOTE2','ESCQUOTE1','QUOTE2','QUOTE1','HASH','PIPE','ID','UNKNOWN','SOL','EOL','NEWLINE','KEYWORD']
def str_of_token(t):
    return _str_of_token[t-OFFSET]
# order doesn't matter in this list, as long as it's the same order as in the preceding list
[INTEGER, PERIOD, COMMA, COLON, SEMICOLON, EXCLAM, PYPAREN, SH_LBRACE, SH_LPAREN, SH, LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, ADD, SUB, MUL, FLOORDIV, DIV, EXP, MOD, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR, INVERT, NOT, AND, OR, LEQ, LT, GEQ, GT, EQ, NEQ, IS, ISNOT, IN, NOTIN, ASN, ESCQUOTE2, ESCQUOTE1, QUOTE2, QUOTE1, HASH, PIPE, ID, UNKNOWN, SOL, EOL, NEWLINE, KEYWORD] = [i+OFFSET for i in range(len(_str_of_token))] # we start at 100 so that we can still safely manually define constants in the frequently used -10..10 range of numbers with no fear of interference or overlap.

CMPOPS = [LEQ,LT,GEQ,GT,EQ,NEQ,IS,ISNOT,IN,NOTIN]
BINOPS = [ADD, SUB, MUL, DIV, FLOORDIV, MOD, EXP, SHIFTRIGHT, SHIFTLEFT, BITAND, BITOR, BITXOR]
UNOPSL  = [ADD, SUB, NOT, INVERT] # yes, ADD and SUB can be unops
UNOPSR  = [] # right side unary operators. There aren't any... yet! Well actually () and [] and . are UNOPSR really.
BOOLOPS= [AND,OR]


COMMON_ALLOWED_CHILDREN = set([QUOTE1,QUOTE2,LBRACKET,LBRACE,LPAREN,SH_LBRACE,SH, SH_LPAREN, HASH])
COMMON_ERROR_ON = set([RPAREN,RBRACE,RBRACKET]) # note that closers are checked first in Atomize so it's safe to include RPAREN in the .error_on field of AParen for example. Hence the COMMON_ERROR_ON can be widely used


# compile the regex and also eat any whitespace that follows it
def regex_compile(regex):
    return re.compile(regex+'\s*')

# order is important in this list!
regex_of_token = {
    INTEGER   : regex_compile(r'(\d+)'),
    PERIOD    : regex_compile(r'\.'),
    COMMA     : regex_compile(r','),
    COLON     : regex_compile(r':'),
    SEMICOLON : regex_compile(r';'),
    EXCLAM    : regex_compile(r'!'),

    # GROUPINGS
    PYPAREN   : regex_compile(r'py\('), # BEFORE ID
    SH_LBRACE : regex_compile(r'sh{'), # BEFORE ID
    SH_LPAREN : regex_compile(r'\(\s*sh\s+'), # BEFORE ID
    SH        : regex_compile(r'sh\s+'), # BEFORE ID, AFTER SH_*
    LPAREN    : regex_compile(r'\('),
    RPAREN    : regex_compile(r'\)'),
    LBRACE    : regex_compile(r'{'),
    RBRACE    : regex_compile(r'}'),
    LBRACKET  : regex_compile(r'\['),
    RBRACKET  : regex_compile(r'\]'),

    # BINOPS
    ADD       : regex_compile(r'\+'),
    SUB       : regex_compile(r'-'),
    MUL       : regex_compile(r'\*'),
    FLOORDIV  : regex_compile(r'//'), # BEFORE DIV
    DIV       : regex_compile(r'/'),
    EXP       : regex_compile(r'\*\*'),
    MOD       : regex_compile(r'%'),
    SHIFTRIGHT: regex_compile(r'>>'), # BEFORE GT
    SHIFTLEFT : regex_compile(r'<<'), # BEFORE LT
    BITAND    : regex_compile(r'&'),
    BITOR     : regex_compile(r'\|'),
    BITXOR    : regex_compile(r'\^'),

    # UNOPS
    INVERT    : regex_compile(r'~'),
    NOT       : regex_compile(r'not'),
    # note that ADD and SUB can also be unops like SUB in '-x' and ADD in '+x' (leaves arg unchanged),

    # BOOLOPS
    AND       : regex_compile(r'and'), # BEFORE ID
    OR        : regex_compile(r'or'), # BEFORE ID

    # COMPARISON
    LEQ       : regex_compile(r'<='), # BEFORE LT
    LT        : regex_compile(r'<'),
    GEQ       : regex_compile(r'>='), # BEFORE GT
    GT        : regex_compile(r'>'),
    EQ        : regex_compile(r'=='), # BEFORE EQ
    NEQ       : regex_compile(r'!='),
    IS        : regex_compile(r'is'), # BEFORE ID
    ISNOT     : regex_compile(r'is\s+not'), # BEFORE ID
    IN        : regex_compile(r'in'), # BEFORE ID
    NOTIN     : regex_compile(r'not\s+in'), # BEFORE ID

    ASN       : regex_compile(r'='),
    ESCQUOTE2 : regex_compile(r'\\\"'),
    ESCQUOTE1 : regex_compile(r'\\\''),
    QUOTE2    : regex_compile(r'\"'),
    QUOTE1    : regex_compile(r'\''),
    HASH      : regex_compile(r'#'),
    PIPE      : regex_compile(r'\|'),
    ID        : regex_compile(r'([a-zA-z_]\w*)'), # must go after keywords like sh
    UNKNOWN   : regex_compile(r'(.)'),
    SOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    EOL : None, # should never be matched against since 'UNKOWN' is a catch-all
    NEWLINE : None, # newlines are manually inserted hence no need for a regex
    KEYWORD : None, # custom matching as a step after ID for efficiency
}
assert len(regex_of_token) == len(_str_of_token)
TOKENS = list(regex_of_token.keys())

for i,tok1 in enumerate(TOKENS):
    for tok2 in enumerate(TOKENS[i+1:]):
        assert tok1 is not tok2
        assert tok1 != tok2




def closer_of_opener(opener_tok):
    if isinstance(opener_tok,Tok):
        opener_tok = opener_tok.typ

    if opener_tok == LPAREN:
        return Tok(RPAREN,'',')')
    if opener_tok == PYPAREN:
        return Tok(RPAREN,'',')')
    if opener_tok == LBRACE:
        return Tok(RBRACE,'','}')
    if opener_tok == LBRACKET:
        return Tok(RBRACKET,'',']')
    if opener_tok == QUOTE1:
        return Tok(QUOTE1,'','\'')
    if opener_tok == QUOTE2:
        return Tok(QUOTE2,'','"')
    if opener_tok == SOL:
        return Tok(EOL,'','')
    raise NotImplementedError


# This is a parsed token
class Tok(LocTagged):
    def __init__(self,typ,data,verbatim, loc):
        super().__init__()
        self.loc = loc
        self.typ=typ            #e.g. IDENTIFIER
        self.data=data          # the contents of the capture group of the Tok's regex, if any.
        self.verbatim=verbatim  # e.g. 'foo'. The verbatim text that the regex matched on
    def __repr__(self):
        data = f"({u.mk_y(self.data)})" if self.data != '' else ''
        return f"{repr(toktyp_of_const[self.typ])}{data}"
    def finish(self): # for LocTagged
        self.finished = True



# a fancy iterator globally used for keeping track of the current state of processing the token stream
class TokStream:
    def __init__(self,token_list):
        self.tkns = token_list
        self.idx = 0

    # returns the number of remaining tokens (including whatever is currently pointed to)
    ##CRFL!
    def __len__(self):
        return max(0,len(self.tkns)-self.idx)
    # a request beyond the end will return None
    ##CRFL!
    def __getitem__(self,rel_idx):
        if self.idx + rel_idx >= len(self.tkns):
            return None #EOL and beyond indicator
        return self.tkns[self.idx+rel_idx]
    # a request of 4 items starting at 2 before the end would yield [tok, tok, None, None]
    # thus it is safe to do v=tstream[:3] then do v[2] for example, as long as you handle the None case
    # (the indexing itself will never throw an error)
    ##CRFL!
    def __getslice__(self, start, end):
        res = self.tkns[self.idx+start:self.idx+end]
        res += [None] * (end-start - len(res)) # fill in rest of request length with Nones
        return res
    def step(self,ntoks=1):
        self.idx += ntoks
        #u.gray("'"+self[0].verbatim+"'" if self[0] is not None else 'None')
    def skip_whitespace(self):
        if self[0].typ == WHITESPACE:
            self.step() #note you can never have mult whitespaces in a row since they consolidate by \s+

    ## left commented for now in hopes that we'll have nice clean code that will never need to do this
    def rewind(self,ntoks=1):
        self.idx -= ntoks


    ## TokStream: LEFT AS WIP. unclear what best design is for these methods. best to make form fit the function by designing States first

def toks_to_typs(tok_list):
    return [t.typ for t in tok_list]



#def initialize(token_list,globals):
#    global tstream
#    tstream = TokStream(token_list,globals)

## Specifications and Assumptions:
## * transition(t) is always called with tstream[0] being t
## * whenever possible we favor run()ing new states only after pointing tstream[0] to the first item that they will be parsing, and we pass any metadata that resulted in their discovery to them. For example when starting a new space-call from "foo 1 2" we were in SNormal mode pointing to 'foo', then we advance twice (through the whitespace) to reach '1' at which point we run_same() SSPACECALL which is fed the proper fname='foo'. Similarly SNormal is initialized by passing it its 'opener' token (e.g. '(') while tstream[0] points one past the opener, as opposed to starting SNormal with tstream[0] pointing to the opener and allowing it to figure that out itself.
        ## * IT IS UNCLEAR WHETHER THIS METHOD OR THE OPPOSITE METHOD MAKES MORE SENSE. The opposite method means that all __init__ constructors for State subclasses generally wouldn't need extra arguments (except in special cases that haven't been encountered yet). Really the answer should be to use whichever makes more sense. Do we want the setup work done by __init__ or do we want it done by the parent
        ## In the following example SSpacecall requires quite a bit of setup involving skipping whitespace that it seems like you might want to package that into init().
        ## note by the way that you can't include skip_whitespace in the __init__ currently because if you do run_next that runs AFTER __init__ which will fuck you up.
#            tstream.step() # now tstream[0] pointing to the whitespace
#            tstream.step() # now tstream[0] pointing one beyond whitespace
#            tstream.skip_whitespace() # skip over any inital whitespace if it exists
#            self.run_same(SSpacecall(self,t.data))
        # EHH ACTUALLY THIS IS A BAD EXAMPLE BC YOU CAN DELETE THE SKIP_WS LINE AND IT WORKS SINCE ALL WS IS AT MOST 1-length by virtue of \s+
        ## Regardless of the method, all States should be clearly annotated with an indicator for what they should be pointing to when run() is called
        ## Note that **good** idea is to combine both approaches: pass in the 'opener' or whatever but also leave it pointing to the opener, and let it do any verification through assert() calls. This would make parser logic bugs much easier to find. State can even have an abstract assert_proper function that gets called at the start of __init__ and both checks validity thru assert calls and does the proper setup. OH WAIT NO thats ***not*** a great idea bc super.__init__ gets called at the start of __init__ so super.init cant be doing all the tests. Also it would mess with the whole build-and-run oneliner tho thats not the end of the world.
    #####^^^^^^^^If it aint broke don't fix it: Just wait until you have a REASON that this is a bad system. Bc right now it works, and it works well^^^^^^^^
    ##### ALSO when you enter SNormal due to a '(' you def wanna be one past that when you first call transition() bc otherwise itll see the '(' and enter into ANOTHER SNormal 
## 
NONE = 0
VERBATIM = 1
POP = -1


### the rules of writing a new State subclass: ###
# Flow of execution for a state, from creation to death:
# __init__(): use this to declare any variables you need, and of course any constructor arguments you want. Start it by calling super().__init__(parent). The key with this is you should NOT interact with tstream as it is at an undefined position at this point.
    # (note that the undefined position thing is because we allow the constructor to be run, then .step() to be called, then .run_*() to be called. And in fact run_next needs to be called using a constructed state, then it .step()s and then .run()s. Hence __init__ will not be seeing the tstream at any sort of consistent, predictable position when it is initialized)
# next run() is invoked (generally by a different state calling run_same or run_next). You can't modify run(). It will call the following few functions. Note that we always assume that run() gets called with tstream[0] being the first token that should be processed, and run() will exit with it pointing to the last token that was processed (it will not step beyond this one).
# pre(): run() will call this first, with no arguments. Use for any setup that depends on tstream. It's best practice to include assert calls (w/ messages) that check that tstream is properly lined up (vastly improves ease of debugging).
# transition():
#   -only call run_same(child) when tstream[0] points to the first token you want child to see. (Call run_next(child) when tstream[1] points to the first token you want child to see)
#   -return POP when the NEXT token (tstream[1]) is the one you want your parent to see. Think of POP as pop_next. (Pop does not actually .step(), but you will return from it into the completion of the parent's transition() function so the step() will automatically occur as transition() returns into run() which calls .step and then .transition again)
# post(text): this will be called with the final accumulated result of the transition() loop, and should do any necessary post processing on it, for example wrapping it in '(' ')' characters for a parenthetical state. post() should also have assert() calls to verify that tstream is properly aligned.



class State:
    def __init__(self,parent,tstream=none, globals=none, debug=none):
        self.parent = parent
        self.halt = False # halt is effectively returning a value AND popping
        self._tstream = parent._tstream if (tstream is None) else tstream
        self._globals = parent._globals if (globals is None) else globals
        self.debug_name = "[debug_name not set: {}]".format(self.__class__)
        self.debug_depth = parent.debug_depth+1 if (parent is not None) else 0
        self.d = parent.d if debug is None else debug
        #self.popped = False
        #self.nostep = False
        #self.tmp = None # just a useful temp var for subclasses that dont want to go to the work of overriding init

    def run(self):
        text = ''
        self.pre()
        self.d.r("starting: "+self.debug_name)
        while True:
            if self.tok() is None:
                assert len(self._tstream)==0,'if tstream[0] is None that must mean we are out of tokens'
                self.d.print("EOL autopop from run()")
                break # autopop on EOL

            _initial_str = f"{self.debug_name}.transition('{self.tok().verbatim}')"
            self.d.y(_initial_str)
            res = self.transition(self.tok())
            assert res is not None, 'for clarity we dont allow transition() to return None. It must return '' or NONE (the global constant) for no extension)'

            _final_str = res
            if res == VERBATIM:
                _final_str = self.tok().verbatim
            elif res == NONE:
                _final_str = '[none]'
            elif res == POP:
                _final_str = 'POP'
            self.d.y(f"{_initial_str} -> '{_final_str}'")

            if isinstance(res,str): # extend text
                text += res
            elif res == POP: # no step on pop
                break
            elif res == NONE: # do nothing. same as res=''
                pass # not a 'continue' bc we still want the later .step() to run
            ## careful. VERBATIM will use whatever tstream[0] is pointing to when transition() exits
            elif res == VERBATIM: # verbatim shortcut
                text += self.tok().verbatim

            if self.halt: # no step on pop
                break
            self.step()
        self.d.p(f"before postproc: {text}")
        out = self.post(text)
        self.d.b(f"{self.debug_name}:{out}")
        return out
    def pre(self): # default implementation
        pass
    def post(self,text): # default implementation
        return text
    def run_same(self,state):
        return state.run()
    def run_next(self,state):
        self._tstream.step()
        return state.run()
    def pop(self,value):
        self.popped = True
        return value
    def transition(self,t):
       raise NotImplementedError # subclasses must override

    def assertpre(self,cond,msg=None):
       m = "pre-assert failed for {}".format(self.debug_name)
       if msg is not None:
           m += ' : ' + msg
       assert cond, m
    def assertpost(self,cond,msg=None):
       m = "post-assert failed for {}".format(self.debug_name)
       if msg is not None:
           m += ' : ' + msg
       assert cond,m
    # functions for managing _tstream and _globals
    # tok()
    # tok(1)
    # tok(list=True)[1:3]
    def tok(self,idx=0,list=False):
        if list:
            return self._tstream
        return self._tstream[idx]
    def step(self):
        return self._tstream.step()
    def rewind(self):
        return self._tstream.rewind()
    # if fname is not in globals or is not callable, return False
    def check_callable(self,fname):
        return callable(self._globals.get(fname))
    # always call check_callable() at some point before argc_of_fname()
    def argc_of_fname(self,fname):
        func = self._globals.get(fname)
        params = inspect.signature(func).parameters.values()
        required_argc = len(list(filter(lambda x: x.default==inspect._empty, params)))
        return required_argc
    def skip_whitespace(self):
        return self._tstream.skip_whitespace()

#   (X      or     {X   etc
#    ^              ^
class SNormal(State):
    def __init__(self,parent,opener): #opener is a Tok
        ## careful. init is called when tstream[0] will not be pointing to whatever token the first transition(t) will be called on. The class e.g. SNormal() will be initialized THEN tstream.idx will be updated (e.g. +1 or +0) THEN run() will be called
        super().__init__(parent)
        self.opener = opener
        self.closer = closer_of_opener(opener) # closer is a TokTyp
        self.debug_name = "SNormal[{}]".format(opener.verbatim)
    def pre(self):
        if self.opener.typ != SOL:
            self.assertpre(self.tok(-1)==self.opener)
    def transition(self,t):
        ## transition(t) always ASSUMES that tstream[0] == t. Feeding an arbitrary token into transition is undefined behavior. Though it should only have an impact on certain peeks
        assert self.tok() is t, "transition(t) assumes that tstream[0] == t and this has been violated"

        if t.typ == self.closer.typ:
            return POP
        elif t.typ == SH_LBRACE:
            return self.run_next(SShmode(self))
        elif t.typ in [LPAREN, LBRACKET, LBRACE]:
            return self.run_next(SNormal(self,t))
        elif t.typ in [QUOTE1, QUOTE2]:
            return self.run_next(SQuote(self,t))
        elif t.typ == ID and self.check_callable(t.data) and (self.tok(1) is None or self.tok(1).typ == WHITESPACE):
            self.step() # now tstream[0] pointing to the whitespace
            self.step() # now tstream[0] pointing one beyond whitespace (which can no longer be a whitespace since WS = \s+)
            return self.run_same(SSpacecall(self,t.data))
        return VERBATIM
    def post(self,text):
        if self.opener.typ != SOL:
            self.assertpost(self.tok().typ == self.closer.typ)
        if self.opener.typ == PYPAREN:
            return '"+str(' + text + ')+"'
        return self.opener.verbatim + text + self.closer.verbatim


#   "X      or      'X
#    ^               ^
class SQuote(State):
    def __init__(self,parent,opener): #opener is a Tok
        super().__init__(parent)
        self.opener = opener
        self.closer = opener # for quotes a closer is it's own opener
        self.debug_name = "SQuote[{}]".format(opener.verbatim)
    def transition(self,t):
        if t.typ == self.closer.typ:
            return POP
        return VERBATIM
    def post(self,text):
        return self.opener.verbatim + text + self.closer.verbatim

#   "X      or      'X
#    ^              ^
class SShquote(State):
    def __init__(self,parent,opener): #opener is a Tok
        super().__init__(parent)
        self.opener = opener
        self.closer = opener # for quotes a closer is it's own opener
        self.debug_name = "SShquote[{}]".format(opener.verbatim)
    def transition(self,t):
        if t.typ == self.closer.typ:
            return POP
        return VERBATIM
    def post(self,text):
        return '\\'+self.opener.verbatim + text + '\\'+self.closer.verbatim

# sh{X
#    ^
class SShmode(State): # set capture_output=False to get ":" line mode or capture_output=True to get "sh{}" expression mode
    def __init__(self,parent,capture_output=True, capture_error=False, exception_on_retcode=None):
        super().__init__(parent)
        self.brace_depth = 1
        self.debug_name = "SShmode"
        self.capture_output = capture_output
        self.capture_error = capture_error
        self.exception_on_retcode = exception_on_retcode
    def transition(self,t):
        if t.typ == LBRACE:
            self.brace_depth += 1
            return VERBATIM
        elif t.typ == RBRACE:
            self.brace_depth -= 1
            if self.brace_depth == 0:
                return POP
            return VERBATIM
        elif t.typ in [QUOTE1, QUOTE2]:
            return self.run_next(SShquote(self,t))
        elif t.typ == PYPAREN:
            return self.run_next(SNormal(self,t))
        return VERBATIM
    def post(self,text):
        return 'backend.sh("{}",capture_output={},capture_error={},exception_on_retcode={})'.format(text,
                self.capture_output,
                self.capture_error,
                self.exception_on_retcode)


# foo    a b c
#        ^non whitespace (note that all contig whitespace is at most length 1 bc of \s+)
# foo a bar b car 
# foo(a,bar(b,car()))
class SSpacecall(State):
    def __init__(self,parent,func_name):
        super().__init__(parent)
        self.func_name = func_name
        self.argc = self.argc_of_fname(self.func_name)
        self.argc_left = self.argc
        self.debug_name = "SSpacecall[{}]".format(func_name)

    def transition(self,t):
        if self.argc == 0:
            return POP ##todo make this better so that "sum one one" works
        self.argc_left -= 1
        if self.argc_left == 0:
            self.halt = True

        dont_abort_verbatim = ['>','>=','<=','==','<','=','*','+','-','/','//'] ##UNFINISHED, Add more to this! in general boolop/binop/unop/cmp. Note i left '=' in since right now we parse '==' as '=','='.
        over = Overloader(self,SNormal(self,Tok(SOL,'','')))
        over.prev_non_ws = None # local var used by lambdas. Last non-whitespace char seen
        def pre_trans(t): # keep track of last non-whitespace seen
            if t.typ != WHITESPACE:
                over.prev_non_ws = t

        over.pre_trans = pre_trans # by closure 'pre' will properly hold the correct references to 'over'
        over.pop = lambda t: (t.typ == WHITESPACE and (over.prev_non_ws is None or over.prev_non_ws.verbatim not in dont_abort_verbatim))
        return self.run_same(over) + ','

    def post(self,text):
        self.rewind() ## TODO fix this grossness, or just accept it. it aligns it so when caller calls next() they recv the last space of the macro, which is imp for recursive macros
        return self.func_name+'('+text[:-1]+')' # kill the last comma with :-1

class Overloader(State):
    def __init__(self,parent,state):
        super().__init__(parent)
        self.debug_name = "Overloader[{}]".format(state.debug_name)
        self.inner = state
        self.pre_trans = lambda t: None
        self.post_trans = lambda t: None
        self.pop = lambda t: False
        self.override = lambda t: NONE
        self.use_override = [] # list of tokens for which override(t) should be used in place of inner.transition(t)
    def transition(self,t): ##maybe allow pre/post to modify res, or something like that. Or be able to selectively use an over.alt(t) function instead of inner.trans whenever self.usealt(t) is true?
        if self.pop(t): return POP
        self.pre_trans(t)
        if t in self.use_override:
            res = self.override(t)
        else:
            res = self.inner.transition(t)
        self.post_trans(t)
        return res

# the first state just used at the start of the line
class SInitial(State):
    def __init__(self,parent,tstream, globals, debug):
        super().__init__(parent, tstream=tstream, globals=globals, debug=debug)
        self.debug_name = "SInitial"
        self.debug = debug
    def transition(self,t):
        self.halt = True # this is a 1 shot transition function (except in case of WHITESPACE, see below)
        ##This should handle the ">a" syntax and the quick-fn-def syntax, and should do a self.run_same to SNormal if neither case is found

        ## and : linestart syntax for sh line

        if t.typ == WHITESPACE:
            self.halt = False # briefly allow us to run another step of SInitial
            return VERBATIM
        if t.typ == COLON or (t.typ == ID and t.data == 'sh'):
            res = self.run_next(SShmode(self,capture_output=False,capture_error=False, exception_on_retcode=False)) ##allow it to kill itself from eol
        else:
            res = self.run_same(SNormal(self,Tok(SOL,'','')))
        assert len(self._tstream)==0,'tstream should be empty since SInitial should consume till EOL'
        return res

leading_whitespace_regex = re.compile(r'(\s*)')


LineData = namedtuple('LineData', 'linetkns, leading_whitespace, newline_tok, commenttkns')

# turns a string into a list of Tokens
# if ur sending an interative line PLEASE strip the trailing \n
def tokenize(s):
    assert not '\t' in s, "Havent yet figured out how tabs should work with indent levels since len('\t') is 1 but they take up 4 visual spaces, just see what python allows and experiment with noexpandtab. If python doesnt allow for mixed tabs and spaces in a file then thats perfect bc then len() works on leading_whitespace"
    tokenized_lines = []
    lineno=1
    for line in s.split('\n'):
        linetkns = []
        commenttkns = []
        list_to_extend = linetkns
        leading_whitespace = leading_whitespace_regex.match(line).group()
        remaining = line[len(leading_whitespace):]
        while remaining != '':
            for TOKEN,regex in regex_of_token.items():
                charno = len(line) - len(remaining)
                match = regex.match(remaining)
                if match is None:
                    continue
                remaining = remaining[match.end():]
                if TOKEN is HASH and list_to_extend is not commenttkns:
                    list_to_extend = commenttkns # switch to adding to list of comment tokens
                    continue
                if TOKEN is ID:
                    if match.group() in keywords:
                        TOKEN = KEYWORD # for efficiency rather than having every keyword in the regex
                grps = match.groups()
                list_to_extend.append(Tok(TOKEN,
                    grps[0] if grps else '', # [] is nontruthy
                    match.group(),
                    Loc(Pos(lineno,charno),Pos(lineno,len(line)-len(remaining))) # lineno,charno TODO watch out for how \t is handled!
                    ))
                list_to_extend[-1].finish()
                break #break unless you 'continue'd before
            else: # nobreak
                raise ValueError(f"Somehow failed to match on any tokens, should be impossible because of the UNKNOWN token")
        #is_comment = (line.strip() == '' or line.strip()[0] == '#') # lines that just have comments or nothing on them
        newline_tok = Tok(NEWLINE, '\n', '\n', (lineno,len(line)+1))
        tokenized_lines.append(LineData(linetkns, leading_whitespace, newline_tok, commenttkns))
        lineno += 1
    return tokenized_lines

INCOMPLETE = -1
# the main function that run the parser
# It goes string -> Token list -> Atom list -> Atom list (w MacroAtoms) -> final python code
# `interpreter` should be True if we're parsing from an interpreter and thus atomize can return INCOMPLETE rather than throwing an error when parens/etc are unclosed
def parse(lines,globals=None,interpreter=False,debug=False):
    debug = True
    debug = Debug(debug)
    linedatalist = tokenize(lines)
    commentlinedata = [linedata.commenttkns for linedata in linedatalist] # may want for something sometime
    debug.print(f"Tokens: {[linedata.linetkns for linedata in linedatalist]}")
    masteratom = atomize(linedatalist,interpreter)
    debug.print(masteratom)
    if atoms is INCOMPLETE:
        print("INCOMPLETE")
        exit(0)
        return INCOMPLETE
    ast = make_ast(masteratom)
    #debug.print(ast)
    ast = comment_mods(commentlinedata,ast) # does whatever mods we want comments to be able to do to the tree
    debug.print(ast)
    #tstream = TokStream(token_list)
    #init = SInitial(parent=None, tstream=tstream, globals=globals, debug=debug)
    #out = init.run()
    return ast
## CAREFUL RELOADING CODEGEN IT WILL RESET THESE VALUES

def comment_mods(commentlinedata,ast):
    return ast

def make_ast(masteratom):
    res = []
    for elemlist in masteratom.body:
        stmtnode = parseStmt(elemlist)
        res.append(stmtnode)
    return Module(res)




parse('#here is a comment',{})
parse('sh echo hey there',{})

parse('({(test)})',{})
parse('print((x + 3)/2.4);print("hi")',{})
parse('test;test2',{})
parse('test\ntest2',{})


print("===")
#parse('''ret''',{})

parse('''
def foo(a,b):
    def bar():
        return a
    x=bar()
    return a+b
foo(10,20)
u
''',{})

parse('''
x
def foo():
# here is a test
''',{})
parse('x = sh{echo hey there $(py expr)}',{})
parse('sh echo hey there',{})



#out = parse('test',{})

#u.blue('========')
#out = parse(':test',{})
#out = parse('x = (sh{test}) + 1',{})
#print(out)

#parse("fname,linect = %parse sh{wc -l $file} (str $1, int $2)")
#parse("z = %parselines1  x (str $1, int $2)")
#parse("sh{echo \"hi \"$there\" you're the \"$one}")
#parse("if %exists? filename:")
#parse("vi_list = %parselines1 (%cat file) (int $1, int $2)")




