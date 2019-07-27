from codegen3 import *


MatchItem = namedtuple('MatchItem','lhs_list rhs')
class MatchExpr(Expr):
    """
    Examples:
    """
    def __init__(self,expr,items):
        super().__init__()
        self.expr = expr
        self.items = items
    @staticmethod
    def build(p):
        """
        match ::= "(" "match" expression "with" match_item+ ")"
        match_item ::= lhs_item+ "->" expression_list_nobar
        lhs_item ::= "|" expression_list_nobar
        """
        def match_item():
            def lhs_item():
                p.token('|')
                p.expression_list_nobar()
            lhs_list = p.list(lhs_item,nonempty=True)
            p.token('->')
            rhs = p.expression_list_nobar()
            return MatchItem(lhs_list,rhs)
        with p.parens():
            p.keyword('match')
            expr = p.expression()
            p.keyword('with')
            items = p.list(match_item,nonempty=True)
        return Match(expr,items)
    def desugar(self):
        """
        Desugars to an expression
        match_expr_fn(`expr`,*`items`) or something like that
        """
        # TODO not done, not worth spending much time on rn anyways.
        return Call(parse_expr('codegen_custom.match_expr_fn'), Arguments([A]))

def match_expr_fn(val,*lhs_rhs_pairs):
    for lhs,rhs in lhs_rhs_pairs:
        if val == lhs() or lhs() is MATCH_WILDCARD:
            return rhs()
    raise MatchFailException

MATCH_WILDCARD = CONSTANT()

class MatchStmt(Stmt):
    pass

@compound(None)
class BlockAssign(Stmt):
    def __init__(self,targets,body):
        self.targets = targets
        self.body = body
    @staticmethod
    def build(p):
        """
        block_assign ::= target_list "=:" suite
        """
        with p.head():
            targets = p.target_list()
            p.token('=')
        body = p.simple_body()
        return BlockAssign(targets,body)
    def desugar(self):
        """
        Desugars to 2 statments:
            1. An argumentless FuncDef of the function _TMP_BLOCKASSIGN holding `body`
            2. An Assignment of _TMP_BLOCKASSIGN() to `targets`
        """
        s1 = FuncDef(
                decorators=None,
                fname='_TMP_BLOCKASSIGN',
                params=Parameters.argless(),
                body=self.body
                is_async=False)
        call = Call(
                func=Var('_TMP_BLOCKASSIGN'),
                args=Arguments([]))
        s2 = Assignment(
                targets=self.targets,
                val=call)
        return [s1,s2] # returns Stmt list




