import pyformlang
from pyformlang.cfg import CFG, Production


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    nullable_vars = cfg.get_nullable_symbols()
    cnf_cfg = cfg.to_normal_form()

    new_productions = set(cnf_cfg.productions)
    for var in nullable_vars:
        new_productions.add(Production(var, []))

    result_cfg = CFG(
        start_symbol=cnf_cfg.start_symbol,
        productions=new_productions,
    )

    return result_cfg.remove_useless_symbols()
