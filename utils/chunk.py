'''
    Created on Sep 06, 2018

    @author: Varela

    Provides functionality to convert ctree into chunks
'''
import re
from collections import namedtuple, OrderedDict

Chunk = namedtuple('Chunk', ('role', 'init', 'finish'))


def chunk_stack_process(time: int, ctree: str, chunk_stack: list):

    for role_ in re.findall('\(([A-Z]*)', ctree):
        chunk_stack.append(Chunk(role=role_, init=time, finish=None))

    finished_chunks = ctree.count(')')
    c = 0
    if finished_chunks > 0:
        stack_len = len(chunk_stack)
        for i_, ck_ in enumerate(reversed(chunk_stack)):
            j_ = stack_len - (i_ + 1)
            if c < finished_chunks and ck_.finish is None:
                chunk_stack[j_] = Chunk(
                    role=ck_.role,
                    init=ck_.init,
                    finish=time
                )
                c += 1
            if c == finished_chunks:
                break


def chunk_stack2ckiob(id_list: list,
                      chunk_stack: list) -> dict:
    '''
        * Priority
        PP -> NP -> ADJVP -> ADJ -> VP ELSE O
    '''
    ckiob_list = []
    ckid_prev = chunk_stack[0]
    for idx in id_list:
        ck_list = [ck_ for ck_ in chunk_stack
                   if ck_.init <= idx and idx <= ck_.finish]

        # PP -> NP -> ADJVP -> ADJ -> VP ELSE O
        ckid = chunk_stack[0]
        for ck in reversed(ck_list):
            if ck.role == 'PP':
                ckid = ck
            elif ck.role == 'NP' and ckid.role not in ('PP'):
                ckid = ck
            elif ck.role == 'ADJP' and ckid.role not in ('PP', 'NP'):
                ckid = ck
            elif ck.role == 'ADVP' and ckid.role not in ('PP', 'NP', 'ADJP'):
                ckid = ck
            elif ck.role == 'VP' and ckid.role not in ('PP', 'NP', 'ADJP', 'ADVP'):
                ckid = ck
            else:
                if ck.role in ('FCL', 'CU', 'ICL'):
                    break

        if ckid == chunk_stack[0]:
            ckiob = 'O'
        else:
            if ckid != ckid_prev:
                ckiob = 'B-{:}'.format(ckid.role)
            else:
                ckiob = 'I-{:}'.format(ckid.role)

        ckiob_list.append(ckiob)
        ckid_prev = ckid

    ckiob_dict = OrderedDict(sorted(
        zip(id_list, ckiob_list),
        key=lambda x: x[0]
    ))
    return ckiob_dict


def chunk_dict_maker(id_dict: dict,
                     proposition_dict: dict,
                     ctree_dict: dict) -> dict:
    #  chunk_dict is a dict of dicts
    #  inner dict contains chunk tuples
    ckiob_dict = dict()
    prev_prop = -1
    chunk_stack = []
    id_list = []
    fst = True
    for idx, prop in proposition_dict.items():
        if prop != prev_prop and not fst:
            ckiob_dict[prev_prop] = chunk_stack2ckiob(id_list, chunk_stack)
            id_list = []
            chunk_stack = []

        chunk_stack_process(id_dict[idx], ctree_dict[idx], chunk_stack)
        id_list.append(id_dict[idx])
        prev_prop = prop
        fst = False

    ckiob_dict[prev_prop] = chunk_stack2ckiob(id_list, chunk_stack)
    return ckiob_dict