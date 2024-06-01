# -*- coding: utf-8 -*-
"""


@author: fhu
"""

from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds
from rdkit import RDLogger

import copy
from itertools import combinations
#from simple import simple_iter
#from igraph import Graph
import argparse

RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-input_file', '-i', required=True,                        
                        help='.smi or .sdf file of molecules to be fragmented')    
    
    parser.add_argument('-output_path', '-o', required=True,
                        help='path of the output fragments file')
    
    parser.add_argument('-maxBlocks', required=True,
                        help='the maximum number of building blocks that the fragments contain')
    
    parser.add_argument('-maxSR', required=True,  # 不切割的最小的环
                        help='only cyclic bonds in smallest SSSR ring of size larger than this value will be cleaved')

    parser.add_argument('-asMols', required=True,
                        help='True of False; if True, MacFrag will reture fragments as molecules and the fragments.sdf file will be output; if False, MacFrag will reture fragments.smi file with fragments representd as SMILES strings')
    
    parser.add_argument('-minFragAtoms', required=True,  # 包含的最小的原子
                        help='the minimum number of atoms that the fragments contain')

    return parser.parse_args()


# These are the definitions that will be applied to fragment molecules:
environs = {
  'L1': '[C;D3]([#0,#6,#7,#8])(=O)',  #original L1
  'L2': '[O;D2]-[#0,#6,#1]',  #original L3
  'L3': '[C;!D1;!$(C=*)]-[#6]',  #original L4
  'L4': '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',  #original L5
  'L5': '[C;D2,D3]-[#6]',  #original L7
  'L6': '[C;!D1;!$(C!-*)]',  #original L8
  'L61': '[C;R1;!D1;!$(C!-*)]',
  'L7': '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',  #original L9
  'L8': '[N;R;$(N(@C(=O))@[#6,#7,#8,#16])]',  #original L10
  'L9': '[S;D2](-[#0,#6])',  #original L11
  'L10': '[S;D4]([#6,#0])(=O)(=O)',  #original L12
  
  'L11': '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',  #original L13
  'L111': '[C;R2;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
  'L112': '[C;R1;$(C(-;@[C,N,O,S;R2])-;@[N,O,S;R2])]',
  
  'L12': '[c;$(c(:[c,n,o,s]):[n,o,s])]',  #original L14
  
  'L13': '[C;$(C(-;@C)-;@C)]',  #original L15
  'L131': '[C;R2;$(C(-;@C)-;@C)]',
  'L132': '[C;R1;$(C(-;@[C;R2])-;@[C;R2])]',
  
  'L14': '[c;$(c(:c):c)]', #original L16
}
reactionDefs = (
  # L1
   [('1', '2', '-'),
    ('1', '4', '-'),
    ('1', '8', '-'),
    ('1', '11', '-'),
    ('1', '12', '-'),
    ('1', '13', '-'),
    ('1', '14', '-')],

  # L2
    [('2', '3', '-'),
    ('2', '11', '-'),
    ('2', '12', '-'),
    ('2', '13', '-'),
    ('2', '14', '-')],

  # L3
    [('3', '4', '-'),
    ('3', '9', '-')],

  # L4
    [('4', '10', '-'),
    ('4', '12', '-'),
    ('4', '14', '-'),
    ('4', '11', '-'),
    ('4', '13', '-')],

  # L5
    [('5', '5', '=')],

  # L6
    [('6', '7', '-'),
    ('6', '8', '-'),
    ('6', '11', '-;!@'),
    ('6', '12', '-'),
    ('6', '13', '-;!@'),
    ('6', '14', '-')],
     
  # L61
    [('61', '111', '-;@'),
    ('61', '131', '-;@')],   

  # L7
    [('7', '11', '-'),  
    ('7', '12', '-'),  
    ('7', '13', '-'),
    ('7', '14', '-')],

  # L8
    [('8', '11', '-'),
    ('8', '12', '-'),
    ('8', '13', '-'),
    ('8', '14', '-')],

  # L9
    [('9', '11', '-'),
    ('9', '12', '-'),
    ('9', '13', '-'),
    ('9', '14', '-')],

  # L11
    [('11', '12', '-'),
    ('11', '13', '-;!@'),
    ('11', '14', '-')],
     
  # L112
    [('112', '132', '-;@')],

  # L12
    [('12', '12', '-'),  
    ('12', '13', '-'),
    ('12', '14', '-')],

  # L13
    [('13', '14', '-')],

  # L14
    [('14', '14', '-')],  
    )

environMatchers = {}
for env, sma in environs.items():  # environs 'L1': '[C;D3]([#0,#6,#7,#8])(=O)'
    
    environMatchers[env] = Chem.MolFromSmarts(sma)  # {'L1': mol  }
   
        
bondMatchers = []

for compats in reactionDefs:  # compats [('3', '4', '-'), ('3', '9', '-')]
    
    tmp = []
    for i1, i2, bType in compats:
        e1 = environs['L%s' % i1]
        e2 = environs['L%s' % i2]
        patt = '[$(%s)]%s[$(%s)]' % (e1, bType, e2)
        patt = Chem.MolFromSmarts(patt)
        tmp.append((i1, i2, bType, patt))
    bondMatchers.append(tmp)  # [(i1, i2, bType, patt) ]


def SSSRsize_filter(bond,maxSR=13):  # 最大多少个环
    
    judge=True
    for i in range(3,maxSR+1):
        if bond.IsInRingSize(i) :
            judge=False
            break           
    return judge
    
    
def searchBonds(mol,maxSR=13):  # 搜寻键 根据环境和键模式在分子中搜索键。track预处理的键并生成符合键信息
    
    bondsDone = set()  # 键部分
    
    envMatches = {}  # {'L1': True, 'L2': False } 存储环境匹配的结果
    for env, patt in environMatchers.items():  # {'L1': mol, 'L2': mol2 } 划分好的环境
        envMatches[env] = mol.HasSubstructMatch(patt)  # 子结构 HasSubstructMatch bool类型
        # 分子是否具有与模式匹配的子结构
        
    for compats in bondMatchers:  # [(i1, i2, bType, patt) ]
                
        for i1, i2, bType, patt in compats:
            if not envMatches['L' + i1] or not envMatches['L' + i2]:  # 均满足 两个环境模式都为True
                continue
            
            matches = mol.GetSubstructMatches(patt)  # 返回与子结构查询匹配的分子原子索引元组(,),(,)
            for match in matches:  # 原子索引组中的原子 单个原子
                if match not in bondsDone and (match[1], match[0]) not in bondsDone:  # 不在键
                    bond=mol.GetBondBetweenAtoms(match[0],match[1])  # 返回两个原子之间的键
                    
                    if not bond.IsInRing():  # 非环键
                        bondsDone.add(match) 
                        yield (((match[0], match[1]), (i1, i2)))
                    elif bond.IsInRing() and  SSSRsize_filter(bond,maxSR=maxSR):  # SSSRsize_filter maxSR以下的环为False就不加
                        bondsDone.add(match)
                        yield (((match[0], match[1]), (i1, i2)))
                    
def mol_with_atom_index(mol):  # 原子索引作为原子映射数添加到分子中。
    for atom in mol.GetAtoms():  # GetAtoms
        atom.SetAtomMapNum(atom.GetIdx())  # SetAtomMapNum设置原子映射数，值为0将清除原子映射 atom.GetIdx()返回原子的索引(在分子中排序)
    return mol

def mol_remove_atom_mapnumber(mol):  # 删除了原子的map number
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return mol

def Get_block_index(blocks):  # 得到block的索引
    
    block_index={}
    i=0
    for bs in blocks:
        tmp=[a.GetAtomMapNum() for a in bs.GetAtoms() if a.GetSymbol()!='*']  # block中的非*原子 GetAtomMapNum获取原子映射编号，如果未设置则返回0
#        tmp=list(set(tmp))
        block_index[tuple(tmp)]=i  # {0:(, , ), 1:(, , ), ...}
        i+=1      
    return block_index  # # {0:(, , ), 1:(, , ), ...}

def extrac_submol(mol,atomList,link):  # 提取子mol文件
    aList_mol=list(range(mol.GetNumAtoms()))
    aList_link=list(set([a[0] for a in link]))
    aList_submol=list(set(atomList+aList_link))
    aList_remove=[a for a in aList_mol if a not in aList_submol]
    eMol=Chem.RWMol(mol)
    
    aList_bbond=[a for a in aList_link if a not in atomList]
    for b in combinations(aList_bbond,2):
        eMol.RemoveBond(b[0],b[1])
    
    aList_remove.sort(reverse=True)
    for a in aList_remove:
        eMol.RemoveAtom(a) 

            
    for ba,btype in link:
        if ba in atomList:
            continue
        tmpatom=[a for a in eMol.GetAtoms() if a.GetAtomMapNum()==ba][0]
        tmpatom.SetIsAromatic(False)
        tmpatom.SetAtomicNum(0)
        tmpatom.SetIsotope(int(btype))
        tmpatom.SetNoImplicit(True)
            
    frag=eMol.GetMol()
   
    for a in frag.GetAtoms():
        a.ClearProp('molAtomMapNumber')   
    return frag
 
    
def simple_iter(graph, k):  # 子图搜索
    cis=[]
#    numb_at_most_k = 0
    # colors for vertices (igraph is to slow) 顶点的颜色(图形会变慢)
    colors = graph.vcount()*[-1]  # 获取图对象中节点的数量 颜色
    # consider only graphs with at least k vertices 只考虑至少有k个顶点的图
    for i in range(graph.vcount() -1, -1, -1):
        # subgraph_set
        subgraph_set = [i]
        # print first induced subgraph conisting only of vertex i
#        subgraph_file.write(print_names(graph, subgraph_set))
        cis.append(copy.deepcopy(subgraph_set))
#        numb_at_most_k += 1
        # extension set
        extension = []
        # list of lists of all exclusive neighbors
        ex_neighs = [[]]
        # color closed neighborhood of start vertex
        colors[i] = 0
        for vertex in graph.neighbors(i):
            if vertex >= i:
                break
            colors[vertex] = 1
            extension.append(vertex)
            ex_neighs[0].append(vertex)
        # lists for the pointers of the extension set
        pointers = [extension.__len__()-1]
        # we save the index in whose closed neighborhood the actual pointers are, to perform the jumps
        poi_col = [1]
        # subgraph size
        sub_size = 1
        # enumerate all subgraphs containing vertex i
        while subgraph_set != []:
            # for each vertex in the extension set, create a new branch
            # if the actual pointer points to null (-1), then the corresponding extension set is empty
            while pointers[-1] > -1:
                last = pointers[-1]
                vertex = extension[last]
                ver_col = colors[vertex]
                # move pointer one to the left
                pointers[-1] -= 1
                act_vertex = pointers[-1]
                # check if vertex is an exclusive neighbor of the vertex last added to the subgraph set
                if ver_col == sub_size:
                    # if yes delete this vertex from the extension set
                    extension.pop()
                # check if pointer jumps, in other words check if the new next vertex in the extension set has a different color than the current vertex
                if act_vertex > -1:
                    if colors[extension[act_vertex]] < poi_col[-1]:
                        # jump with pointer (we stay at the same point if the color of the actual vertex is one less)
                        pointers[-1] = pointers[poi_col[-1]-2]
                        # update color
                        poi_col[-1] = colors[extension[pointers[-1]]]
                # create next child
                subgraph_set.append(vertex)
#                numb_at_most_k += 1
#                subgraph_file.write(print_names(graph, subgraph_set))
                cis.append(copy.deepcopy(subgraph_set))
                sub_size += 1
                # check if adding this vertex leads to a solution
                if sub_size == k:
                    # check time limit
#                    time_now = clock()
#                    if time_now > time_max:
 #                       return numb_at_most_k
                    subgraph_set.pop()
                    sub_size -= 1
                # otherwise create new enumeration tree node
                else:
                    # find the exclusive neighbors of vertex
                    ex_neighs.append([])
                    found = False
                    for neig in graph.neighbors(vertex):
                        if neig >= i:
                            break
                        if colors[neig] == -1:
                            colors[neig] = sub_size
                            ex_neighs[-1].append(neig)
                            extension.append(neig)
                            found = True
                    # if there is an exclusive neighbor, the new pointer points to him, otherwise to the old pointer
                    if found == True:
                        pointers.append(extension.__len__()-1)
                        poi_col.append(sub_size)
                    else:
                        pointers.append(pointers[-1])
                        poi_col.append(poi_col[-1])
            # now the last pointer points to null, so restore the parent
            pointers.pop()
            poi_col.pop()
            subgraph_set.pop()
            for vertex in ex_neighs[-1]:
                colors[vertex] = -1
            ex_neighs.pop()
            sub_size -= 1
        # remove color from the root
        colors[i] = -1
    return cis       

       
def MacFrag(mol,maxBlocks=6,maxSR=12,asMols=False,minFragAtoms=1):  # 切割函数 主入口
    # maxBlocks 砌块最大数量  maxSR 等于小于该值环结构不切 minFragAtoms=1 最小单位1
    fragPool={}
    mol=mol_with_atom_index(mol)    # 带原子映射的mol文件
    #print(f'mol_with_atom is {mol}')
    bonds=list(searchBonds(mol,maxSR=maxSR))  # 找到切割的原子  [((match[0], match[1]), (i1, i2)), ]
    bonds_re = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in bonds]
    print(f'search bonds is {bonds}')  # 找到

    print(f'bonds_re is {bonds_re}')  # 找到
    tmp = Chem.FragmentOnBonds(mol, bonds_re, addDummies=False)  # mol文件 加不加虚拟原子
    frags_idx_lst = Chem.GetMolFrags(tmp)

    print(f'frags_idx_lst is {frags_idx_lst}')  # 找到
    return frags_idx_lst



    #fragmol=BreakBRICSBonds(mol, bonds=bonds)  # 使用指定键断裂该分子 该断的都断 返回多个mol文件
    #print(f'fragmol is {fragmol}')  #

    #blocks=Chem.GetMolFrags(fragmol,asMols=False)  # 从分子中找到不连接的片段
    #print(f'blocks is {blocks}')

    #blocks_atom_num = Chem.GetMolFrags(fragmol, asMols=False)
    #print(f'blocks_atom_num is {blocks_atom_num}')  # 将大于原子数量的虚拟原子去掉？

#smiles = 'CC1=CN=C(NC2=CC(COC/C=C/COC3)=C(OCCN4CCCCC4)C=C2)N=C1NC5=CC3=CC=C5'
#mol = Chem.MolFromSmiles(smiles)
#MacFrag(mol)
# def main():
#     opt = parse_args()
#     # input_file=opt.input_file
#
#     dir=opt.output_path
#     asMols=opt.asMols  # false
#     maxBlocks=int(opt.maxBlocks)  # 1
#     maxSR=int(opt.maxSR)  # 不切割的最大环大小
#     minFragAtoms=int(opt.minFragAtoms)  # 最小的片段原子数
#     write_file(input_file,dir,maxBlocks,maxSR,asMols,minFragAtoms)
#
# if __name__ == '__main__':
#     main()
