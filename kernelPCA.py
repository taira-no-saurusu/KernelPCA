import networkx as nx
import collections
import copy
import numpy as np

"""
networkxのgraphインスタンスから隣接行列を返す
G: networkx.Graph
return : 隣接行列
"""
def nx_to_adj(G):
    return np.array(nx.adjacency_matrix(G,weight=None).todense().astype(int))


"""
隣接行列から次数行列を返す
adj : 隣接行列(ndarray)
return : 次数行列(ndarray)
"""
def adj_to_ord(adj):
    # 行列のサイズ
    matrix_size = len(adj)

    # 次数行列初期化（ゼロ行列）
    ord = np.zeros([matrix_size, matrix_size], int)

    for a in range(matrix_size):
        # 行に対応するノードの次数を取得して次数行列となるordの対角成分に追加
        ord[a][a] = np.size(adj[a][adj[a] != 0])

    return ord

"""
隣接行列からグラフラプラシアンを生成
adj : 隣接行列（ndarray）
return : グラフラプラシアン(ndarray)
"""
def adj_to_lap(g):
    # 次数行列を取得
    matrix_ord = adj_to_ord(g)
    return matrix_ord - g
