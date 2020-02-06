import pyodbc
import pandas as pd
import codecs
import json
import networkx as nx
from sklearn.cluster import SpectralClustering, DBSCAN

# pip install python-louvain
import community


def main():
    with codecs.open('app.config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)

    connection_string_251 = config['connection_string_251']
    conn = pyodbc.connect(connection_string_251)

    with codecs.open(r'query/Doc-Word.sql', 'r', encoding='utf-8') as file:
        User_Word_query = file.read()

    data = pd.read_sql_query(User_Word_query, conn)

    # prune data
    data = data[data['WordCount'] >= 5]
    data.reset_index(drop=True, inplace=True)

    Nodes = data.groupby(['Document'])['Word'].count()
    Nodes = pd.DataFrame(Nodes)
    Nodes.reset_index(inplace=True)

    # creat graph
    data_merge = pd.merge(data, data, how='inner', left_on=['Word'], right_on=['Word'])
    edges = data_merge.groupby(['Document_x', 'Document_y']).count()
    edges.reset_index(inplace=True)

    G = nx.from_pandas_edgelist(edges, 'Document_x', 'Document_y', 'Word')
    G.remove_edges_from(G.selfloop_edges())
    Edges = nx.to_pandas_edgelist(G, source='Document_x', target='Document_y')
    Edges.rename(columns={'Document_x': 'source', 'Document_y': 'target', 'Word': 'weight'}, inplace=True)

    Edges = pd.merge(Edges, Nodes, how='left', left_on=['source'], right_on=['Document'])
    Edges = pd.merge(Edges, Nodes, how='left', left_on=['target'], right_on=['Document'])
    Edges.drop(['source', 'target'], axis=1, inplace=True)

    Edges['sim'] = Edges['weight'] / (Edges['Word_x'] + Edges['Word_y'] - Edges['weight'])
    Edges['dis'] = 1 - Edges['sim']

    # prune graph
    tmp = pd.DataFrame(Edges['sim'].sort_values(ascending=False))
    tmp.reset_index(drop=True, inplace=True)
    thresh = int(0.2 * len(tmp))
    thresh = tmp.iloc[thresh]['sim']

    prune_edges = False
    if prune_edges:
        Edges = Edges[Edges['sim'] >= thresh]
        Edges.reset_index(drop=True, inplace=True)
        # save pruned edges
        Edges.to_csv(r'data/Prune_Edges.csv', columns=['Document_x', 'Document_y', 'sim'], index=False)

    # similarity and distance graph
    G_sim = nx.from_pandas_edgelist(Edges, 'Document_x', 'Document_y', 'sim')
    # nx.write_edgelist(G_sim, r'data/sim_graph.csv')

    G_dis = nx.from_pandas_edgelist(Edges, 'Document_x', 'Document_y', 'dis')
    # nx.write_edgelist(G_sim, r'data/dis_graph.csv')

    # ######################## run DBSCAN ########################
    dbs_data = DBSCAN(eps=17, min_samples=10, metric='precomputed').fit(nx.adjacency_matrix(G_data))

    # ######################## Spectral Clustering ########################
    spect_cluster = SpectralClustering(n_clusters=10, affinity='precomputed', n_init=100).fit(
        nx.adjacency_matrix(G_dis))
    spectral_cluster_res = pd.DataFrame(columns=['Document', 'Class'])
    spectral_cluster_res['Document'] = G_dis.nodes
    spectral_cluster_res['Class'] = spect_cluster.labels_
    spectral_cluster_res.to_csv(r'data/spectral_res.csv', index=False)

    # ######################## Run Louvain #########################
    partitions = community.best_partition(G)

    ######################## Save Files ########################
    # Save Edge List
    # nx.write_edgelist(G, 'graph.csv')

    # with open('louvain.txt', 'w') as f:
    #     json.dump(partitions, f)
    #
    # # DBSCAN Results
    # cluster_res = pd.DataFrame(columns=['Document', 'Class'])
    # cluster_res['Document'] = G.nodes
    # cluster_res['Class'] = dbs_data.labels_
    # cluster_res.to_csv('dbs_res.csv', index=False)

    # ############################ SubGraph ##############################
    # # DBSCAN subgraph
    # sub_nodes = cluster_res.loc[cluster_res['Class'] == 8]
    # sub_graph = G.subgraph(sub_nodes['Concept'])
    # sub_spect_cluster = SpectralClustering(n_clusters=20, affinity='precomputed', n_init=100).fit(
    #     nx.adjacency_matrix(sub_graph))
    # sub_cluster_res = pd.DataFrame(columns=['Concept', 'Class'])
    # sub_cluster_res['Concept'] = sub_graph.nodes
    # sub_cluster_res['Class'] = sub_spect_cluster.labels_
    # sub_cluster_res.to_csv('sub_spect_res.csv', index=False)
    #
    # # Remove DBSCAN Noises from Graph
    # noise_ind = np.where(dbs_data.labels_ == -1)[0]
    # noise_nodes = np.array(G.nodes)[noise_ind]
    # G.remove_nodes_from(noise_nodes)
    #
    # # Louvain subgraph
    # nodes = pd.DataFrame([{"Concept": k, "Class": v} for k, v in partitions.items()])
    # sub_nodes = nodes.loc[(nodes['Class'] == 2) | (nodes['Class'] == 3)]
    # sub_nodes.reset_index(drop=True,inplace=True)
    # sub_graph = nx.subgraph(G_data, sub_nodes['Concept'])
    #
    # # Run Louvain
    # sub_partitions = community.best_partition(sub_graph)
    # with open('sub_louvain.txt', 'w') as f:
    #     json.dump(sub_partitions, f)
    #
    # newf = data.pivot(index='FK_ConceptId', columns='FK_TagId')
    # newf.fillna(0, inplace=True)
    # newf = newf.astype(int)
    #
    # newf.to_csv(r'data.csv', sep='\t', index=True, header=True)
    #
    # dis = pdist(newf, 'euclidean')

    print('done')


if __name__ == '__main__':
    main()
