# protein-dynamics script for analysis of two human prion protein states (normal & misfolded)
# written by Sahil Ganatra

import prody
import numpy as np
from prody import *
import networkx as nx
import matplotlib.pyplot as plt

# store ANM normal modes for future use in cross_corr
anm_1 = {}
anm_2 = {}

# store residue numbers for future use in build_allosteric_network
residues_1 = []
residues_2 = []

# function to compute ANM alpha-carbon fluctuations
def anm_flux(pdb1, pdb2):

    # fetch first PDB file
    PrP = parsePDB(pdb1)
    if PrP is None:
        print('Failed to download or parse PDB file.')
        return

    # use alpha carbons for analysis
    calphas_1QLX = PrP.select('calpha')
    if calphas_1QLX is None:
        print(f'Unable to identify alpha carbons in the structure: {pdb1}')
        return

    # fetch second PDB file
    PrP_ab = parsePDB(pdb2)
    if PrP_ab is None:
        print(f'Failed to download or parse PDB file: {pdb2}')
        return
    truncated_PrP_ab = PrP_ab.select('resnum 121 to 230') # truncate the structure of 2KUN to provide common ROI for analysis

    calphas_2KUN = truncated_PrP_ab.select('calpha') # utlize truncated structure for alpha-carbon analysis
    if calphas_2KUN is None:
        print(f'Unable to identify alpha carbons in the structure: {pdb2}')
        return

    # align residues between the two structures to account for missing residues in 1QLX
    residues_1QLX = calphas_1QLX.getResnums()
    residues_2KUN = calphas_2KUN.getResnums()
    common_residues = np.intersect1d(residues_1QLX, residues_2KUN)

    calphas_1QLX_aligned = calphas_1QLX.select(f'resnum {" ".join(map(str, common_residues))}')
    calphas_2KUN_aligned = calphas_2KUN.select(f'resnum {" ".join(map(str, common_residues))}')

    # store aligned residue numbers for future use
    residues_1.append(calphas_1QLX_aligned.getResnums())
    residues_2.append(calphas_2KUN_aligned.getResnums())

    print(f'Aligned residues for 1QLX: {calphas_1QLX_aligned.getResnums()}')
    print(f'Aligned residues for 2KUN: {calphas_2KUN_aligned.getResnums()}')

    # perform ANM normal mode analysis on 1QLX
    calpha_coords_1QLX = prody.getCoords(calphas_1QLX_aligned)
    anm_1QLX = ANM(calphas_1QLX_aligned)
    anm_1QLX.buildHessian(calpha_coords_1QLX, cutoff = 15.0, gamma = 1.0)
    anm_1QLX.calcModes()

    # store 1QLX modes to anm_1 dictionary
    anm_1[pdb1] = anm_1QLX

    # calculate squared fluctuations
    squared_flucts_1QLX = prody.calcSqFlucts(anm_1QLX)
    max_sq_flux_1QLX = round(max(squared_flucts_1QLX), 3)
    max_residue_1QLX = squared_flucts_1QLX.argmax() + 1 # residues are 1-indexed

    print(f'Max Squared Fluctuation: {max_sq_flux_1QLX} at residue: {max_residue_1QLX}')

    calphas_1QLX_aligned.setBetas(squared_flucts_1QLX) # set the B-factor column to squared fluctuations

    mod_PDB1 = f'{pdb1}-fluctuations.pdb'
    writePDB(mod_PDB1, calphas_1QLX)

    # perform ANM normal mode analysis on 2KUN
    calpha_coords_2KUN = prody.getCoords(calphas_2KUN_aligned)
    anm_2KUN = ANM(calphas_2KUN_aligned)
    anm_2KUN.buildHessian(calpha_coords_2KUN, cutoff = 15.0, gamma = 1.0)
    anm_2KUN.calcModes()

    # store 1QLX modes to anm_1 dictionary
    anm_2[pdb2] = anm_2KUN

    # calculate squared fluctuations
    squared_flucts_2KUN = prody.calcSqFlucts(anm_2KUN)
    max_sq_fluct_2KUN = round(max(squared_flucts_2KUN), 3)
    max_residue_2KUN = squared_flucts_2KUN.argmax() + 1 # residues are 1-indexed

    print(f'Max squared fluctuation: {max_sq_fluct_2KUN} at residue: {max_residue_2KUN}')

    calphas_2KUN_aligned.setBetas(squared_flucts_2KUN)

    mod_PDB2 = f'{pdb2}-fluctuations.pdb'
    writePDB(mod_PDB2, calphas_2KUN)

    # plot the fluctuations across residues for both structures
    plt.figure(figsize = (10, 7))
    plt.plot(squared_flucts_1QLX, label = '1QLX', lw = 2, color = 'purple')
    plt.plot(squared_flucts_2KUN, label = '2KUN', lw = 2, color = 'green')
    plt.xlabel('Residue Index')
    plt.ylabel('Squared Fluctuation (log-scaled)')
    plt.yscale('log')
    plt.grid()
    plt.title('Anisotropic Network Model (ANM) Cα Fluctuations: 1QLX vs. 2KUN')
    plt.show()

# run ANM analysis on the two structures
anm_flux('1QLX', '2KUN')

# store cross-correlation matrices for future analysis in analyze_correlations and build_allosteric_network
cross_corr1 = []
cross_corr2 = []

# function to run cross-correlation analysis
def cross_corr(anm_1, anm_2):

    # calculate cross-correlation matrices for each structure
    cross_corr_1QLX = prody.calcCrossCorr(anm_1)
    cross_corr_2KUN = prody.calcCrossCorr(anm_2)

    # append matrices to external lists for use in analyze_correlations function
    cross_corr1.append(cross_corr_1QLX)
    cross_corr2.append(cross_corr_2KUN)

    # visualize each cross-correlation matrix as a heatmap
    plt.figure(figsize = (10, 5))

    # 1QLX
    plt.subplot(1, 2, 1)
    plt.imshow(cross_corr_1QLX, cmap = 'coolwarm', interpolation = 'nearest')
    plt.colorbar(shrink = 0.5)
    plt.title('1QLX')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    
    # 2KUN
    plt.subplot(1, 2, 2)
    plt.imshow(cross_corr_2KUN, cmap = 'coolwarm', interpolation = 'nearest')
    plt.colorbar(shrink = 0.5)
    plt.title('2KUN')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    
    plt.tight_layout()
    plt.show()

# run the cross-correlation analysis on the ANM objects from each dictionary
cross_corr(anm_1['1QLX'], anm_2['2KUN'])

# run analysis on the correlation matrices
def analyze_correlations(matrix, pdb_id):

    # convert lists to NumPy arrays
    matrix = np.array(matrix)

    off_diag_mask = ~np.eye(matrix.shape[0], dtype = bool) # mask: True for off-diagonal elements
    print(off_diag_mask.shape)
    off_diag_vals = matrix[off_diag_mask]

    # find the maximum and minimum off-diagonal values
    max_corr = np.max(off_diag_vals)
    min_corr = np.min(off_diag_vals)

    # indices of maximum and minimum values
    max_indices = np.unravel_index(np.argmax(matrix * off_diag_mask), matrix.shape)
    min_indices = np.unravel_index(np.argmin(matrix * off_diag_mask), matrix.shape)

    print(f'Maximum off-diagonal correlation for {pdb_id}: {max_corr:.3f} between residues {max_indices[0]+1} and {max_indices[1]+1}')
    print(f'Minimum off-diagonal correlation for {pdb_id}: {min_corr:.3f} between residues {min_indices[0]+1} and {min_indices[1]+1}')

# run the correlation analysis on each matrix (pass the matrix directly, not the list)
analyze_correlations(cross_corr1[0], '1QLX')
analyze_correlations(cross_corr2[0], '2KUN')

def build_allosteric_networks(matrix1, matrix2, res_nums1, res_nums2, threshold, pdb_id1, pdb_id2):

    G1 = nx.Graph()
    G2 = nx.Graph()

    # add nodes (residues)
    for res in res_nums1:
        G1.add_node(res)
    edge_widths1 = [] # store edge widths for network 1

    # add edges for highly correlated residue pairs
    n1 = len(res_nums1)
    for i in range(n1):
        for j in range(i + 1, n1):
            if matrix1[i, j] > threshold:
                G1.add_edge(res_nums1[i], res_nums1[j], weight = matrix1[i, j])
                edge_widths1.append(5 * matrix1[i, j])
    
    # add nodes (residues)
    for res in res_nums2:
        G2.add_node(res)
    edge_widths2 = [] # store edge widths for network 2
    
    # add edges for highly correlated residue pairs
    n2 = len(res_nums2)
    for i in range(n2):
        for j in range(i + 1, n2):
            if matrix2[i, j] > threshold:
                G2.add_edge(res_nums2[i], res_nums2[j], weight = matrix2[i, j])
                edge_widths1.append(5 * matrix2[i, j])
    
    # print network statistics
    print(f'Network for {pdb_id1}: Nodes (residues): {G1.number_of_nodes()}, Edges (interactions): {G1.number_of_edges()}')
    print(f'Network for {pdb_id2}: Nodes (residues): {G2.number_of_nodes()}, Edges (interactions): {G2.number_of_edges()}')

    # visualize the network
    pos = nx.spring_layout(G1, seed = 42, k = 1.5) # use G1's layout for consistency

    # calculate degree centrality for node sizes
    centrality1 = nx.degree_centrality(G1)
    centrality2 = nx.degree_centrality(G2)

    # determine the top nodes in each network by degree centrality
    top_hubs_1 = sorted(centrality1.items(), key = lambda x: x[1], reverse = True)[:5]
    top_hubs_2 = sorted(centrality2.items(), key = lambda x: x[1], reverse = True)[:5]

    # print the top nodes (residues)
    print('Top hubs in 1QLX by degree centrality: ', top_hubs_1)
    print('Top hubs in 2KUN by degree centrality: ', top_hubs_2)

    # scale centrality values to appropriate node sizes
    node_size1 = [250 + 5000 * centrality1[node] for node in G1.nodes()]
    node_size2 = [250 + 5000 * centrality2[node] for node in G2.nodes()]

    # visualize graph networks as two subplots
    plt.figure(figsize = (10, 7))

    # graph 1
    plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(G1, pos, node_size = node_size1, node_color = 'purple', alpha = 0.8, label = f'{pdb_id1}')
    nx.draw_networkx_edges(G1, pos, width = edge_widths1, edge_color = 'purple', alpha = 0.5)
    nx.draw_networkx_labels(G1, pos, font_size = 8, font_color = 'white')
    plt.title(f'Allosteric Network: {pdb_id1} (Threshold = {threshold})')
    plt.axis('on')

    # graph 2
    plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(G2, pos, node_size = node_size2, node_color = 'green', alpha = 0.8, label = f'{pdb_id2}')
    nx.draw_networkx_edges(G2, pos, width = edge_widths2, edge_color = 'green', alpha = 0.5)
    nx.draw_networkx_labels(G2, pos, font_size = 8, font_color = 'white')
    plt.title(f'Allosteric Network: {pdb_id2} (Threshold = {threshold})')
    plt.axis('on')

    plt.tight_layout()
    plt.show()

# build the allosteric networks for each protein state
build_allosteric_networks(cross_corr1[0], cross_corr2[0], residues_1[0], residues_2[0], threshold = 0.65, pdb_id1 = '1QLX', pdb_id2 = '2KUN')
