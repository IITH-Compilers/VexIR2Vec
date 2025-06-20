# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Reconstruction of the cfg"""

import angr
import networkx as nx


class functionCfgNormalizer:
    def removeIsolateBlocks(self, function):
        instance_graph = function.graph.copy()
        for block in list(function.block_addrs_set):

            block_node = function.get_node(block)
            try:
                if not block_node.predecessors and not block_node.successors:
                    instance_graph.remove_node(block_node)

            except nx.exception.NetworkXError or ValueError:
                pass
        return instance_graph

    def normalizeCfgToGraph(self, function):
        graph = self.removeIsolateBlocks(function)
        return graph


"""
binary_name = "readelf.out"
binary = angr.Project(binary_name, auto_load_libs=False)
binary_cfg = binary.analyses.CFGFast()
print("[*] CFG Generated Successfully")
binary_functions = binary_cfg.kb.functions.values()
special_register_offsets = [binary.arch.sp_offset, binary.arch.ip_offset, binary.arch.bp_offset]
"""
