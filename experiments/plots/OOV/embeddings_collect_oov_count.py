# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Symbolic Embeddings Class for Collecting OOV Counts"""

import pyvex as py
import numpy as np
import entities_collect_oov_count as Entities


class symbolicEmbeddings:
    def __init__(self, vocab):
        self.vfile = open(vocab, "r")
        self.__sev, self.dim = self.parseSEV()
        self.wo = 1
        self.wt = 0.5
        self.wa = 0.2
        self.wl = 0.2
        self.wrhs = 1
        self.unknown_count = 0
        self.unk_key_dict = {}

    def __lookupVocab(self, key):
        try:
            return self.__sev[key]
        except KeyError as e:
            if key not in self.unk_key_dict.keys():
                self.unk_key_dict[key] = 0
            self.unk_key_dict[key] += 1
            self.unknown_count += 1
            return np.zeros(self.dim)

    def parseSEV(self):
        embeddings = {}
        for line in self.vfile:
            opc = line.split(":")[0]
            vec = line.split(":")[1].split(",\n")[0]
            vec = vec.split(", ")
            vec[0] = vec[0].split("[")[1]
            vec[-1] = vec[-1].split("]")[0]
            embeddings[opc] = np.array(vec, dtype=np.float)
        return embeddings, embeddings[opc].size

    def processExpTriplet(self, triplet):
        assert type(triplet) is dict
        opc = triplet["opc"]
        ty = triplet["type"]
        args = triplet["arg"]

        if opc is not None:
            self.__lookupVocab(opc)

            if ty is not None and ty != "":
                self.__lookupVocab(ty)

            for arg in args:
                if isinstance(arg, str):
                    if arg != "":
                        self.__lookupVocab(arg)
                else:
                    self.processExpTriplet(arg)

    def processStmt(self, triplet):
        opc = triplet["opc"]
        loc = triplet["loc"]
        rhs = triplet["rhs"]
        ty = triplet["type"]
        args = triplet["arg"]

        if opc != "":
            self.__lookupVocab(opc)

            if loc != "":
                if isinstance(loc, str):
                    self.__lookupVocab(loc)
                else:
                    self.processExpTriplet(loc)

            if rhs != "":
                if isinstance(rhs, str):
                    self.__lookupVocab(rhs)
                elif isinstance(rhs, dict):
                    self.processExpTriplet(rhs)
                else:
                    assert isinstance(rhs, list)
                    for r in rhs:
                        self.processStmt(r)

            if ty != "":
                self.__lookupVocab(ty)

            for arg in args:
                if isinstance(arg, str):
                    if arg != "":
                        self.__lookupVocab(arg)
                else:
                    self.processExpTriplet(arg)

    def processSB(self, block):
        sbVec = np.zeros(self.dim)
        triplets = Entities.Entities().processSB(block)
        if triplets is None:
            return
        for triplet in triplets:
            if triplet == "IRSB":
                continue
            opc = triplet["opc"]
            if opc is not None:
                self.processStmt(triplet)
        return sbVec

    def processFunc(self, func, genSBVec=False):
        if genSBVec:
            sbVecs = []
            for block in func.blocks:
                if block.size > 0:
                    self.processSB(block)
            return sbVecs

        funcVec = np.zeros(self.dim)
        triplets = Entities.Entities().processFunc(func)
        if triplets is None:
            return
        for triplet in triplets:
            if triplet == "IRSB":
                continue

            opc = triplet["opc"]
            if opc is not None:
                self.processStmt(triplet)
        return funcVec
