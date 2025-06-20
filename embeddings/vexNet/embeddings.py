# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Used in the implementation of embedding tasks"""

import pyvex as py
import numpy as np
import entities as Entities
import normalize as Normalize
import keywords as Keywords
import json
import os
import re
import random
import angr
import hashlib
import random
import numpy as np


class symbolicEmbeddings:
    def __init__(self, vocab, seed_value=1004):
        self.vfile = open(vocab, "r")
        self.__sev, self.dim = self.parseSEV()
        self.keywords = Keywords.Keywords()
        self.delimiters = "=|,|\(|\)|,|\{|\}|::|->| |:|\[|\]"
        self.block_weights = {}
        self.seed_value = seed_value
        random.seed(self.seed_value)

    def generateHash(self, string):
        # Encode the string to bytes before hashing
        encoded_string = string.encode("utf-8")

        # Generate the hash using SHA-256 algorithm
        hashed_string = hashlib.sha256(encoded_string).hexdigest()

        return hashed_string

    def __lookupVocab(self, key):
        try:
            return self.__sev[key]
        except KeyError as e:
            print("Key not found in vocab - ", key)
            return np.zeros(self.dim)

    def parseSEV(self):
        embeddings = {}
        for line in self.vfile:
            print(line)
            exit(0)
            opc = line.split(":")[0].lower()
            vec = line.split(":")[1].split(",\n")[0]
            vec = vec.split(", ")
            print(vec)
            exit(0)
            vec[0] = vec[0].split("[")[1]
            vec[-1] = vec[-1].split("]")[0]
            embeddings[opc] = np.array(vec, dtype=np.float)
        return embeddings, embeddings[opc].size

    def tokenize(self, stmt):
        tokens = re.split(self.delimiters, stmt)
        tokens = [string.strip() for string in tokens if string.strip()]
        return tokens

    def canonicalizeTokens(self, tokens):
        const_pattern = r"0|1|0x[0-9a-f]+|f+"
        dec_pattern = r"d(?:32|64|128)"
        int_pattern = r"(ity_)?i(?:1|8|16|32|64|128)"
        float_pattern = r"f(?:16|32|64|128)(?:i)?"
        vec_pattern = r"v(?:128|256)"
        var_pattern = r"t[0-9]+"
        reg_pattern = r"r[0-9]+"
        jk_pattern = r"ijk_[a-z]+"
        new_tokens = []
        for token in tokens:
            if re.match(const_pattern, token):
                new_tokens.append("constant")
            elif re.match(int_pattern, token):
                new_tokens.append("integer")
            elif re.match(float_pattern, token):
                new_tokens.append("float")
            elif re.match(vec_pattern, token):
                new_tokens.append("vector")
            elif re.match(dec_pattern, token):
                new_tokens.append("decimal")
            elif re.match(var_pattern, token):
                new_tokens.append("variable")
            elif re.match(reg_pattern, token):
                new_tokens.append("register")
            elif re.match(jk_pattern, token):
                new_tokens.append("jump_kind")
            else:
                new_tokens.append(token.lower())
        return new_tokens

    def randomWalk(self, func, block_addresses, k=72, n=2):
        random.seed(1004)
        np.random.seed(1004)
        func_block_freq = dict()
        walk_visited_blocks_list = []  # list of random walks

        blocks = []

        # Putting all block of size>0 in a list
        for block in func.graph:
            if block.size > 0 and block.addr in block_addresses:
                blocks.append(block)

                # Continue until each block of the function is visited in atleast n random walks:
        while True:
            # Choosing possible starting blocks
            starting_blocks = [
                block
                for block in blocks
                if block not in func_block_freq or func_block_freq[block] < n
            ]

            if (
                len(starting_blocks) == 0
            ):  # each block is visited in atleast n random walks
                break

                # Start with a random block
            current_block = random.choice(starting_blocks)

            walk_visited_blocks = []

            num_vis = 0

            # Starting a random walk...

            # Continue current walk until the desired number of blocks are visited or all blocks are visited
            while num_vis < k and current_block not in walk_visited_blocks:
                # updating freq of current block
                if current_block in func_block_freq:
                    func_block_freq[current_block] += 1
                else:
                    func_block_freq[current_block] = 1

                walk_visited_blocks.append(current_block)

                if current_block.size > 0:
                    num_vis += 1

                try:
                    # Check if the current block has a single successor
                    if len(current_block.successors()) == 1:
                        successors_list = current_block.successors()
                        current_block = successors_list[0]
                        if current_block.addr not in block_addresses:
                            break

                            # If the current block has multiple successors (branches), choose one randomly
                    elif len(current_block.successors()) > 1:
                        current_block = random.choice(current_block.successors())
                        if current_block.addr not in block_addresses:
                            break

                    else:  # No successor of current block
                        break

                except Exception as err:
                    break

            walk_visited_blocks_list.append(walk_visited_blocks)
        return walk_visited_blocks_list

    def processFunc(self, func, normalize, bbfreq_dict=None):

        func_O_Vec = np.zeros(self.dim)
        func_T_Vec = np.zeros(self.dim)
        func_A_Vec = np.zeros(self.dim)

        block_addresses = [block.addr for block in func.blocks]
        bbWalks = self.randomWalk(func, block_addresses)

        addr_blk_dict = {}
        for block in func.blocks:
            addr_blk_dict[block.addr] = block

        for bbwalk in bbWalks:
            self.norm = Normalize.Normalizer(normalize)

            for blocknode in bbwalk:

                block_O_Vec = np.zeros(self.dim)
                block_T_Vec = np.zeros(self.dim)
                block_A_Vec = np.zeros(self.dim)
                block = addr_blk_dict[blocknode.addr]
                inst_list = []
                try:
                    for stmt in block.vex.statements:
                        stmt_str = str(stmt).lower()
                        tokens = self.tokenize(stmt_str)

                        # angr's VEX IR does not handle all the cases
                        if stmt.tag == "Ist_Dirty":
                            continue

                        func_flag = False
                        puti_flag = False
                        geti_flag = False
                        if stmt.tag == "Ist_WrTmp":
                            if stmt.data.tag == "Iex_CCall":
                                func_flag = True
                                # tokens[1] = "function"
                            elif stmt.data.tag == "Iex_GetI":
                                geti_flag = True
                                # tokens[2] = "r1234" # random register in place of array
                                # tokens[3] = "remove" # workaround for removing next token; will be removed in get_keyword

                        if stmt.tag == "Ist_PutI":
                            puti_flag = True
                            # tokens[1] = "r1234" # random register in place of array
                            # tokens[2] = "remove" # workaround for removing next token; will be removed in get_keyword

                        new_stmt = stmt_str

                        for i, token in enumerate(tokens):
                            replace_token = token
                            try:
                                if i == 1:
                                    if func_flag == True:
                                        replace_token = "function"
                                    elif puti_flag == True:
                                        replace_token = "r1234"
                                elif i == 2:
                                    if geti_flag == True:
                                        replace_token = "r1234"
                                    elif puti_flag == True:
                                        replace_token = "remove"
                                elif i == 3 and geti_flag == True:
                                    replace_token = "remove"

                                new_stmt = new_stmt.replace(
                                    token, self.keywords.get_keyword(replace_token), 1
                                )

                            except KeyError as e:
                                # VARIABLE, CONSTANT, REGISTER, INTEGER, FLOAT, VECTOR, DECIMAL are not in keywords dict; so allow them
                                pass

                        inst_list.append(new_stmt)

                except angr.errors.SimEngineError as e:
                    continue
                normalized_inst_list = self.norm.transform_window(inst_list)

                for normalized_stmt in normalized_inst_list:
                    stmt_O_Vec = np.zeros(self.dim)
                    stmt_T_Vec = np.zeros(self.dim)
                    stmt_A_Vec = np.zeros(self.dim)
                    if normalized_stmt is None or normalized_stmt == "":
                        continue

                    tokens = self.tokenize(normalized_stmt)
                    tokens = self.canonicalizeTokens(tokens)

                    if stmt.tag == "Ist_WrTmp" and stmt.data.tag == "Iex_CCall":
                        tokens[1] = "function"

                    for token in tokens:
                        val_O = np.zeros(self.dim)
                        val_T = np.zeros(self.dim)
                        val_A = np.zeros(self.dim)
                        if token in [
                            "register",
                            "integer",
                            "float",
                            "vector",
                            "decimal",
                        ] or token.startswith("ity_"):
                            val_T = self.__lookupVocab(token)
                        elif token in [
                            "variable",
                            "constant",
                            "jump_kind",
                            "none",
                            "dirty",
                            "exit",
                        ] or token.startswith(
                            (
                                "ilgop_",
                                "imbe_",
                                "catoddlanesv",
                                "catevenlanesv",
                                "ctz",
                                "qnarrowbin",
                                "clz",
                            )
                        ):
                            val_A = self.__lookupVocab(token)
                        else:
                            val_O = self.__lookupVocab(token)

                        stmt_O_Vec += val_O
                        stmt_T_Vec += val_T
                        stmt_A_Vec += val_A

                    block_O_Vec += stmt_O_Vec
                    block_T_Vec += stmt_T_Vec
                    block_A_Vec += stmt_A_Vec

                bbstr = "|".join(normalized_inst_list)

                bbstr_hash = self.generateHash(bbstr)

                if bbfreq_dict is not None:
                    if bbstr_hash in bbfreq_dict:

                        func_O_Vec += block_O_Vec / bbfreq_dict[bbstr_hash]
                        func_T_Vec += block_T_Vec / bbfreq_dict[bbstr_hash]
                        func_A_Vec += block_A_Vec / bbfreq_dict[bbstr_hash]
                    else:
                        func_O_Vec += block_O_Vec
                        func_T_Vec += block_T_Vec
                        func_A_Vec += block_A_Vec
                else:
                    func_O_Vec += block_O_Vec
                    func_T_Vec += block_T_Vec
                    func_A_Vec += block_A_Vec

        return func_O_Vec, func_T_Vec, func_A_Vec
