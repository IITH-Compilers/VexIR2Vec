# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Normalization of vectors"""

import angr
import sys
import os
import re


class normalizer:
    def __init__(self, level=1, num_passes=2):
        self.initialize()
        self.level = level
        self.initRegex()
        self.num_passes = num_passes

        # table for regex patterns

    def initRegex(self):
        self.regex = {
            "hex_match": "(0x[0-9a-f]+)",
            "negative_hex": "0x[89a-f][0-9a-f]*",
            "offset": "offset=",
            "r": "r",
            "var": "(t\d+)",
            "var_r": "(t\d+)",
            "def_var": "(t\d+) = .*",
            "def_var_r": "(t\d+) = .*",
            "nan": "(?<!\w)nan(?!\w)",  #  with lookarounds
            # 1
            "add_sub_match": "(add|sub)\((.*),(.*)\)",
            "add_detect": "add",  # for detecting in line
            "add_instr": "add",  # for replacing in regex
            "sub_detect": "sub",
            "sub_instr": "sub",
            # 2
            "conversion_match": "= .*(trunc|ext).*\((.+)\)",
            "conversion_sub": "\w*(trunc|ext).*\((.+)\)",
            # 3, 7
            "load_detect": "load",
            "load_sub": "load.*\(.*\)",
            "load_match": "= load.*\((.*)\)",
            "store_detect": "store",
            "store_sub": "store\(.*\)",
            "store_match": "store\((.*)\) =",
            "get_detect": "get",
            "get_sub": "get.*\(.*\)",
            "get_match": "= get.*\((.*)\)",
            "put_detect": "put(",
            "put_sub": "put\(.*\)",
            "put_match": "put\((.*)\) =",
            # 4
            "copy_match_r": "(t\d+) = (t\d+)",
            "no_digit_ahead": "(?!\d)",
            "sub_left_positive": "sub.*\(t\d+,",
            # 5.
            "mul_detect": "mul",
            "div_detect": "div",
            # 6
            "assign_const": "= 0x[0-9a-f]+",
            "assign_var": "= t\d+",
            "expression_match": "t\d+ = (.*)",
            "LHS_expr_match": "(t\d+) = (.*)",
            # 7
            "RHS_match": "= (.*)",
            "LHS_match_r": "(t\d+) =",
        }

        # considering registers in higher levels
        if self.level >= 3:
            optional = {
                "var_r": "([rt]\d+)",
                "def_var_r": "([rt]\d+) = .*",
                "LHS_match_r": "([rt]\d+) =",
                "copy_match_r": "([rt]\d+) = ([rt]\d+)",
            }
            self.regex.update(optional)

            # generate empty dictionaries

    def initialize(self):
        # stores original variable for a copy
        self.copies_dict = dict()
        # stores value of a variable
        self.const_dict = dict()
        # stores (base,offset) for a variable used in linear calculations
        self.offset_dict = dict()
        # stores original variable which contains a subexpression
        self.expr_dict = dict()
        # stores content of memory
        self.memory_dict = dict()
        # stores line numbers for ununsed variable declarations
        self.unused_dict = dict()
        # stores: prev_use, prev_line_num, first_use
        self.address_dict = dict()
        self.register_dict = dict()

        self.loaded_dict = dict()
        self.direct_address = 1 << 24

        self.line_number = 0

    def cleanLines(self, line):
        if "------ imark" in line or "abihint" in line:
            return None
        if self.regex["offset"] in line:
            line = re.sub(self.regex["offset"], self.regex["r"], line)

        return line

    def hexToDec(self, hexstr, nibbles=None):
        if hexstr == "nan":
            return "nan"
        num = int(hexstr, 16)
        if re.search(self.regex["negative_hex"], hexstr):
            if nibbles is None:
                nibbles = len(hexstr) - 2
            ceiling = 1 << 4 * nibbles
            # get 2s complement of negative number, then put negative sign in decimal
            num = -(ceiling - num)
        return num

    def decToHex(self, num, nibbles=16):
        if num == "nan":
            return "nan"
        if num < 0:
            ceiling = 1 << 4 * nibbles
            num = (
                ceiling + num
            )  # take 2s complement of -num ( -num is positive in decimal)
        hexstr = ("{:0" + str(nibbles) + "x}").format(num)
        if len(hexstr) - 2 > nibbles:
            hexstr = hexstr[-nibbles:-1]  # trunc if needed
        return "0x" + hexstr

    def negateHex(self, hexstr, nibbles=16):
        if hexstr == "nan":
            return "nan"
        if nibbles is None:
            nibbles = len(hexstr) - 2
        num = int(hexstr, 16)
        ceiling = 1 << 4 * nibbles  # ex:for 32 bits: 100000000
        pos = ceiling - num  # 2's complement
        # for padding with correct 0s
        return ("0x{:0" + str(nibbles) + "x}").format(pos)

    def getDefUseVars(self, line, registers=True, address=True):

        # get rid of addresses
        if not address and "store" in line:
            line = re.sub(self.regex["store_sub"], "store", line)
        if not address and "load" in line:
            line = re.sub(self.regex["load_sub"], "load", line)

        if registers:
            use_vars = re.findall(self.regex["var_r"], line)
            def_var = re.findall(self.regex["def_var_r"], line)
        else:
            use_vars = re.findall(self.regex["var"], line)
            def_var = re.findall(self.regex["def_var"], line)

        if len(def_var) > 0:
            use_vars.remove(def_var[0])
        return def_var, use_vars

        # -----------------------SIMPLIFICATION------------------------------

        # 1. switches between add and sub to obtain positive immediate

    def removeNegativeImm(self, line):
        if line is None:
            return line
        instr = re.findall(self.regex["add_sub_match"], line)
        if len(instr) == 0:
            return line
        instr_type = instr[0][0]  # instr not found
        left_op = instr[0][1]
        right_op = instr[0][2]

        negatives = re.findall(self.regex["negative_hex"], line)
        if len(negatives) != 1:
            return line  # check for exactly 1 negative imm
        positive = self.negateHex(negatives[0])

        if instr_type == self.regex["add_instr"]:
            line = re.sub(self.regex["add_instr"], self.regex["sub_instr"], line)
            # add(a, -b) --> sub(a, b)
            if negatives[0] == right_op:
                line = re.sub("," + right_op, "," + positive, line)
                # add(-a, b) --> sub(b, a)
            elif negatives[0] == left_op:
                line = re.sub(left_op + "," + right_op, right_op + "," + positive, line)
                # sub(a, -b) --> add(a+b)
        elif instr_type == self.regex["sub_instr"] and negatives[0] == right_op:
            line = re.sub(self.regex["sub_instr"], self.regex["add_instr"], line)
            line = re.sub("," + right_op, "," + positive, line)  # substitutions

        return line

        # 2. removes size conversions/truncations

    def removeConversions(self, line):
        if line is None:
            return line

        matches = re.findall(self.regex["conversion_match"], line)
        if len(matches) != 1:
            return line
        return re.sub(self.regex["conversion_sub"], matches[0][1], line)

        # returns bool decision to keep current line

    def removeUnusedPut(self, register, line):
        if register in self.register_dict:
            line_num, last_use = self.register_dict[register]
            # register last used in put -> old definition unused and maybe erased
            if last_use == "put":
                self.block_result[line_num] = None
        return

        # 3. simplification by removing get/put

    def registerOptimization(self, line, mode="gp"):
        if line is None:
            return line

            # register promotion
        if "g" in mode and self.regex["get_detect"] in line:
            match = re.findall(self.regex["get_match"], line)
            if len(match) > 0:
                register = match[0]
                line = re.sub(self.regex["get_sub"], register, line)

            if self.level >= 1:
                self.register_dict[register] = (self.line_number, "get")

        if "p" in mode and self.regex["put_detect"] in line:
            match = re.findall(self.regex["put_match"], line)
            if len(match) > 0:
                register = match[0]
                line = re.sub(self.regex["put_sub"], register, line)

            if self.level >= 1:
                self.removeUnusedPut(register, line)
                self.register_dict[register] = (self.line_number, "put")

        return line

        # 4. substitute copies with original variable

    def copyPropagation(self, line):
        if line is None:
            return line

        _, use_vars = self.getDefUseVars(line)
        for v in use_vars:
            if v in self.copies_dict and v != self.copies_dict[v]:
                line = re.sub(
                    v + self.regex["no_digit_ahead"], self.copies_dict[v], line
                )  # substitutions
            else:
                self.copies_dict[v] = v

        if "if" in line:
            return line

            # n1. invalid LHS
        left_var = re.findall(self.regex["LHS_match_r"], line)  # For re-assignment
        if (
            len(left_var) == 1 and left_var[0] in self.copies_dict
        ):  # delete past value for reassignment
            del self.copies_dict[left_var[0]]

        copy_instr = re.findall(self.regex["copy_match_r"], line)

        if len(copy_instr) != 0:
            LHS = copy_instr[0][0]
            RHS = copy_instr[0][1]
            # initialize right vars appearing for first time
            if RHS not in self.copies_dict:
                self.copies_dict[RHS] = RHS

            if self.regex["r"] in self.copies_dict[RHS]:
                self.copies_dict[LHS] = LHS

            else:  # "t" in self.copies_dict[RHS]:
                self.copies_dict[LHS] = self.copies_dict[
                    RHS
                ]  # LHS derived from parent of RHS
                if "t" in LHS:
                    return None  # does not erase put statements. They are handled separately

        return line

        # 5. substitute and solve for constants (inc. nan)

    def constantPropagation(self, line):
        if line is None:
            return line
        _, use_vars = self.getDefUseVars(line)
        for v in use_vars:
            if v in self.const_dict:
                # lookahead to not sub t5 in t52
                line = re.sub(
                    v + self.regex["no_digit_ahead"],
                    self.decToHex(self.const_dict[v]),
                    line,
                )

        if "if" in line or "=" not in line:
            return line
            # no propagation in if case

            # n1. invalid LHS
        left_var = re.findall(self.regex["LHS_match_r"], line)  # Assignment
        if len(left_var) == 0:
            return line

        if left_var[0] in self.const_dict:  # delete past value for reassignment
            del self.const_dict[left_var[0]]

            # p1. store nan
        if re.search(self.regex["nan"], line):
            self.const_dict[left_var[0]] = "nan"
            return None

            # n2. RHS is not constant if a variable still exists
        if re.search(self.regex["var_r"], line.split("=")[1]):
            return line

        calculated = self.calcInstr(line)
        # n3. RHS cannot be solved
        if calculated is None:
            return line
            # p2. store calculated const
        self.const_dict[left_var[0]] = calculated  # add to const dict

        if self.regex["r"] in left_var[0]:
            return line  # does not erase put statements. They are handled separately

        return None

        # x. tracks linear relations for subexpr detection

    def offsetEvaluation(self, line):
        if line is None:
            return line

        right_side = line.split("=")[1]
        left_side = line.split("=")[0]
        right_vals = re.findall(self.regex["hex_match"], right_side)  # constants in RHS

        offset_change = None
        # add(var, const) and add(const, var)
        if self.regex["add_detect"] in right_side and len(right_vals) == 1:
            offset_change = self.hexToDec(right_vals[0])

            # sub(var, const)
        elif (
            re.search(self.regex["sub_left_positive"], right_side)
            and len(right_vals) == 1
        ):
            offset_change = -self.hexToDec(right_vals[0])

        right_var = re.findall(self.regex["var"], right_side)
        left_var = re.findall(self.regex["var"], left_side)
        # only run for above two cases
        if offset_change is not None and len(right_var) == 1 and len(left_var) == 1:
            right_var = right_var[0]
            left_var = left_var[0]

            # init tuple (original copy of x, 0) for x
            if right_var not in self.offset_dict:
                original = self.copies_dict[right_var]
                self.offset_dict[right_var] = (original, 0)

                # right_var is (base, old_offset)
            new_offset = self.offset_dict[right_var][1] + offset_change
            # new_offset = old_offset + offset_change
            base = self.offset_dict[right_var][0]
            self.offset_dict[left_var] = (base, new_offset)

            line = re.sub(
                right_var + self.regex["no_digit_ahead"], base, line
            )  # normalize base+offset
            if self.regex["add_detect"] in right_side:
                line = re.sub(right_vals[0], self.decToHex(new_offset), line)
            elif self.regex["sub_detect"] in right_side:
                line = re.sub(right_vals[0], self.decToHex(-new_offset), line)

            if (new_offset >= 0 and self.regex["sub_detect"] in right_side) or (
                new_offset < 0 and self.regex["add_detect"] in right_side
            ):
                line = self.removeNegativeImm(line)  # due to new offset
        return line

        # 6,

    def subexpressionMatching(self, line):
        if line is None:
            return line
        if re.search(self.regex["assign_var"], line):
            return line
        if re.search(self.regex["assign_const"], line):
            return line

        RHS = re.findall(self.regex["expression_match"], line)
        if len(RHS) != 0 and RHS[0] in self.expr_dict:
            line = line.replace(RHS[0], self.expr_dict[RHS[0]])  # RHS -> expr_dict[RHS]

        if line is None or "if" in line:
            return line
            # can comment these ifs
        if re.search(self.regex["assign_var"], line):
            return line
        if re.search(self.regex["assign_const"], line):
            return line

        matches = re.findall(self.regex["LHS_expr_match"], line)

        if len(matches) != 0:
            RHS = matches[0][1]
            LHS = matches[0][0]
            self.expr_dict[RHS] = LHS  # expr[RHS] = LHS

        return line

        # removes last memory usage of a location, of a particular type (load/store)

    def erasePrevMemUse(self, address, use_type):
        if address in self.address_dict:
            prev_use, line_num, _ = self.address_dict[address]
            if prev_use == use_type:
                self.block_result[line_num] = None
        return

        # converts indirect addressing to direct address

    def indirectToDirect(self, address, line):
        if address in self.loaded_dict:
            line_num = self.loaded_dict[address]
            self.block_result[line_num] = None
            self.const_dict[address] = self.direct_address
            self.direct_address += 8
        return line  # self.constant_substitution(line)  # was sub earlier

        # 7 prevents storing and loading of consts to/from memory

    def reduceLoadStore(self, line):  # not all
        if line is None:
            return line

        if "load" in line:
            # address maybe var or const
            try:
                match = re.findall(self.regex["load_match"], line)
                if len(match) == 0:
                    return line
                address = match[0]
                line = self.indirectToDirect(address, line)
            except:
                print(line)
                raise IndexError

                # substitute from memory_dict, address already seen before
            if address in self.address_dict:
                if address in self.memory_dict:
                    line = re.sub(
                        self.regex["load_sub"], self.memory_dict[address], line
                    )
                    # erasePrevMemUse(address, 'store')
                if self.level >= 3:
                    self.erasePrevMemUse(address, "store")
                self.address_dict[address][0:2] = ["load", self.line_number]

            else:
                self.address_dict[address] = ["load", self.line_number, "load"]
                # TESTING: tracks use of address
                # tracking variables loaded from outside
                def_var, _ = self.getDefUseVars(line)
                try:
                    self.loaded_dict[def_var[0]] = self.line_number
                except IndexError:
                    pass

        elif "store" in line:
            RHS = re.findall(self.regex["RHS_match"], line)
            if len(RHS) == 1:
                match = re.findall(self.regex["store_match"], line)
                if len(match) == 0:
                    return line
                address = match[0]
                line = self.indirectToDirect(address, line)

                # earlier store gets overwritten
                self.erasePrevMemUse(address, "store")

                # store RHS in memory_dict
                self.memory_dict[address] = RHS[0]
                if address in self.address_dict:
                    self.address_dict[address][0:2] = ["store", self.line_number]
                else:
                    self.address_dict[address] = ["store", self.line_number, "store"]
                    # TESTING: tracks use of address

        return line

        # constant folding

    def calcInstr(self, line):
        # print(line, 'is const' )        # DEBUG
        right_side = line.split("=")[1]
        right_vals = re.findall(self.regex["hex_match"], right_side)
        res = None

        if " " + self.regex["add_detect"] in line:
            res = self.hexToDec(right_vals[0]) + self.hexToDec(right_vals[1])
        elif " " + self.regex["sub_detect"] in line:
            res = self.hexToDec(right_vals[0]) - self.hexToDec(right_vals[1])
        elif " " + self.regex["mul_detect"] in line:
            res = self.hexToDec(right_vals[0]) * self.hexToDec(right_vals[1])
        elif " " + self.regex["div_detect"] in line:
            if self.hexToDec(right_vals[1]) != 0:
                res = "nan"
            else:
                res = self.hexToDec(right_vals[0]) // self.hexToDec(right_vals[1])
        elif re.search(self.regex["assign_const"], line):  # exactly one
            res = self.hexToDec(right_vals[0])
        return res

        # 8. track use def (ignoring inside load/store)

    def trackUseDef(self, line):
        # note: we cannot just check RHS for uses, eg: if() statements
        if line is None:
            return line

        def_var, use_vars = self.getDefUseVars(line, registers=False, address=False)
        if len(def_var) != 0:
            # store def line in case it remains unused
            self.unused_dict[def_var[0]] = self.line_number

        for v in use_vars:
            if v in self.unused_dict:
                self.unused_dict.pop(v)  # Use
        return

        # gets rid of useless calculations (such as pointer arithmetic after load/store removal)

    def removeUselessDefs(self, mode="a"):
        if mode == "n":
            return  # modes: n-no, a-address only, y-yes

        for useless_var in self.unused_dict:
            # gives line number of definition
            i = self.unused_dict[useless_var]

            address_loaded = (
                useless_var in self.address_dict
                and self.address_dict[useless_var][0]
                == "load"  # last use is load -> not exporting
                and self.address_dict[useless_var][2]
                == "store"  # first use is store -> not importing
            )
            if mode == "y" or (mode == "a" and address_loaded):
                self.block_result[i] = None
                # self.block_result[i] += " --- > DELETED USELESS DEF"
        return

    def canonicalizeLine(self, line):
        line = self.cleanLines(line)  # 1.0 margins, debug lines
        if line is None:
            return line  # directly skip empty lines

        if self.level <= 0:
            return line
        line = self.removeNegativeImm(line)  # 1.1
        line = self.removeConversions(line)  # 1.2
        return line

        # applies optimizations on a line

    def transformLine(self, line, iter=0):

        # 1. Cleaning and canonicalization only in first iteration
        if iter == 0:
            line = self.canonicalizeLine(line)

        if self.level > -1:
            # 2. register optimization, remove useless puts
            line = self.registerOptimization(line, mode="gp")

            # RHS Evaluation
            # 3. substituting copies with original
            line = self.copyPropagation(line)

        if self.level >= 2:
            # 4. evaluation of instr with consTants
            line = self.constantPropagation(line)

        if self.level >= 3:
            # 5. simplifies add/sub, brings common_base + offset
            line = self.offsetEvaluation(line)

        if self.level >= 2:
            # 6. replaces RHS if it has been defined before
            line = self.subexpressionMatching(line)

        if self.level >= 2:
            # 7. stores and loads values from python dict
            line = self.reduceLoadStore(line)

        if self.level >= 3:
            self.trackUseDef(line)

        return line

        # apply transformations over each line

    def runPass(self, stmt_list, iter=0):
        self.initialize()

        # self.IRSB = False
        self.block_result = []

        for line in stmt_list:
            result = self.transformLine(str(line), iter)
            if result is not None:
                self.block_result.append(result)
                self.line_number += 1

        if self.level >= 3:
            self.removeUselessDefs(mode="a")  # 8b. n-no, y-yes, a-address only

        self.block_result = [x for x in self.block_result if x is not None]  # cleanup
        return self.block_result

        # apply iterations of pass

    def transformWindow(self, stmt_list):
        for i in range(self.num_passes):
            prev_hash = hash(str(stmt_list))
            stmt_list = self.runPass(stmt_list, iter=i)

            # early exit is hash changes
            new_hash = hash(str(stmt_list))
            # print(prev_hash, new_hash, i)
            if prev_hash == new_hash:
                break
            prev_hash = new_hash

        return stmt_list
