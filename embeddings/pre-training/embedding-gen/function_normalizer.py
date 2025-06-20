# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Used to decide helpful functions"""

import angr


class functionNormalizer:
    def __init__(self, binary_project):
        self._special_register_offsets = [
            binary_project.arch.sp_offset,
            binary_project.arch.ip_offset,
            binary_project.arch.bp_offset,
        ]

    def findDumbFunction(self, function):
        # True: The Function is not useful
        # False: The Functions is useful
        dumb_flag = False
        if len(function.block_addrs_set) == 1:
            block = next(function.blocks)
            try:
                if block.vex.jumpkind in ["Ijk_Boring", "Ijk_Ret"]:
                    dumb_flag = True
            except angr.errors.SimTranslationError:
                dumb_flag = True
            except angr.errors.SimEngineError as e:
                dumb_flag = True
        return dumb_flag

    def recurseVexStatement(self, statement):
        # True: general registers used
        # False: No general registers used
        try:
            expressions = next(iter(statement.expressions))
            bool_val = self.recurseVexStatement(expressions)
            return bool_val
        except AttributeError:
            try:
                if statement.offset not in self._special_register_offsets:
                    return True
            except AttributeError:
                return False
        except StopIteration:
            try:
                if statement.offset not in self._special_register_offsets:
                    return True
            except AttributeError:
                return False

    def isNotUsefulFunction(self, function):
        # True: The function is not useful
        # False: the function is useful
        dumb_flag = self.findDumbFunction(function)
        if dumb_flag:
            block = next(function.blocks)
            try:
                vex_stmt = block.vex.statements
                if any(stmt.tag != "Ist_IMark" for stmt in vex_stmt):
                    for stmt in vex_stmt:
                        stmt_flag = self.recurseVexStatement(stmt)
                        if stmt_flag:
                            dumb_flag = False
                            break
            except angr.errors.SimTranslationError:
                pass
            except angr.errors.SimEngineError:
                pass

        return dumb_flag


"""
#Driver Code
binary_name = "readelf.out"
binary = angr.Project(binary_name, auto_load_libs=False)
binary_cfg = binary.analyses.CFGFast()
print("[*] CFG Generated Successfully")
binary_functions = binary_cfg.kb.functions.values()


verifier = functionNormalizer(binary)
yes=0
no=0
for func in binary_functions:
    flag = verifier.is_not_useful_function(func)
    if(flag):
        no+=1
    else:
        yes+=1
print("Useful:",yes)
print("Dumb:",no)
"""
