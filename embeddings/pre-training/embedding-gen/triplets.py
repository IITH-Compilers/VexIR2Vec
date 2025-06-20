# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Used to process triplets"""

import pyvex as py
import entities as Entities


class triplets:
    def __init__(self):
        self.tStr = ""
        self.entities = Entities.entities()

    def processExpTriplet(self, triplet, tString="", prevOpc="", k=1, rec=False):
        assert type(triplet) is dict
        opc = triplet["opc"]
        ty = triplet["type"]
        args = triplet["arg"]
        if opc is not None:
            if rec:
                if prevOpc == "":
                    tString += " " + opc + "\n"
                else:
                    # tString += "here2\n"
                    tString += prevOpc + " ARG" + str(k) + " " + opc + "\n"
            if ty is not None and ty != "":
                tString += opc + " TYPE " + ty + "\n"
            i = 1
            for arg in args:
                if isinstance(arg, str):
                    if arg != "":
                        # tString += "here3\n"
                        tString += opc + " ARG" + str(i) + " " + arg + "\n"
                else:
                    tString = self.processExpTriplet(arg, tString, opc, i, True)
                i = i + 1
        return tString

    def printIRExpr(self, er: py.expr.IRExpr, isRHS: bool, irsb):
        triplets = self.entities.processIRExpr(er, isRHS, irsb)
        tString = self.processExpTriplet(triplets)
        self.tStr += tString
        return

    def processStmt(self, triplet, tString=""):  # handle next
        opc = triplet["opc"]
        loc = triplet["loc"]
        rhs = triplet["rhs"]
        ty = triplet["type"]
        args = triplet["arg"]
        # print(triplet)
        # sdf
        if opc != "":
            if loc != "":
                # print(type(loc))
                if isinstance(loc, str):
                    tString += opc + " LOC " + loc + "\n"
                else:
                    tString += opc + " LOC " + loc["opc"] + "\n"
                    tString += self.processExpTriplet(loc, "", opc)
                    # else:
                    #     tString += opc + " LOC NONE" + "\n"
            if rhs != "":
                if isinstance(rhs, str):
                    tString += opc + " RHS " + rhs + "\n"
                elif isinstance(rhs, dict):
                    tString += opc + " RHS " + rhs["opc"] + "\n"
                    tString += self.processExpTriplet(rhs, "", opc)
                else:
                    assert isinstance(rhs, list)
                    prevrhs = ""
                    for r in rhs:
                        if prevrhs != "":
                            tString += prevrhs + " NEXT " + r["opc"] + "\n"
                        tString += opc + " RHS " + r["opc"] + "\n"
                        prevrhs = r["opc"]
                        tString = self.processStmt(r, tString)
            if ty != "":
                tString += opc + " TYPE " + ty + "\n"
            i = 1
            for arg in args:
                # print('here')
                if isinstance(arg, str):
                    # tString += str(arg) + "\n"
                    if arg != "":
                        # tString += "here1\n"
                        tString += opc + " ARG" + str(i) + " " + arg + "\n"
                else:
                    # tString += "here##\n"
                    tString += opc + " ARG" + str(i) + " " + arg["opc"] + "\n"
                    tString += self.processExpTriplet(arg, "", opc, i)
                i = i + 1
                # sdsa
        return tString

    def processFunc(self, func):
        self.tStr = ""
        prevOpc = ""
        # {'opc' : "", 'loc':"", 'rhs':"", 'type': "", 'arg' : []}
        triplets = self.entities.processFunc(func)
        # print("name - ", func.name)
        if triplets is None:
            return
        for triplet in triplets:
            if triplet == "IRSB":
                if prevOpc != "":
                    self.tStr += prevOpc + " NEXT NONE\n"
                    prevOpc = ""
                continue
                # if prevOpc != "":
                #     self.tStr += prevOpc + "\n"
                # self.tStr += "-----------------------------\n"
                # self.tStr += str(triplet) + "\n"
            opc = triplet["opc"]
            loc = triplet["loc"]
            rhs = triplet["rhs"]
            ty = triplet["type"]
            args = triplet["arg"]

            if opc is not None:
                if prevOpc != "":
                    self.tStr += prevOpc + " NEXT " + opc + "\n"
                prevOpc = opc
                self.tStr += self.processStmt(triplet)
                # self.tStr += "-----------------------------\n"
        if prevOpc != "":
            self.tStr += prevOpc + " NEXT NONE\n"
        return self.tStr
        # print("written")
