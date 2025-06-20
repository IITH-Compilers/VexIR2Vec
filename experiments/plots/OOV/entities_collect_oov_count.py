# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""VexIR2Vec Symbolic Embeddings Class for Collecting OOV Counts"""

import pyvex as py


class entities:
    def processIrop(self, op: str):
        if op in ["Iop_Add8", "Iop_Add16", "Iop_Add32", "Iop_Add64"]:
            return "Add"

        if op in ["Iop_Sub8", "Iop_Sub16", "Iop_Sub32", "Iop_Sub64"]:
            return "Sub"

        if op in ["Iop_Mul8", "Iop_Mul16", "Iop_Mul32", "Iop_Mul64"]:
            return "Mul"

        if op in ["Iop_Mod8", "Iop_Mod16", "Iop_Mod32", "Iop_Mod64"]:
            return "Mod"

        if op in ["Iop_Or8", "Iop_Or16", "Iop_Or32", "Iop_Or64"]:
            return "Or"

        if op in ["Iop_And8", "Iop_And16", "Iop_And32", "Iop_And64"]:
            return "And"

        if op in ["Iop_Xor8", "Iop_Xor16", "Iop_Xor32", "Iop_Xor64"]:
            return "Xor"

        if op in ["Iop_Shl8", "Iop_Shl16", "Iop_Shl32", "Iop_Shl64"]:
            return "Shl"

        if op in ["Iop_Shr8", "Iop_Shr16", "Iop_Shr32", "Iop_Shr64"]:
            return "Shr"

        if op in ["Iop_Sar8", "Iop_Sar16", "Iop_Sar32", "Iop_Sar64"]:
            return "Sar"

        if op in ["Iop_CmpEQ8", "Iop_CmpEQ16", "Iop_CmpEQ32", "Iop_CmpEQ64"]:
            return "CmpEQ"

        if op in ["Iop_CmpNE8", "Iop_CmpNE16", "Iop_CmpNE32", "Iop_CmpNE64"]:
            return "CmpNE"

        if op in [
            "Iop_CasCmpEQ8",
            "Iop_CasCmpEQ16",
            "Iop_CasCmpEQ32",
            "Iop_CasCmpEQ64",
        ]:
            return "CasCmpEQ"

        if op in [
            "Iop_CasCmpNE8",
            "Iop_CasCmpNE16",
            "Iop_CasCmpNE32",
            "Iop_CasCmpNE64",
        ]:
            return "CasCmpNE"

        if op in [
            "Iop_ExpCmpNE8",
            "Iop_ExpCmpNE16",
            "Iop_ExpCmpNE32",
            "Iop_ExpCmpNE64",
        ]:
            return "ExpCmpNE"

        if op in ["Iop_Not8", "Iop_Not16", "Iop_Not32", "Iop_Not64"]:
            return "Not"

        if op == "Iop_Not1":
            return "Not"
        if op == "Iop_And1":
            return "And"
        if op == "Iop_Or1":
            return "Or"

        if op == "Iop_8Uto16":
            return "Ext"
        if op == "Iop_8Uto32":
            return "Ext"
        if op == "Iop_16Uto32":
            return "Ext"
        if op == "Iop_8Sto16":
            return "Ext"
        if op == "Iop_8Sto32":
            return "Ext"
        if op == "Iop_16Sto32":
            return "Ext"
        if op == "Iop_32Sto64":
            return "Ext"
        if op == "Iop_32Uto64":
            return "Ext"
        if op == "Iop_32to8":
            return "Trunc"
        if op == "Iop_16Uto64":
            return "Ext"
        if op == "Iop_16Sto64":
            return "Ext"
        if op == "Iop_8Uto64":
            return "Ext"
        if op == "Iop_8Sto64":
            return "Ext"
        if op == "Iop_64to16":
            return "Trunc"
        if op == "Iop_64to8":
            return "Trunc"

        if op == "Iop_32to1":
            return "Trunc"
        if op == "Iop_64to1":
            return "Trunc"
        if op == "Iop_1Uto8":
            return "Ext"
        if op == "Iop_1Uto32":
            return "Ext"
        if op == "Iop_1Uto64":
            return "Ext"
        if op == "Iop_1Sto8":
            return "Ext"
        if op == "Iop_1Sto16":
            return "Ext"
        if op == "Iop_1Sto32":
            return "Ext"
        if op == "Iop_1Sto64":
            return "Ext"

        if op in [
            "Iop_MullS8",
            "Iop_MullS16",
            "Iop_MullS32",
            "Iop_MullS64",
            "Iop_MullU8",
            "Iop_MullU16",
            "Iop_MullU32",
            "Iop_MullU64",
        ]:
            return "Mull"

        if op in ["Iop_Clz64", "Iop_Clz32"]:
            return "Clz"
        if op in ["Iop_Ctz64", "Iop_Ctz32"]:
            return "Ctz"

        if op in ["Iop_ClzNat64", "Iop_ClzNat32"]:
            return "ClzNat"
        if op in ["Iop_CtzNat64", "Iop_CtzNat32"]:
            return "CtzNat"

        if op in ["Iop_PopCount64", "Iop_PopCount32"]:
            return "PopCount"

        if op == "Iop_CmpLT32S":
            return "CmpLT"
        if op == "Iop_CmpLE32S":
            return "CmpLE"
        if op == "Iop_CmpLT32U":
            return "CmpLT"
        if op == "Iop_CmpLE32U":
            return "CmpLE"

        if op == "Iop_CmpLT64S":
            return "CmpLT"
        if op == "Iop_CmpLE64S":
            return "CmpLE"
        if op == "Iop_CmpLT64U":
            return "CmpLT"
        if op == "Iop_CmpLE64U":
            return "CmpLE"
        if op in ["Iop_CmpNEZ8", "Iop_CmpNEZ16", "Iop_CmpNEZ32", "Iop_CmpNEZ64"]:
            return "CmpNEZ"
        if op in ["Iop_CmpwNEZ32", "Iop_CmpwNEZ64"]:
            return "CmpwNEZ"
        if op in ["Iop_Left8", "Iop_Left16", "Iop_Left32", "Iop_Left64"]:
            return "Left"
        if op == "Iop_Max32U":
            return "Max"

        if op == "Iop_CmpORD32U":
            return "CmpORD"
        if op == "Iop_CmpORD32S":
            return "CmpORD"

        if op == "Iop_CmpORD64U":
            return "CmpORD"
        if op == "Iop_CmpORD64S":
            return "CmpORD"

        if op == "Iop_DivU8":
            return "Div"
        if op == "Iop_DivU16":
            return "Div"
        if op == "Iop_DivU32":
            return "Div"
        if op == "Iop_DivU64":
            return "Div"
        if op == "Iop_DivS8":
            return "Div"
        if op == "Iop_DivS16":
            return "Div"
        if op == "Iop_DivS32":
            return "Div"
        if op == "Iop_DivS64":
            return "Div"
        if op == "Iop_DivU64E":
            return "Div"
        if op == "Iop_DivS64E":
            return "Div"
        if op == "Iop_DivU32E":
            return "Div"
        if op == "Iop_DivS32E":
            return "Div"
        if op == "Iop_DivU16E":
            return "Div"
        if op == "Iop_DivS16E":
            return "Div"
        if op == "Iop_DivU8E":
            return "Div"
        if op == "Iop_DivS8E":
            return "Div"

        if op == "Iop_DivModU64to32":
            return "DivMod"
        if op == "Iop_DivModS64to32":
            return "DivMod"

        if op == "Iop_DivModU32to32":
            return "DivMod"
        if op == "Iop_DivModS32to32":
            return "DivMod"

        if op == "Iop_DivModU128to64":
            return "DivMod"
        if op == "Iop_DivModS128to64":
            return "DivMod"

        if op == "Iop_DivModS64to64":
            return "DivMod"
        if op == "Iop_DivModU64to64":
            return "DivMod"

        if op == "Iop_16HIto8":
            return "HTrunc"
        if op == "Iop_16to8":
            return "Trunc"
        if op == "Iop_8HLto16":
            return "HLExt"

        if op == "Iop_32HIto16":
            return "HTrunc"
        if op == "Iop_32to16":
            return "Trunc"
        if op == "Iop_16HLto32":
            return "HLExt"

        if op == "Iop_64HIto32":
            return "HTrunc"
        if op == "Iop_64to32":
            return "Trunc"
        if op == "Iop_32HLto64":
            return "HLExt"

        if op == "Iop_128HIto64":
            return "HTrunc"
        if op == "Iop_128to64":
            return "Trunc"
        if op == "Iop_64HLto128":
            return "HLExt"

        if op == "Iop_CmpF32":
            return "CmpF"
        if op == "Iop_F32toI32S":
            return "ConvFI"
        if op == "Iop_F32toI64S":
            return "ConvFI"
        if op == "Iop_I32StoF32":
            return "ConvIF"
        if op == "Iop_I64StoF32":
            return "ConvIF"

        if op == "Iop_AddF64":
            return "AddF"
        if op == "Iop_SubF64":
            return "SubF"
        if op == "Iop_MulF64":
            return "MulF"
        if op == "Iop_DivF64":
            return "DivF"
        if op == "Iop_AddF64r32":
            return "AddF"
        if op == "Iop_SubF64r32":
            return "SubF"
        if op == "Iop_MulF64r32":
            return "MulF"
        if op == "Iop_DivF64r32":
            return "DivF"
        if op == "Iop_AddF32":
            return "AddF"
        if op == "Iop_SubF32":
            return "SubF"
        if op == "Iop_MulF32":
            return "MulF"
        if op == "Iop_DivF32":
            return "DivF"

        if op == "Iop_AddF128":
            return "AddF"
        if op == "Iop_SubF128":
            return "SubF"
        if op == "Iop_MulF128":
            return "MulF"
        if op == "Iop_DivF128":
            return "DivF"

        if op == "Iop_TruncF128toI64S":
            return "TruncFI"
        if op == "Iop_TruncF128toI32S":
            return "TruncFI"
        if op == "Iop_TruncF128toI64U":
            return "TruncFI"
        if op == "Iop_TruncF128toI32U":
            return "TruncFI"

        if op == "Iop_MAddF128":
            return "MAddF"
        if op == "Iop_MSubF128":
            return "MSubF"
        if op == "Iop_NegMAddF128":
            return "NegMAddF"
        if op == "Iop_NegMSubF128":
            return "NegMSubF"

        if op == "Iop_AbsF128":
            return "AbsF"
        if op == "Iop_NegF128":
            return "NegF"
        if op == "Iop_SqrtF128":
            return "SqrtF"
        if op == "Iop_CmpF128":
            return "CmpF"

        if op == "Iop_F64HLtoF128":
            return "HLExtF"
        if op == "Iop_F128HItoF64":
            return "HTruncF"
        if op == "Iop_F128LOtoF64":
            return "LTruncF"
        if op == "Iop_I32StoF128":
            return "ExtIF"
        if op == "Iop_I64StoF128":
            return "ExtIF"
        if op == "Iop_I32UtoF128":
            return "ExtIF"
        if op == "Iop_I64UtoF128":
            return "ExtIF"
        if op == "Iop_F128toI32S":
            return "TruncFI"
        if op == "Iop_F128toI64S":
            return "TruncFI"
        if op == "Iop_F128toI32U":
            return "TruncFI"
        if op == "Iop_F128toI64U":
            return "TruncFI"
        if op == "Iop_F32toF128":
            return "ExtF"
        if op == "Iop_F64toF128":
            return "ExtF"
        if op == "Iop_F128toF64":
            return "TruncF"
        if op == "Iop_F128toF32":
            return "TruncF"
        if op == "Iop_F128toI128S":
            return "ConvFI"
        if op == "Iop_RndF128":
            return "RndF"

        if op == "Iop_MAddF32":
            return "MAddF"
        if op == "Iop_MSubF32":
            return "MSubF"

        if op == "Iop_ScaleF64":
            return "ScaleF"
        if op == "Iop_AtanF64":
            return "AtanF"
        if op == "Iop_Yl2xF64":
            return "Yl2xF"
        if op == "Iop_Yl2xp1F64":
            return "Yl2xp1F"
        if op == "Iop_PRemF64":
            return "PRemF"
        if op == "Iop_PRemC3210F64":
            return "PRemC3210F"
        if op == "Iop_PRem1F64":
            return "PRem1F"
        if op == "Iop_PRem1C3210F64":
            return "PRem1C3210F"
        if op == "Iop_NegF64":
            return "NegF"
        if op == "Iop_AbsF64":
            return "AbsF"
        if op == "Iop_NegF32":
            return "NegF"
        if op == "Iop_AbsF32":
            return "AbsF"
        if op in ["Iop_SqrtF64", "Iop_SqrtF32"]:
            return "SqrtF"
        if op == "Iop_SinF64":
            return "SinF"
        if op == "Iop_CosF64":
            return "CosF"
        if op == "Iop_TanF64":
            return "TanF"
        if op == "Iop_2xm1F64":
            return "2xm1F"

        if op == "Iop_MAddF64":
            return "MAddF"
        if op == "Iop_MSubF64":
            return "MSubF"
        if op == "Iop_MAddF64r32":
            return "MAddF"
        if op == "Iop_MSubF64r32":
            return "MSubF"

        if op == "Iop_RSqrtEst5GoodF64":
            return "RSqrtEst5GoodF"
        if op == "Iop_RoundF64toF64_NEAREST":
            return "RndF"
        if op == "Iop_RoundF64toF64_NegINF":
            return "RndF"
        if op == "Iop_RoundF64toF64_PosINF":
            return "RndF"
        if op == "Iop_RoundF64toF64_ZERO":
            return "RndF"

        if op == "Iop_TruncF64asF32":
            return "TruncF"

        if op == "Iop_RecpExpF64":
            return "RecpExpF"
        if op == "Iop_RecpExpF32":
            return "RecpExpF"

        if op == "Iop_MaxNumF64":
            return "MaxNumF"
        if op == "Iop_MinNumF64":
            return "MinNumF"
        if op == "Iop_MaxNumF32":
            return "MaxNumF"
        if op == "Iop_MinNumF32":
            return "MinNumF"

        if op == "Iop_F16toF64":
            return "ExtF"
        if op == "Iop_F64toF16":
            return "TruncF"
        if op == "Iop_F16toF32":
            return "ExtF"
        if op == "Iop_F32toF16":
            return "TruncF"

        if op == "Iop_QAdd32S":
            return "QAddV"
        if op == "Iop_QSub32S":
            return "QSubV"
        if op == "Iop_Add16x2":
            return "AddV"
        if op == "Iop_Sub16x2":
            return "SubV"
        if op == "Iop_QAdd16Sx2":
            return "QAddV"
        if op == "Iop_QAdd16Ux2":
            return "QAddV"
        if op == "Iop_QSub16Sx2":
            return "QSubV"
        if op == "Iop_QSub16Ux2":
            return "QSubV"
        if op == "Iop_HAdd16Ux2":
            return "HAddV"
        if op == "Iop_HAdd16Sx2":
            return "HAddV"
        if op == "Iop_HSub16Ux2":
            return "HSubV"
        if op == "Iop_HSub16Sx2":
            return "HSubV"

        if op == "Iop_Add8x4":
            return "AddV"
        if op == "Iop_Sub8x4":
            return "SubV"
        if op == "Iop_QAdd8Sx4":
            return "QAddV"
        if op == "Iop_QAdd8Ux4":
            return "QAddV"
        if op == "Iop_QSub8Sx4":
            return "QSubV"
        if op == "Iop_QSub8Ux4":
            return "QSubV"
        if op == "Iop_HAdd8Ux4":
            return "HAddV"
        if op == "Iop_HAdd8Sx4":
            return "HAddV"
        if op == "Iop_HSub8Ux4":
            return "HSubV"
        if op == "Iop_HSub8Sx4":
            return "HSubV"
        if op == "Iop_Sad8Ux4":
            return "SadV"

        if op == "Iop_CmpNEZ16x2":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ8x4":
            return "CmpNEZV"
        if op == "Iop_Reverse8sIn32_x1":
            return "ReverseChunks"

        if op == "Iop_CmpF64":
            return "CmpF"

        if op == "Iop_F64toI16S":
            return "TruncFI"
        if op == "Iop_F64toI32S":
            return "TruncFI"
        if op == "Iop_F64toI64S":
            return "ConvFI"
        if op == "Iop_F64toI64U":
            return "ConvFI"
        if op == "Iop_F32toI32U":
            return "ConvFI"
        if op == "Iop_F32toI64U":
            return "ExtFI"

        if op == "Iop_F64toI32U":
            return "TruncFI"

        if op == "Iop_I32StoF64":
            return "ExtIF"
        if op == "Iop_I64StoF64":
            return "ConvIF"
        if op == "Iop_I64UtoF64":
            return "ConvIF"
        if op == "Iop_I32UtoF32":
            return "ConvIF"
        if op == "Iop_I64UtoF32":
            return "TruncIF"

        if op == "Iop_I32UtoF64":
            return "ExtIF"

        if op == "Iop_F32toF64":
            return "ExtF"
        if op == "Iop_F64toF32":
            return "TruncF"

        if op == "Iop_RoundF128toInt":
            return "RndFI"
        if op == "Iop_RoundF64toInt":
            return "RndFI"
        if op == "Iop_RoundF32toInt":
            return "RndFI"
        if op == "Iop_RoundF64toF32":
            return "RndTruncF"

        if op == "Iop_ReinterpF64asI64":
            return "ReinterpFI"
        if op == "Iop_ReinterpI64asF64":
            return "ReinterpIF"
        if op == "Iop_ReinterpF32asI32":
            return "ReinterpFI"
        if op == "Iop_ReinterpI32asF32":
            return "ReinterpIF"

        if op == "Iop_I32UtoF32x4_DEP":
            return "ConvIFV_DEP"
        if op == "Iop_I32StoF32x4_DEP":
            return "ConvIFV_DEP"

        if op == "Iop_I32StoF32x4":
            return "ConvIF"
        if op == "Iop_F32toI32Sx4":
            return "ConvFI"

        if op == "Iop_F32toF16x4_DEP":
            return "TruncFV_DEP"
        if op == "Iop_F32toF16x4":
            return "TruncFV"
        if op == "Iop_F16toF32x4":
            return "ExtFV"
        if op == "Iop_F16toF64x2":
            return "ExtFV"
        if op == "Iop_F64toF16x2_DEP":
            return "TruncFV_DEP"

        if op == "Iop_RSqrtEst32Fx4":
            return "RSqrtEstFV"
        if op == "Iop_RSqrtEst32Ux4":
            return "RSqrtEstV"
        if op == "Iop_RSqrtEst32Fx2":
            return "RSqrtEstFV"
        if op == "Iop_RSqrtEst32Ux2":
            return "RSqrtEstV"

        if op == "Iop_QF32toI32Ux4_RZ":
            return "QConvFIV"
        if op == "Iop_QF32toI32Sx4_RZ":
            return "QConvFIV"

        if op == "Iop_F32toI32Ux4_RZ":
            return "ConvFIV"
        if op == "Iop_F32toI32Sx4_RZ":
            return "ConvFIV"

        if op == "Iop_I32UtoF32x2_DEP":
            return "ConvIFV_DEP"
        if op == "Iop_I32StoF32x2_DEP":
            return "ConvIFV_DEP"

        if op == "Iop_F32toI32Ux2_RZ":
            return "ConvFIV"
        if op == "Iop_F32toI32Sx2_RZ":
            return "ConvFIV"

        if op == "Iop_RoundF32x4_RM":
            return "RndFV"
        if op == "Iop_RoundF32x4_RP":
            return "RndFV"
        if op == "Iop_RoundF32x4_RN":
            return "RndFV"
        if op == "Iop_RoundF32x4_RZ":
            return "RndFV"

        if op == "Iop_Abs8x8":
            return "AbsV"
        if op == "Iop_Abs16x4":
            return "AbsV"
        if op == "Iop_Abs32x2":
            return "AbsV"
        if op == "Iop_Add8x8":
            return "AddV"
        if op == "Iop_Add16x4":
            return "AddV"
        if op == "Iop_Add32x2":
            return "AddV"
        if op == "Iop_QAdd8Ux8":
            return "QAddV"
        if op == "Iop_QAdd16Ux4":
            return "QAddV"
        if op == "Iop_QAdd32Ux2":
            return "QAddV"
        if op == "Iop_QAdd64Ux1":
            return "QAddV"
        if op == "Iop_QAdd8Sx8":
            return "QAddV"
        if op == "Iop_QAdd16Sx4":
            return "QAddV"
        if op == "Iop_QAdd32Sx2":
            return "QAddV"
        if op == "Iop_QAdd64Sx1":
            return "QAddV"
        if op == "Iop_PwAdd8x8":
            return "PwAddV"
        if op == "Iop_PwAdd16x4":
            return "PwAddV"
        if op == "Iop_PwAdd32x2":
            return "PwAddV"
        if op == "Iop_PwAdd32Fx2":
            return "PwAddV"
        if op == "Iop_PwAddL8Ux8":
            return "PwAddLV"
        if op == "Iop_PwAddL16Ux4":
            return "PwAddLV"
        if op == "Iop_PwAddL32Ux2":
            return "PwAddLV"
        if op == "Iop_PwAddL8Sx8":
            return "PwAddLV"
        if op == "Iop_PwAddL16Sx4":
            return "PwAddLV"
        if op == "Iop_PwAddL32Sx2":
            return "PwAddLV"
        if op == "Iop_Sub8x8":
            return "SubV"
        if op == "Iop_Sub16x4":
            return "SubV"
        if op == "Iop_Sub32x2":
            return "SubV"
        if op == "Iop_QSub8Ux8":
            return "QSubV"
        if op == "Iop_QSub16Ux4":
            return "QSubV"
        if op == "Iop_QSub32Ux2":
            return "QSubV"
        if op == "Iop_QSub64Ux1":
            return "QSubV"
        if op == "Iop_QSub8Sx8":
            return "QSubV"
        if op == "Iop_QSub16Sx4":
            return "QSubV"
        if op == "Iop_QSub32Sx2":
            return "QSubV"
        if op == "Iop_QSub64Sx1":
            return "QSubV"
        if op == "Iop_Mul8x8":
            return "MulV"
        if op == "Iop_Mul16x4":
            return "MulV"
        if op == "Iop_Mul32x2":
            return "MulV"
        if op == "Iop_Mul32Fx2":
            return "MulV"
        if op == "Iop_PolynomialMul8x8":
            return "PolynomialMulV"
        if op == "Iop_MulHi16Ux4":
            return "MulHiV"
        if op == "Iop_MulHi16Sx4":
            return "MulHiV"
        if op == "Iop_QDMulHi16Sx4":
            return "QDMulHiV"
        if op == "Iop_QDMulHi32Sx2":
            return "QDMulHiV"
        if op == "Iop_QRDMulHi16Sx4":
            return "QRDMulHiV"
        if op == "Iop_QRDMulHi32Sx2":
            return "QRDMulHiV"
        if op == "Iop_QDMull16Sx4":
            return "QDMullV"
        if op == "Iop_QDMull32Sx2":
            return "QDMullV"
        if op == "Iop_Avg8Ux8":
            return "AvgV"
        if op == "Iop_Avg16Ux4":
            return "AvgV"
        if op == "Iop_Max8Sx8":
            return "MaxV"
        if op == "Iop_Max16Sx4":
            return "MaxV"
        if op == "Iop_Max32Sx2":
            return "MaxV"
        if op == "Iop_Max8Ux8":
            return "MaxV"
        if op == "Iop_Max16Ux4":
            return "MaxV"
        if op == "Iop_Max32Ux2":
            return "MaxV"
        if op == "Iop_Min8Sx8":
            return "MinV"
        if op == "Iop_Min16Sx4":
            return "MinV"
        if op == "Iop_Min32Sx2":
            return "MinV"
        if op == "Iop_Min8Ux8":
            return "MinV"
        if op == "Iop_Min16Ux4":
            return "MinV"
        if op == "Iop_Min32Ux2":
            return "MinV"
        if op == "Iop_PwMax8Sx8":
            return "PwMaxV"
        if op == "Iop_PwMax16Sx4":
            return "PwMaxV"
        if op == "Iop_PwMax32Sx2":
            return "PwMaxV"
        if op == "Iop_PwMax8Ux8":
            return "PwMaxV"
        if op == "Iop_PwMax16Ux4":
            return "PwMaxV"
        if op == "Iop_PwMax32Ux2":
            return "PwMaxV"
        if op == "Iop_PwMin8Sx8":
            return "PwMinV"
        if op == "Iop_PwMin16Sx4":
            return "PwMinV"
        if op == "Iop_PwMin32Sx2":
            return "PwMinV"
        if op == "Iop_PwMin8Ux8":
            return "PwMinV"
        if op == "Iop_PwMin16Ux4":
            return "PwMinV"
        if op == "Iop_PwMin32Ux2":
            return "PwMinV"
        if op == "Iop_CmpEQ8x8":
            return "CmpEQV"
        if op == "Iop_CmpEQ16x4":
            return "CmpEQV"
        if op == "Iop_CmpEQ32x2":
            return "CmpEQV"
        if op == "Iop_CmpGT8Ux8":
            return "CmpGTV"
        if op == "Iop_CmpGT16Ux4":
            return "CmpGTV"
        if op == "Iop_CmpGT32Ux2":
            return "CmpGTV"
        if op == "Iop_CmpGT8Sx8":
            return "CmpGTV"
        if op == "Iop_CmpGT16Sx4":
            return "CmpGTV"
        if op == "Iop_CmpGT32Sx2":
            return "CmpGTV"
        if op == "Iop_Cnt8x8":
            return "CntV"
        if op == "Iop_Clz8x8":
            return "ClzV"
        if op == "Iop_Clz16x4":
            return "ClzV"
        if op == "Iop_Clz32x2":
            return "ClzV"
        if op == "Iop_Cls8x8":
            return "ClsV"
        if op == "Iop_Cls16x4":
            return "ClsV"
        if op == "Iop_Cls32x2":
            return "ClsV"
        if op == "Iop_ShlN8x8":
            return "ShlNV"
        if op == "Iop_ShlN16x4":
            return "ShlNV"
        if op == "Iop_ShlN32x2":
            return "ShlNV"
        if op == "Iop_ShrN8x8":
            return "ShrNV"
        if op == "Iop_ShrN16x4":
            return "ShrNV"
        if op == "Iop_ShrN32x2":
            return "ShrNV"
        if op == "Iop_SarN8x8":
            return "SarNV"
        if op == "Iop_SarN16x4":
            return "SarNV"
        if op == "Iop_SarN32x2":
            return "SarNV"
        if op == "Iop_QNarrowBin16Sto8Ux8":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin16Sto8Sx8":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin32Sto16Sx4":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin64Sto32Sx4":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin64Uto32Ux4":
            return "QNarrowBin"
        if op == "Iop_NarrowBin16to8x8":
            return "NarrowBin"
        if op == "Iop_NarrowBin32to16x4":
            return "NarrowBin"
        if op == "Iop_NarrowBin64to32x4":
            return "NarrowBin"
        if op == "Iop_InterleaveHI8x8":
            return "HInterleaveV"
        if op == "Iop_InterleaveHI16x4":
            return "HInterleaveV"
        if op == "Iop_InterleaveHI32x2":
            return "HInterleaveV"
        if op == "Iop_InterleaveLO8x8":
            return "LInterleaveV"
        if op == "Iop_InterleaveLO16x4":
            return "LInterleaveV"
        if op == "Iop_InterleaveLO32x2":
            return "LInterleaveV"
        if op == "Iop_CatOddLanes8x8":
            return "CatOddLanesV"
        if op == "Iop_CatOddLanes16x4":
            return "CatOddLanesV"
        if op == "Iop_CatEvenLanes8x8":
            return "CatEvenLanesV"
        if op == "Iop_CatEvenLanes16x4":
            return "CatEvenLanesV"
        if op == "Iop_InterleaveOddLanes8x8":
            return "InterleaveOddLanesV"
        if op == "Iop_InterleaveOddLanes16x4":
            return "InterleaveOddLanesV"
        if op == "Iop_InterleaveEvenLanes8x8":
            return "InterleaveEvenLanesV"
        if op == "Iop_InterleaveEvenLanes16x4":
            return "InterleaveEvenLanesV"
        if op == "Iop_Shl8x8":
            return "ShlV"
        if op == "Iop_Shl16x4":
            return "ShlV"
        if op == "Iop_Shl32x2":
            return "ShlV"
        if op == "Iop_Shr8x8":
            return "ShrV"
        if op == "Iop_Shr16x4":
            return "ShrV"
        if op == "Iop_Shr32x2":
            return "ShrV"
        if op == "Iop_QShl8x8":
            return "QShlV"
        if op == "Iop_QShl16x4":
            return "QShlV"
        if op == "Iop_QShl32x2":
            return "QShlV"
        if op == "Iop_QShl64x1":
            return "QShlV"
        if op == "Iop_QSal8x8":
            return "QSalV"
        if op == "Iop_QSal16x4":
            return "QSalV"
        if op == "Iop_QSal32x2":
            return "QSalV"
        if op == "Iop_QSal64x1":
            return "QSalV"
        if op == "Iop_QShlNsatUU8x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU16x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU32x2":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU64x1":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU8x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU16x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU32x2":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU64x1":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS8x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS16x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS32x2":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS64x1":
            return "QShlNsatV"
        if op == "Iop_Sar8x8":
            return "SarV"
        if op == "Iop_Sar16x4":
            return "SarV"
        if op == "Iop_Sar32x2":
            return "SarV"
        if op == "Iop_Sal8x8":
            return "SalV"
        if op == "Iop_Sal16x4":
            return "SalV"
        if op == "Iop_Sal32x2":
            return "SalV"
        if op == "Iop_Sal64x1":
            return "SalV"
        if op == "Iop_Perm8x8":
            return "PermV"
        if op == "Iop_PermOrZero8x8":
            return "PermOrZeroV"
        if op == "Iop_Reverse8sIn16_x4":
            return "ReverseChunks"
        if op == "Iop_Reverse8sIn32_x2":
            return "ReverseChunks"
        if op == "Iop_Reverse16sIn32_x2":
            return "ReverseChunks"
        if op == "Iop_Reverse8sIn64_x1":
            return "ReverseChunks"
        if op == "Iop_Reverse16sIn64_x1":
            return "ReverseChunks"
        if op == "Iop_Reverse32sIn64_x1":
            return "ReverseChunks"
        if op == "Iop_Abs32Fx2":
            return "AbsFV"
        if op == "Iop_GetMSBs8x8":
            return "GetMSBsV"
        if op == "Iop_GetMSBs8x16":
            return "GetMSBsV"

        if op == "Iop_CmpNEZ32x2":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ16x4":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ8x8":
            return "CmpNEZV"

        if op == "Iop_Add32Fx4":
            return "AddFV"
        if op == "Iop_Add32Fx2":
            return "AddFV"
        if op == "Iop_Add32F0x4":
            return "AddFV"
        if op == "Iop_Add64Fx2":
            return "AddFV"
        if op == "Iop_Add64F0x2":
            return "AddFV"

        if op == "Iop_Div32Fx4":
            return "DivFV"
        if op == "Iop_Div32F0x4":
            return "DivFV"
        if op == "Iop_Div64Fx2":
            return "DivFV"
        if op == "Iop_Div64F0x2":
            return "DivFV"

        if op == "Iop_Max32Fx8":
            return "MaxFV"
        if op == "Iop_Max32Fx4":
            return "MaxFV"
        if op == "Iop_Max32Fx2":
            return "MaxFV"
        if op == "Iop_PwMax32Fx4":
            return "PwMaxFV"
        if op == "Iop_PwMax32Fx2":
            return "PwMaxFV"
        if op == "Iop_Max32F0x4":
            return "MaxFV"
        if op == "Iop_Max64Fx4":
            return "MaxFV"
        if op == "Iop_Max64Fx2":
            return "MaxFV"
        if op == "Iop_Max64F0x2":
            return "MaxFV"

        if op == "Iop_Min32Fx8":
            return "MinFV"
        if op == "Iop_Min32Fx4":
            return "MinFV"
        if op == "Iop_Min32Fx2":
            return "MinFV"
        if op == "Iop_PwMin32Fx4":
            return "PwMinFV"
        if op == "Iop_PwMin32Fx2":
            return "PwMinFV"
        if op == "Iop_Min32F0x4":
            return "MinFV"
        if op == "Iop_Min64Fx4":
            return "MinFV"
        if op == "Iop_Min64Fx2":
            return "MinFV"
        if op == "Iop_Min64F0x2":
            return "MinFV"

        if op == "Iop_Mul32Fx4":
            return "MulFV"
        if op == "Iop_Mul32F0x4":
            return "MulFV"
        if op == "Iop_Mul64Fx2":
            return "MulFV"
        if op == "Iop_Mul64F0x2":
            return "MulFV"

        if op == "Iop_RecipEst32Ux2":
            return "RecipEstV"
        if op == "Iop_RecipEst32Fx2":
            return "RecipEstFV"
        if op == "Iop_RecipEst32Fx4":
            return "RecipEstFV"
        if op == "Iop_RecipEst32Fx8":
            return "RecipEstFV"
        if op == "Iop_RecipEst32Ux4":
            return "RecipEstV"
        if op == "Iop_RecipEst32F0x4":
            return "RecipEstFV"
        if op == "Iop_RecipStep32Fx2":
            return "RecipStepFV"
        if op == "Iop_RecipStep32Fx4":
            return "RecipStepFV"
        if op == "Iop_RecipEst64Fx2":
            return "RecipEstFV"
        if op == "Iop_RecipStep64Fx2":
            return "RecipStepFV"

        if op == "Iop_Abs32Fx4":
            return "AbsFV"
        if op == "Iop_Abs64Fx2":
            return "AbsFV"
        if op == "Iop_RSqrtStep32Fx4":
            return "RSqrtStepFV"
        if op == "Iop_RSqrtStep64Fx2":
            return "RSqrtStepFV"
        if op == "Iop_RSqrtStep32Fx2":
            return "RSqrtStepFV"
        if op == "Iop_RSqrtEst64Fx2":
            return "RSqrtEstFV"

        if op == "Iop_RSqrtEst32F0x4":
            return "RSqrtEstFV"
        if op == "Iop_RSqrtEst32Fx8":
            return "RSqrtEstFV"

        if op == "Iop_Sqrt32Fx4":
            return "SqrtFV"
        if op == "Iop_Sqrt32F0x4":
            return "SqrtFV"
        if op == "Iop_Sqrt64Fx2":
            return "SqrtFV"
        if op == "Iop_Sqrt64F0x2":
            return "SqrtFV"
        if op == "Iop_Sqrt32Fx8":
            return "SqrtFV"
        if op == "Iop_Sqrt64Fx4":
            return "SqrtFV"

        if op == "Iop_Scale2_32Fx4":
            return "Scale2_FV"
        if op == "Iop_Scale2_64Fx2":
            return "Scale2_FV"
        if op == "Iop_Log2_32Fx4":
            return "Log2_FV"
        if op == "Iop_Log2_64Fx2":
            return "Log2_FV"
        if op == "Iop_Exp2_32Fx4":
            return "Exp2_FV"

        if op == "Iop_Sub32Fx4":
            return "SubFV"
        if op == "Iop_Sub32Fx2":
            return "SubFV"
        if op == "Iop_Sub32F0x4":
            return "SubFV"
        if op == "Iop_Sub64Fx2":
            return "SubFV"
        if op == "Iop_Sub64F0x2":
            return "SubFV"

        if op == "Iop_CmpEQ32Fx4":
            return "CmpEQFV"
        if op == "Iop_CmpLT32Fx4":
            return "CmpLTFV"
        if op == "Iop_CmpLE32Fx4":
            return "CmpLEFV"
        if op == "Iop_CmpGT32Fx4":
            return "CmpGTFV"
        if op == "Iop_CmpGE32Fx4":
            return "CmpGEFV"
        if op == "Iop_CmpUN32Fx4":
            return "CmpUNFV"
        if op == "Iop_CmpEQ64Fx2":
            return "CmpEQFV"
        if op == "Iop_CmpLT64Fx2":
            return "CmpLTFV"
        if op == "Iop_CmpLE64Fx2":
            return "CmpLEFV"
        if op == "Iop_CmpUN64Fx2":
            return "CmpUNFV"
        if op == "Iop_CmpGT32Fx2":
            return "CmpGTFV"
        if op == "Iop_CmpEQ32Fx2":
            return "CmpEQFV"
        if op == "Iop_CmpGE32Fx2":
            return "CmpGEFV"

        if op == "Iop_CmpEQ32F0x4":
            return "CmpEQFV"
        if op == "Iop_CmpLT32F0x4":
            return "CmpLTFV"
        if op == "Iop_CmpLE32F0x4":
            return "CmpLEFV"
        if op == "Iop_CmpUN32F0x4":
            return "CmpUNFV"
        if op == "Iop_CmpEQ64F0x2":
            return "CmpEQFV"
        if op == "Iop_CmpLT64F0x2":
            return "CmpLTFV"
        if op == "Iop_CmpLE64F0x2":
            return "CmpLEFV"
        if op == "Iop_CmpUN64F0x2":
            return "CmpUNFV"

        if op == "Iop_Neg64Fx2":
            return "NegFV"
        if op == "Iop_Neg32Fx4":
            return "NegFV"
        if op == "Iop_Neg32Fx2":
            return "NegFV"

        if op == "Iop_F32x4_2toQ16x8":
            return "ConvFQ"
        if op == "Iop_F64x2_2toQ32x4":
            return "ConvFQ"

        if op == "Iop_V128to64":
            return "TruncV"
        if op == "Iop_V128HIto64":
            return "HTruncV"
        if op == "Iop_64HLtoV128":
            return "HLExtV"

        if op == "Iop_64UtoV128":
            return "ExtV"
        if op == "Iop_SetV128lo64":
            return "SetVlo"

        if op == "Iop_ZeroHI64ofV128":
            return "ZeroHV"
        if op == "Iop_ZeroHI96ofV128":
            return "ZeroHV"
        if op == "Iop_ZeroHI112ofV128":
            return "ZeroHV"
        if op == "Iop_ZeroHI120ofV128":
            return "ZeroHV"

        if op == "Iop_32UtoV128":
            return "ExtV"
        if op == "Iop_V128to32":
            return "TruncV"
        if op == "Iop_SetV128lo32":
            return "SetVlo"

        if op == "Iop_Dup8x16":
            return "DupV"
        if op == "Iop_Dup16x8":
            return "DupV"
        if op == "Iop_Dup32x4":
            return "DupV"
        if op == "Iop_Dup8x8":
            return "DupV"
        if op == "Iop_Dup16x4":
            return "DupV"
        if op == "Iop_Dup32x2":
            return "DupV"

        if op == "Iop_NotV128":
            return "NotV"
        if op == "Iop_AndV128":
            return "AndV"
        if op == "Iop_OrV128":
            return "OrV"
        if op == "Iop_XorV128":
            return "XorV"

        if op == "Iop_CmpNEZ8x16":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ16x8":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ32x4":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ64x2":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ128x1":
            return "CmpNEZV"

        if op == "Iop_Abs8x16":
            return "AbsV"
        if op == "Iop_Abs16x8":
            return "AbsV"
        if op == "Iop_Abs32x4":
            return "AbsV"
        if op == "Iop_Abs64x2":
            return "AbsV"

        if op == "Iop_Add8x16":
            return "AddV"
        if op == "Iop_Add16x8":
            return "AddV"
        if op == "Iop_Add32x4":
            return "AddV"
        if op == "Iop_Add64x2":
            return "AddV"
        if op == "Iop_Add128x1":
            return "AddV"
        if op == "Iop_QAdd8Ux16":
            return "QAddV"
        if op == "Iop_QAdd16Ux8":
            return "QAddV"
        if op == "Iop_QAdd32Ux4":
            return "QAddV"
        if op == "Iop_QAdd8Sx16":
            return "QAddV"
        if op == "Iop_QAdd16Sx8":
            return "QAddV"
        if op == "Iop_QAdd32Sx4":
            return "QAddV"
        if op == "Iop_QAdd64Ux2":
            return "QAddV"
        if op == "Iop_QAdd64Sx2":
            return "QAddV"

        if op == "Iop_QAddExtUSsatSS8x16":
            return "QAddExtsat"
        if op == "Iop_QAddExtUSsatSS16x8":
            return "QAddExtsat"
        if op == "Iop_QAddExtUSsatSS32x4":
            return "QAddExtsat"
        if op == "Iop_QAddExtUSsatSS64x2":
            return "QAddExtsat"
        if op == "Iop_QAddExtSUsatUU8x16":
            return "QAddExtsat"
        if op == "Iop_QAddExtSUsatUU16x8":
            return "QAddExtsat"
        if op == "Iop_QAddExtSUsatUU32x4":
            return "QAddExtsat"
        if op == "Iop_QAddExtSUsatUU64x2":
            return "QAddExtsat"

        if op == "Iop_PwAdd8x16":
            return "PwAddV"
        if op == "Iop_PwAdd16x8":
            return "PwAddV"
        if op == "Iop_PwAdd32x4":
            return "PwAddV"
        if op == "Iop_PwAddL8Ux16":
            return "PwAddLV"
        if op == "Iop_PwAddL16Ux8":
            return "PwAddLV"
        if op == "Iop_PwAddL32Ux4":
            return "PwAddLV"
        if op == "Iop_PwAddL64Ux2":
            return "PwAddLV"
        if op == "Iop_PwAddL8Sx16":
            return "PwAddLV"
        if op == "Iop_PwAddL16Sx8":
            return "PwAddLV"
        if op == "Iop_PwAddL32Sx4":
            return "PwAddLV"
        if op == "Iop_PwExtUSMulQAdd8x16":
            return "PwExtUSMulQAdd"

        if op == "Iop_Sub8x16":
            return "SubV"
        if op == "Iop_Sub16x8":
            return "SubV"
        if op == "Iop_Sub32x4":
            return "SubV"
        if op == "Iop_Sub64x2":
            return "SubV"
        if op == "Iop_Sub128x1":
            return "SubV"
        if op == "Iop_QSub8Ux16":
            return "QSubV"
        if op == "Iop_QSub16Ux8":
            return "QSubV"
        if op == "Iop_QSub32Ux4":
            return "QSubV"
        if op == "Iop_QSub8Sx16":
            return "QSubV"
        if op == "Iop_QSub16Sx8":
            return "QSubV"
        if op == "Iop_QSub32Sx4":
            return "QSubV"
        if op == "Iop_QSub64Ux2":
            return "QSubV"
        if op == "Iop_QSub64Sx2":
            return "QSubV"

        if op == "Iop_Mul8x16":
            return "MulV"
        if op == "Iop_Mul16x8":
            return "MulV"
        if op == "Iop_Mul32x4":
            return "MulV"
        if op == "Iop_Mull8Ux8":
            return "MullV"
        if op == "Iop_Mull8Sx8":
            return "MullV"
        if op == "Iop_Mull16Ux4":
            return "MullV"
        if op == "Iop_Mull16Sx4":
            return "MullV"
        if op == "Iop_Mull32Ux2":
            return "MullV"
        if op == "Iop_Mull32Sx2":
            return "MullV"
        if op == "Iop_PolynomialMul8x16":
            return "PolynomialMulV"
        if op == "Iop_PolynomialMull8x8":
            return "PolynomialMullV"
        if op == "Iop_MulHi8Ux16":
            return "MulHiV"
        if op == "Iop_MulHi16Ux8":
            return "MulHiV"
        if op == "Iop_MulHi32Ux4":
            return "MulHiV"
        if op == "Iop_MulHi8Sx16":
            return "MulHiV"
        if op == "Iop_MulHi16Sx8":
            return "MulHiV"
        if op == "Iop_MulHi32Sx4":
            return "MulHiV"
        if op == "Iop_QDMulHi16Sx8":
            return "QDMulHiV"
        if op == "Iop_QDMulHi32Sx4":
            return "QDMulHiV"
        if op == "Iop_QRDMulHi16Sx8":
            return "QRDMulHiV"
        if op == "Iop_QRDMulHi32Sx4":
            return "QRDMulHiV"

        if op == "Iop_MullEven8Ux16":
            return "MullEvenV"
        if op == "Iop_MullEven16Ux8":
            return "MullEvenV"
        if op == "Iop_MullEven32Ux4":
            return "MullEvenV"
        if op == "Iop_MullEven8Sx16":
            return "MullEvenV"
        if op == "Iop_MullEven16Sx8":
            return "MullEvenV"
        if op == "Iop_MullEven32Sx4":
            return "MullEvenV"

        if op == "Iop_PolynomialMulAdd8x16":
            return "PolynomialMulAddV"
        if op == "Iop_PolynomialMulAdd16x8":
            return "PolynomialMulAddV"
        if op == "Iop_PolynomialMulAdd32x4":
            return "PolynomialMulAddV"
        if op == "Iop_PolynomialMulAdd64x2":
            return "PolynomialMulAddV"

        if op == "Iop_Avg8Ux16":
            return "AvgV"
        if op == "Iop_Avg16Ux8":
            return "AvgV"
        if op == "Iop_Avg32Ux4":
            return "AvgV"
        if op == "Iop_Avg64Ux2":
            return "AvgV"
        if op == "Iop_Avg8Sx16":
            return "AvgV"
        if op == "Iop_Avg16Sx8":
            return "AvgV"
        if op == "Iop_Avg32Sx4":
            return "AvgV"
        if op == "Iop_Avg64Sx2":
            return "AvgV"

        if op == "Iop_Max8Sx16":
            return "MaxV"
        if op == "Iop_Max16Sx8":
            return "MaxV"
        if op == "Iop_Max32Sx4":
            return "MaxV"
        if op == "Iop_Max64Sx2":
            return "MaxV"
        if op == "Iop_Max8Ux16":
            return "MaxV"
        if op == "Iop_Max16Ux8":
            return "MaxV"
        if op == "Iop_Max32Ux4":
            return "MaxV"
        if op == "Iop_Max64Ux2":
            return "MaxV"

        if op == "Iop_Min8Sx16":
            return "MinV"
        if op == "Iop_Min16Sx8":
            return "MinV"
        if op == "Iop_Min32Sx4":
            return "MinV"
        if op == "Iop_Min64Sx2":
            return "MinV"
        if op == "Iop_Min8Ux16":
            return "MinV"
        if op == "Iop_Min16Ux8":
            return "MinV"
        if op == "Iop_Min32Ux4":
            return "MinV"
        if op == "Iop_Min64Ux2":
            return "MinV"

        if op == "Iop_CmpEQ8x16":
            return "CmpEQV"
        if op == "Iop_CmpEQ16x8":
            return "CmpEQV"
        if op == "Iop_CmpEQ32x4":
            return "CmpEQV"
        if op == "Iop_CmpEQ64x2":
            return "CmpEQV"
        if op == "Iop_CmpGT8Sx16":
            return "CmpGTV"
        if op == "Iop_CmpGT16Sx8":
            return "CmpGTV"
        if op == "Iop_CmpGT32Sx4":
            return "CmpGTV"
        if op == "Iop_CmpGT64Sx2":
            return "CmpGTV"
        if op == "Iop_CmpGT8Ux16":
            return "CmpGTV"
        if op == "Iop_CmpGT16Ux8":
            return "CmpGTV"
        if op == "Iop_CmpGT32Ux4":
            return "CmpGTV"
        if op == "Iop_CmpGT64Ux2":
            return "CmpGTV"

        if op == "Iop_Cnt8x16":
            return "CntV"
        if op == "Iop_Clz8x16":
            return "ClzV"
        if op == "Iop_Clz16x8":
            return "ClzV"
        if op == "Iop_Clz32x4":
            return "ClzV"
        if op == "Iop_Clz64x2":
            return "ClzV"
        if op == "Iop_Cls8x16":
            return "ClsV"
        if op == "Iop_Cls16x8":
            return "ClsV"
        if op == "Iop_Cls32x4":
            return "ClsV"
        if op == "Iop_Ctz8x16":
            return "CtzV"
        if op == "Iop_Ctz16x8":
            return "CtzV"
        if op == "Iop_Ctz32x4":
            return "CtzV"
        if op == "Iop_Ctz64x2":
            return "CtzV"

        if op == "Iop_ShlV128":
            return "ShlV"
        if op == "Iop_ShrV128":
            return "ShrV"
        if op == "Iop_SarV128":
            return "SarV"

        if op == "Iop_ShlN8x16":
            return "ShlNV"
        if op == "Iop_ShlN16x8":
            return "ShlNV"
        if op == "Iop_ShlN32x4":
            return "ShlNV"
        if op == "Iop_ShlN64x2":
            return "ShlNV"
        if op == "Iop_ShrN8x16":
            return "ShrNV"
        if op == "Iop_ShrN16x8":
            return "ShrNV"
        if op == "Iop_ShrN32x4":
            return "ShrNV"
        if op == "Iop_ShrN64x2":
            return "ShrNV"
        if op == "Iop_SarN8x16":
            return "SarNV"
        if op == "Iop_SarN16x8":
            return "SarNV"
        if op == "Iop_SarN32x4":
            return "SarNV"
        if op == "Iop_SarN64x2":
            return "SarNV"

        if op == "Iop_Shl8x16":
            return "ShlV"
        if op == "Iop_Shl16x8":
            return "ShlV"
        if op == "Iop_Shl32x4":
            return "ShlV"
        if op == "Iop_Shl64x2":
            return "ShlV"
        if op == "Iop_QSal8x16":
            return "QSalV"
        if op == "Iop_QSal16x8":
            return "QSalV"
        if op == "Iop_QSal32x4":
            return "QSalV"
        if op == "Iop_QSal64x2":
            return "QSalV"
        if op == "Iop_QShl8x16":
            return "QShlV"
        if op == "Iop_QShl16x8":
            return "QShlV"
        if op == "Iop_QShl32x4":
            return "QShlV"
        if op == "Iop_QShl64x2":
            return "QShlV"
        if op == "Iop_QShlNsatSS8x16":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS16x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS32x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSS64x2":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU8x16":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU16x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU32x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatUU64x2":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU8x16":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU16x8":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU32x4":
            return "QShlNsatV"
        if op == "Iop_QShlNsatSU64x2":
            return "QShlNsatV"
        if op == "Iop_Shr8x16":
            return "ShrV"
        if op == "Iop_Shr16x8":
            return "ShrV"
        if op == "Iop_Shr32x4":
            return "ShrV"
        if op == "Iop_Shr64x2":
            return "ShrV"
        if op == "Iop_Sar8x16":
            return "SarV"
        if op == "Iop_Sar16x8":
            return "SarV"
        if op == "Iop_Sar32x4":
            return "SarV"
        if op == "Iop_Sar64x2":
            return "SarV"
        if op == "Iop_Sal8x16":
            return "SalV"
        if op == "Iop_Sal16x8":
            return "SalV"
        if op == "Iop_Sal32x4":
            return "SalV"
        if op == "Iop_Sal64x2":
            return "SalV"
        if op == "Iop_Rol8x16":
            return "RolV"
        if op == "Iop_Rol16x8":
            return "RolV"
        if op == "Iop_Rol32x4":
            return "RolV"
        if op == "Iop_Rol64x2":
            return "RolV"

        if op == "Iop_QandUQsh8x16":
            return "QandQshV"
        if op == "Iop_QandUQsh16x8":
            return "QandQshV"
        if op == "Iop_QandUQsh32x4":
            return "QandQshV"
        if op == "Iop_QandUQsh64x2":
            return "QandQshV"
        if op == "Iop_QandSQsh8x16":
            return "QandQshV"
        if op == "Iop_QandSQsh16x8":
            return "QandQshV"
        if op == "Iop_QandSQsh32x4":
            return "QandQshV"
        if op == "Iop_QandSQsh64x2":
            return "QandQshV"
        if op == "Iop_QandUQRsh8x16":
            return "QandQRshV"
        if op == "Iop_QandUQRsh16x8":
            return "QandQRshV"
        if op == "Iop_QandUQRsh32x4":
            return "QandQRshV"
        if op == "Iop_QandUQRsh64x2":
            return "QandQRshV"
        if op == "Iop_QandSQRsh8x16":
            return "QandQRshV"
        if op == "Iop_QandSQRsh16x8":
            return "QandQRshV"
        if op == "Iop_QandSQRsh32x4":
            return "QandQRshV"
        if op == "Iop_QandSQRsh64x2":
            return "QandQRshV"

        if op == "Iop_Sh8Sx16":
            return "ShV"
        if op == "Iop_Sh16Sx8":
            return "ShV"
        if op == "Iop_Sh32Sx4":
            return "ShV"
        if op == "Iop_Sh64Sx2":
            return "ShV"
        if op == "Iop_Sh8Ux16":
            return "ShV"
        if op == "Iop_Sh16Ux8":
            return "ShV"
        if op == "Iop_Sh32Ux4":
            return "ShV"
        if op == "Iop_Sh64Ux2":
            return "ShV"
        if op == "Iop_Rsh8Sx16":
            return "RshV"
        if op == "Iop_Rsh16Sx8":
            return "RshV"
        if op == "Iop_Rsh32Sx4":
            return "RshV"
        if op == "Iop_Rsh64Sx2":
            return "RshV"
        if op == "Iop_Rsh8Ux16":
            return "RshV"
        if op == "Iop_Rsh16Ux8":
            return "RshV"
        if op == "Iop_Rsh32Ux4":
            return "RshV"
        if op == "Iop_Rsh64Ux2":
            return "RshV"

        if op == "Iop_QandQShrNnarrow16Uto8Ux8":
            return "QandQShrNnarrow"
        if op == "Iop_QandQShrNnarrow32Uto16Ux4":
            return "QandQShrNnarrow"
        if op == "Iop_QandQShrNnarrow64Uto32Ux2":
            return "QandQShrNnarrow"
        if op == "Iop_QandQSarNnarrow16Sto8Sx8":
            return "QandQSarNnarrow"
        if op == "Iop_QandQSarNnarrow32Sto16Sx4":
            return "QandQSarNnarrow"
        if op == "Iop_QandQSarNnarrow64Sto32Sx2":
            return "QandQSarNnarrow"
        if op == "Iop_QandQSarNnarrow16Sto8Ux8":
            return "QandQSarNnarrow"
        if op == "Iop_QandQSarNnarrow32Sto16Ux4":
            return "QandQSarNnarrow"
        if op == "Iop_QandQSarNnarrow64Sto32Ux2":
            return "QandQSarNnarrow"
        if op == "Iop_QandQRShrNnarrow16Uto8Ux8":
            return "QandQRShrNnarrow"
        if op == "Iop_QandQRShrNnarrow32Uto16Ux4":
            return "QandQRShrNnarrow"
        if op == "Iop_QandQRShrNnarrow64Uto32Ux2":
            return "QandQRShrNnarrow"
        if op == "Iop_QandQRSarNnarrow16Sto8Sx8":
            return "QandQRSarNnarrow"
        if op == "Iop_QandQRSarNnarrow32Sto16Sx4":
            return "QandQRSarNnarrow"
        if op == "Iop_QandQRSarNnarrow64Sto32Sx2":
            return "QandQRSarNnarrow"
        if op == "Iop_QandQRSarNnarrow16Sto8Ux8":
            return "QandQRSarNnarrow"
        if op == "Iop_QandQRSarNnarrow32Sto16Ux4":
            return "QandQRSarNnarrow"
        if op == "Iop_QandQRSarNnarrow64Sto32Ux2":
            return "QandQRSarNnarrow"

        if op == "Iop_NarrowBin16to8x16":
            return "NarrowBin"
        if op == "Iop_NarrowBin32to16x8":
            return "NarrowBin"
        if op == "Iop_QNarrowBin16Uto8Ux16":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin32Sto16Ux8":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin16Sto8Ux16":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin32Uto16Ux8":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin16Sto8Sx16":
            return "QNarrowBin"
        if op == "Iop_QNarrowBin32Sto16Sx8":
            return "QNarrowBin"
        if op == "Iop_NarrowUn16to8x8":
            return "NarrowUn"
        if op == "Iop_NarrowUn32to16x4":
            return "NarrowUn"
        if op == "Iop_NarrowUn64to32x2":
            return "NarrowUn"
        if op == "Iop_QNarrowUn16Uto8Ux8":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn32Uto16Ux4":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn64Uto32Ux2":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn16Sto8Sx8":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn32Sto16Sx4":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn64Sto32Sx2":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn16Sto8Ux8":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn32Sto16Ux4":
            return "QNarrowUn"
        if op == "Iop_QNarrowUn64Sto32Ux2":
            return "QNarrowUn"
        if op == "Iop_Widen8Uto16x8":
            return "Widen"
        if op == "Iop_Widen16Uto32x4":
            return "Widen"
        if op == "Iop_Widen32Uto64x2":
            return "Widen"
        if op == "Iop_Widen8Sto16x8":
            return "Widen"
        if op == "Iop_Widen16Sto32x4":
            return "Widen"
        if op == "Iop_Widen32Sto64x2":
            return "Widen"

        if op == "Iop_InterleaveHI8x16":
            return "HInterleaveV"
        if op == "Iop_InterleaveHI16x8":
            return "HInterleaveV"
        if op == "Iop_InterleaveHI32x4":
            return "HInterleaveV"
        if op == "Iop_InterleaveHI64x2":
            return "HInterleaveV"
        if op == "Iop_InterleaveLO8x16":
            return "LInterleaveV"
        if op == "Iop_InterleaveLO16x8":
            return "LInterleaveV"
        if op == "Iop_InterleaveLO32x4":
            return "LInterleaveV"
        if op == "Iop_InterleaveLO64x2":
            return "LInterleaveV"

        if op == "Iop_CatOddLanes8x16":
            return "CatOddLanesV"
        if op == "Iop_CatOddLanes16x8":
            return "CatOddLanesV"
        if op == "Iop_CatOddLanes32x4":
            return "CatOddLanesV"
        if op == "Iop_CatEvenLanes8x16":
            return "CatEvenLanesV"
        if op == "Iop_CatEvenLanes16x8":
            return "CatEvenLanesV"
        if op == "Iop_CatEvenLanes32x4":
            return "CatEvenLanesV"

        if op == "Iop_InterleaveOddLanes8x16":
            return "InterleaveOddLanesV"
        if op == "Iop_InterleaveOddLanes16x8":
            return "InterleaveOddLanesV"
        if op == "Iop_InterleaveOddLanes32x4":
            return "InterleaveOddLanesV"
        if op == "Iop_InterleaveEvenLanes8x16":
            return "InterleaveEvenLanesV"
        if op == "Iop_InterleaveEvenLanes16x8":
            return "InterleaveEvenLanesV"
        if op == "Iop_InterleaveEvenLanes32x4":
            return "InterleaveEvenLanesV"
        if op == "Iop_PackOddLanes8x16":
            return "InterleavePackOddLanesV"
        if op == "Iop_PackOddLanes16x8":
            return "InterleavePackOddLanesV"
        if op == "Iop_PackOddLanes32x4":
            return "InterleavePackOddLanesV"
        if op == "Iop_PackEvenLanes8x16":
            return "InterleavePackEvenLanesV"
        if op == "Iop_PackEvenLanes16x8":
            return "InterleavePackEvenLanesV"
        if op == "Iop_PackEvenLanes32x4":
            return "InterleavePackEvenLanesV"

        if op == "Iop_GetElem8x16":
            return "GetElemV"
        if op == "Iop_GetElem16x8":
            return "GetElemV"
        if op == "Iop_GetElem32x4":
            return "GetElemV"
        if op == "Iop_GetElem64x2":
            return "GetElemV"

        if op == "Iop_SetElem8x16":
            return "SetElemV"
        if op == "Iop_SetElem16x8":
            return "SetElemV"
        if op == "Iop_SetElem32x4":
            return "SetElemV"
        if op == "Iop_SetElem64x2":
            return "SetElemV"

        if op == "Iop_GetElem8x8":
            return "GetElemV"
        if op == "Iop_GetElem16x4":
            return "GetElemV"
        if op == "Iop_GetElem32x2":
            return "GetElemV"
        if op == "Iop_SetElem8x8":
            return "SetElemV"
        if op == "Iop_SetElem16x4":
            return "SetElemV"
        if op == "Iop_SetElem32x2":
            return "SetElemV"

        if op == "Iop_Slice64":
            return "Slice"
        if op == "Iop_SliceV128":
            return "SliceV"

        if op == "Iop_Perm8x16":
            return "PermV"
        if op == "Iop_PermOrZero8x16":
            return "PermOrZeroV"
        if op == "Iop_Perm32x4":
            return "PermV"
        if op == "Iop_Perm8x16x2":
            return "PermV"
        if op == "Iop_Reverse8sIn16_x8":
            return "ReverseChunks"
        if op == "Iop_Reverse8sIn32_x4":
            return "ReverseChunks"
        if op == "Iop_Reverse16sIn32_x4":
            return "ReverseChunks"
        if op == "Iop_Reverse8sIn64_x2":
            return "ReverseChunks"
        if op == "Iop_Reverse16sIn64_x2":
            return "ReverseChunks"
        if op == "Iop_Reverse32sIn64_x2":
            return "ReverseChunks"
        if op == "Iop_Reverse1sIn8_x16":
            return "ReverseChunks"

        if op == "Iop_F32ToFixed32Ux4_RZ":
            return "ConvFFixedV"
        if op == "Iop_F32ToFixed32Sx4_RZ":
            return "ConvFFixedV"
        if op == "Iop_Fixed32UToF32x4_RN":
            return "ConvFixedFV"
        if op == "Iop_Fixed32SToF32x4_RN":
            return "ConvFixedFV"
        if op == "Iop_F32ToFixed32Ux2_RZ":
            return "ConvFFixedV"
        if op == "Iop_F32ToFixed32Sx2_RZ":
            return "ConvFFixedV"
        if op == "Iop_Fixed32UToF32x2_RN":
            return "ConvFixedFV"
        if op == "Iop_Fixed32SToF32x2_RN":
            return "ConvFixedFV"

        if op == "Iop_D32toD64":
            return "ExtD"
        if op == "Iop_D64toD32":
            return "TruncD"
        if op == "Iop_AddD64":
            return "AddD"
        if op == "Iop_SubD64":
            return "SubD"
        if op == "Iop_MulD64":
            return "MulD"
        if op == "Iop_DivD64":
            return "DivD"
        if op == "Iop_ShlD64":
            return "ShlD"
        if op == "Iop_ShrD64":
            return "ShrD"
        if op == "Iop_D64toI32S":
            return "TruncDI"
        if op == "Iop_D64toI32U":
            return "TruncDI"
        if op == "Iop_D64toI64S":
            return "ConvDI"
        if op == "Iop_D64toI64U":
            return "ConvDI"
        if op == "Iop_I32StoD64":
            return "ExtID"
        if op == "Iop_I32UtoD64":
            return "ExtID"
        if op == "Iop_I64StoD64":
            return "ConvID"
        if op == "Iop_I64UtoD64":
            return "ConvID"
        if op == "Iop_I32StoD128":
            return "ExtID"
        if op == "Iop_I32UtoD128":
            return "ExtID"
        if op == "Iop_I64StoD128":
            return "ExtID"
        if op == "Iop_I64UtoD128":
            return "ExtID"
        if op == "Iop_D64toD128":
            return "ExtD"
        if op == "Iop_D128toD64":
            return "TruncD"
        if op == "Iop_D128toI32S":
            return "TruncDI"
        if op == "Iop_D128toI32U":
            return "TruncDI"
        if op == "Iop_D128toI64S":
            return "TruncDI"
        if op == "Iop_D128toI64U":
            return "TruncDI"
        if op == "Iop_F32toD32":
            return "ConvFD"
        if op == "Iop_F32toD64":
            return "ExtFD"
        if op == "Iop_F32toD128":
            return "ExtFD"
        if op == "Iop_F64toD32":
            return "TruncFD"
        if op == "Iop_F64toD64":
            return "ConvFD"
        if op == "Iop_F64toD128":
            return "ExtFD"
        if op == "Iop_F128toD32":
            return "TruncFD"
        if op == "Iop_F128toD64":
            return "TruncFD"
        if op == "Iop_F128toD128":
            return "ConvFD"
        if op == "Iop_D32toF32":
            return "ConvDF"
        if op == "Iop_D32toF64":
            return "ExtDF"
        if op == "Iop_D32toF128":
            return "ExtDF"
        if op == "Iop_D64toF32":
            return "TruncDF"
        if op == "Iop_D64toF64":
            return "ConvDF"
        if op == "Iop_D64toF128":
            return "ExtDF"
        if op == "Iop_D128toF32":
            return "TruncDF"
        if op == "Iop_D128toF64":
            return "TruncDF"
        if op == "Iop_D128toF128":
            return "ConvDF"
        if op == "Iop_AddD128":
            return "AddD"
        if op == "Iop_SubD128":
            return "SubD"
        if op == "Iop_MulD128":
            return "MulD"
        if op == "Iop_DivD128":
            return "DivD"
        if op == "Iop_ShlD128":
            return "ShlD"
        if op == "Iop_ShrD128":
            return "ShrD"
        if op in ["Iop_RoundD64toInt", "Iop_RoundD128toInt"]:
            return "RndDI"
        if op in ["Iop_QuantizeD64", "Iop_QuantizeD128"]:
            return "QuantizeD"
        if op in ["Iop_ExtractExpD64", "Iop_ExtractExpD128"]:
            return "ExtractExpD"
        if op in ["Iop_ExtractSigD64", "Iop_ExtractSigD128"]:
            return "ExtractSigD"
        if op in ["Iop_InsertExpD64", "Iop_InsertExpD128"]:
            return "InsertExpD"
        if op in ["Iop_CmpD64", "Iop_CmpD128"]:
            return "CmpD"
        if op in ["Iop_CmpExpD64", "Iop_CmpExpD128"]:
            return "CmpExpD"
        if op == "Iop_D64HLtoD128":
            return "HLExtD"
        if op == "Iop_D128HItoD64":
            return "HTruncD"
        if op == "Iop_D128LOtoD64":
            return "LTruncD"
        if op in ["Iop_SignificanceRoundD64", "Iop_SignificanceRoundD128"]:
            return "SignificanceRndD"
        if op == "Iop_ReinterpI64asD64":
            return "ReinterpID"
        if op == "Iop_ReinterpD64asI64":
            return "ReinterpDI"
        if op == "Iop_V256to64_0":
            return "TruncVI"
        if op == "Iop_V256to64_1":
            return "TruncVI"
        if op == "Iop_V256to64_2":
            return "TruncVI"
        if op == "Iop_V256to64_3":
            return "TruncVI"
        if op == "Iop_64x4toV256":
            return "MergeIV"
        if op == "Iop_V256toV128_0":
            return "TruncVI"
        if op == "Iop_V256toV128_1":
            return "TruncVI"
        if op == "Iop_V128HLtoV256":
            return "HLExtV"
        if op == "Iop_DPBtoBCD":
            return "DPBtoBCD"
        if op == "Iop_BCDtoDPB":
            return "BCDtoDPB"
        if op == "Iop_Add64Fx4":
            return "AddFV"
        if op == "Iop_Sub64Fx4":
            return "SubFV"
        if op == "Iop_Mul64Fx4":
            return "MulFV"
        if op == "Iop_Div64Fx4":
            return "DivFV"
        if op == "Iop_Add32Fx8":
            return "AddFV"
        if op == "Iop_Sub32Fx8":
            return "SubFV"
        if op == "Iop_Mul32Fx8":
            return "MulFV"
        if op == "Iop_Div32Fx8":
            return "DivFV"
        if op == "Iop_I32StoF32x8":
            return "ConvIFV"
        if op == "Iop_F32toI32Sx8":
            return "ConvFIV"
        if op == "Iop_F32toF16x8":
            return "TruncFV"
        if op == "Iop_F16toF32x8":
            return "ExtFV"
        if op == "Iop_AndV256":
            return "AndV"
        if op == "Iop_OrV256":
            return "OrV"
        if op == "Iop_XorV256":
            return "XorV"
        if op == "Iop_NotV256":
            return "NotV"
        if op == "Iop_CmpNEZ64x4":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ32x8":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ16x16":
            return "CmpNEZV"
        if op == "Iop_CmpNEZ8x32":
            return "CmpNEZV"

        if op == "Iop_Add8x32":
            return "AddV"
        if op == "Iop_Add16x16":
            return "AddV"
        if op == "Iop_Add32x8":
            return "AddV"
        if op == "Iop_Add64x4":
            return "AddV"
        if op == "Iop_Sub8x32":
            return "SubV"
        if op == "Iop_Sub16x16":
            return "SubV"
        if op == "Iop_Sub32x8":
            return "SubV"
        if op == "Iop_Sub64x4":
            return "SubV"
        if op == "Iop_QAdd8Ux32":
            return "QAddV"
        if op == "Iop_QAdd16Ux16":
            return "QAddV"
        if op == "Iop_QAdd8Sx32":
            return "QAddV"
        if op == "Iop_QAdd16Sx16":
            return "QAddV"
        if op == "Iop_QSub8Ux32":
            return "QSubV"
        if op == "Iop_QSub16Ux16":
            return "QSubV"
        if op == "Iop_QSub8Sx32":
            return "QSubV"
        if op == "Iop_QSub16Sx16":
            return "QSubV"

        if op == "Iop_Mul16x16":
            return "MulV"
        if op == "Iop_Mul32x8":
            return "MulV"
        if op == "Iop_MulHi16Ux16":
            return "MulHiV"
        if op == "Iop_MulHi16Sx16":
            return "MulHiV"

        if op == "Iop_Avg8Ux32":
            return "AvgV"
        if op == "Iop_Avg16Ux16":
            return "AvgV"

        if op == "Iop_Max8Sx32":
            return "MaxV"
        if op == "Iop_Max16Sx16":
            return "MaxV"
        if op == "Iop_Max32Sx8":
            return "MaxV"
        if op == "Iop_Max8Ux32":
            return "MaxV"
        if op == "Iop_Max16Ux16":
            return "MaxV"
        if op == "Iop_Max32Ux8":
            return "MaxV"

        if op == "Iop_Min8Sx32":
            return "MinV"
        if op == "Iop_Min16Sx16":
            return "MinV"
        if op == "Iop_Min32Sx8":
            return "MinV"
        if op == "Iop_Min8Ux32":
            return "MinV"
        if op == "Iop_Min16Ux16":
            return "MinV"
        if op == "Iop_Min32Ux8":
            return "MinV"

        if op == "Iop_CmpEQ8x32":
            return "CmpEQV"
        if op == "Iop_CmpEQ16x16":
            return "CmpEQV"
        if op == "Iop_CmpEQ32x8":
            return "CmpEQV"
        if op == "Iop_CmpEQ64x4":
            return "CmpEQV"
        if op == "Iop_CmpGT8Sx32":
            return "CmpGTV"
        if op == "Iop_CmpGT16Sx16":
            return "CmpGTV"
        if op == "Iop_CmpGT32Sx8":
            return "CmpGTV"
        if op == "Iop_CmpGT64Sx4":
            return "CmpGTV"

        if op == "Iop_ShlN16x16":
            return "ShlNV"
        if op == "Iop_ShlN32x8":
            return "ShlNV"
        if op == "Iop_ShlN64x4":
            return "ShlNV"
        if op == "Iop_ShrN16x16":
            return "ShrNV"
        if op == "Iop_ShrN32x8":
            return "ShrNV"
        if op == "Iop_ShrN64x4":
            return "ShrNV"
        if op == "Iop_SarN16x16":
            return "SarNV"
        if op == "Iop_SarN32x8":
            return "SarNV"

        if op == "Iop_Perm32x8":
            return "PermV"

        if op == "Iop_CipherV128":
            return "CipherV"
        if op == "Iop_CipherLV128":
            return "CipherLV"
        if op == "Iop_NCipherV128":
            return "NCipherV"
        if op == "Iop_NCipherLV128":
            return "NCipherLV"
        if op == "Iop_CipherSV128":
            return "CipherSV"

        if op in ["Iop_SHA256", "Iop_SHA512"]:
            return "SHA"
        if op == "Iop_BCDAdd":
            return "BCDAdd"
        if op == "Iop_BCDSub":
            return "BCDSub"
        if op == "Iop_I128StoBCD128":
            return "bcdcfsq."
        if op == "Iop_BCD128toI128S":
            return "bcdctsq."
        if op == "Iop_Rotx32":
            return "bitswap"
        if op == "Iop_Rotx64":
            return "dbitswap"

        if op == "Iop_PwBitMtxXpose64x2":
            return "BitMatrixTranspose"
            # return "Iop_UNKNOWN"
        return op

    def processIRType(self, ty: str):
        if ty == "Ity_INVALID":
            return "Ity_INVALID"
        if ty == "Ity_I1":
            return "INTEGER"
        if ty == "Ity_I8":
            return "INTEGER"
        if ty == "Ity_I16":
            return "INTEGER"
        if ty == "Ity_I32":
            return "INTEGER"
        if ty == "Ity_I64":
            return "INTEGER"
        if ty == "Ity_I128":
            return "INTEGER"
        if ty == "Ity_F16":
            return "FLOAT"
        if ty == "Ity_F32":
            return "FLOAT"
        if ty == "Ity_F64":
            return "FLOAT"
        if ty == "Ity_D32":
            return "DECIMAL"
        if ty == "Ity_D64":
            return "DECIMAL"
        if ty == "Ity_D128":
            return "DECIMAL"
        if ty == "Ity_F128":
            return "FLOAT"
        if ty == "Ity_V128":
            return "VECTOR"
        if ty == "Ity_V256":
            return "VECTOR"
            # return "Ity_UNKNOWN"
        return ty

        # In[ ]:

    def processIREffect(self, fx):  # space at beg and \n at end
        # return " " + fx + "\n"
        return fx

    def processIRExpr(self, er: py.expr.IRExpr, irsb):
        exp = {"opc": "", "type": "", "arg": []}
        if isinstance(er, py.const.IRConst):
            exp["opc"] = "CONSTANT"
            exp["type"] = self.processIRType(er.type)
            return exp

        assert isinstance(er, py.expr.IRExpr)
        tag = er.tag
        if tag == "Iex_Binder":
            exp["opc"] = "Binder"
            exp["type"] = "INTEGER"
            exp["arg"].append("CONSTANT")
            return exp
        if tag == "Iex_Get":  # RETTYPE
            exp["opc"] = "Get"
            exp["type"] = self.processIRType(er.ty)
            exp["arg"].append("REGISTER")
            return exp
        if tag == "Iex_GetI":
            exp["opc"] = "GetI"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            exp["arg"].append(self.processIRExpr(er.ix, irsb))
            return exp
        if tag == "Iex_RdTmp":
            exp["opc"] = "VARIABLE"
            exp["type"] = self.processIRType(irsb.tyenv.types[er.tmp])
            return exp
        if tag == "Iex_Qop":
            exp["opc"] = self.processIrop(er.op)
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            for i in range(4):
                exp["arg"].append(self.processIRExpr(er.args[i], irsb))
            return exp
        if tag == "Iex_Triop":
            exp["opc"] = self.processIrop(er.op)
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            for i in range(3):
                exp["arg"].append(self.processIRExpr(er.args[i], irsb))
            return exp
        if tag == "Iex_Binop":
            exp["opc"] = self.processIrop(er.op)
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            for i in range(2):
                exp["arg"].append(self.processIRExpr(er.args[i], irsb))
            return exp
        if tag == "Iex_Unop":
            exp["opc"] = self.processIrop(er.op)
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            exp["arg"].append(self.processIRExpr(er.args[0], irsb))
            return exp
        if tag == "Iex_Load":
            exp["opc"] = "Load"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            exp["arg"].append(self.processIRExpr(er.addr, irsb))
            return exp
        if tag == "Iex_Const":
            exp["opc"] = "CONSTANT"
            exp["type"] = self.processIRType(er.con.type)
            return exp
        if tag == "Iex_ITE":
            exp["opc"] = "ITE"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            exp["arg"].append(self.processIRExpr(er.cond, irsb))
            exp["arg"].append(self.processIRExpr(er.iftrue, irsb))
            exp["arg"].append(self.processIRExpr(er.iffalse, irsb))
            return exp
        if tag == "Iex_CCall":
            exp["opc"] = "FUNCTION"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            for e in er.args:
                exp["arg"].append(self.processIRExpr(e, irsb))
            return exp
        if tag == "Iex_VECRET":
            exp["opc"] = "VECRET"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            return exp
        if tag == "Iex_GSPTR":
            exp["opc"] = "GSPTR"
            exp["type"] = self.processIRType(er.result_type(irsb.tyenv))
            return exp
        else:
            exp["opc"] = "UNKNOWN_OPC"
            exp["type"] = "UNKNOWN_TYPE"
            return exp

    def resetSt(self):
        return {"opc": "", "loc": "", "rhs": "", "type": "", "arg": []}

    def processExit(self, irsb, dst, jk):
        putrhs = self.resetSt()
        putrhs["opc"] = "Put"
        putrhs["loc"] = "REGISTER"
        putrhs["rhs"] = self.processIRExpr(dst, irsb)

        exitrhs = self.resetSt()
        exitrhs["opc"] = "Exit"
        arg = "JUMP_KIND"
        exitrhs["arg"] = [arg]

        return putrhs, exitrhs

    def processSB(self, block):
        irsb = block.vex
        sts = []
        for stmt in block.vex.statements:
            st = self.resetSt()
            tag = stmt.tag

            if tag == "Ist_NoOp":
                continue

            elif tag == "Ist_AbiHint":
                continue

            elif tag == "Ist_Put":
                st["opc"] = "Put"
                st["loc"] = "REGISTER"
                st["rhs"] = self.processIRExpr(stmt.data, irsb)
                sts.append(st)
                continue

            elif tag == "Ist_PutI":
                st["opc"] = "PutI"
                st["loc"] = self.processIRExpr(stmt.ix, irsb)
                st["rhs"] = self.processIRExpr(stmt.data, irsb)
                sts.append(st)
                continue

            elif tag == "Ist_MBE":
                st["opc"] = "MBE"
                st["arg"].append(stmt.event)
                sts.append(st)
                continue
            elif tag == "Ist_IMark":
                continue
            elif tag == "Ist_WrTmp":
                st["opc"] = "WrTmp"
                st["rhs"] = self.processIRExpr(stmt.data, irsb)
                sts.append(st)
                continue

            elif tag == "Ist_Store":
                st["opc"] = "Store"
                st["loc"] = self.processIRExpr(stmt.addr, irsb)
                st["rhs"] = self.processIRExpr(stmt.data, irsb)
                sts.append(st)
                continue

            elif tag == "Ist_StoreG":
                st["opc"] = "If"
                st["arg"].append(self.processIRExpr(stmt.guard, irsb))

                rhs = self.resetSt()
                rhs["opc"] = "Store"
                rhs["loc"] = self.processIRExpr(stmt.addr, irsb)
                rhs["rhs"] = self.processIRExpr(stmt.data, irsb)

                st["rhs"] = rhs
                sts.append(st)
                continue

            elif tag == "Ist_LoadG":
                st["opc"] = "WrTmp"

                ifrhs = self.resetSt()
                ifrhs["opc"] = "If"
                ifrhs["arg"].append(self.processIRExpr(stmt.guard, irsb))
                if stmt.cvt == "ILGop_INVALID":
                    rhs = "ILGop_INVALID"
                elif stmt.cvt == "ILGop_IdentV128":
                    rhs = "ILGop_IdentV"
                elif stmt.cvt.startswith("ILGop_Ident"):
                    rhs = "ILGop_Ident"
                elif stmt.cvt.startswith("ILGop_"):
                    rhs = "ILGop_Ext"
                newEx = {
                    "opc": "Load",
                    "type": self.processIRType(stmt.dst),
                    "arg": self.processIRExpr(stmt.addr, irsb),
                }
                ifrhs["rhs"] = {"opc": rhs, "type": "", "arg": newEx}

                elserhs = self.resetSt()
                elserhs["opc"] = "Else"
                elserhs["rhs"] = self.processIRExpr(stmt.alt, irsb)

                st["rhs"] = [ifrhs, elserhs]
                sts.append(st)
                continue

            elif tag == "Ist_Dirty":
                st["opc"] = "WrTmp"

                dirtyrhs = self.resetSt()
                dirtyrhs["opc"] = "Dirty"
                dirtyrhs["arg"].append(self.processIRExpr(stmt.guard, irsb))
                invalid_vals = (0xFFFFFFFF, -1)
                ty = ""
                if stmt.tmp not in invalid_vals:
                    ty = self.processIRType(irsb.tyenv.types[stmt.tmp])
                newEx = {
                    "opc": "FUNCTION",
                    "type": ty,
                    "arg": [self.processIRExpr(e, irsb) for e in stmt.args],
                }
                dirtyrhs["arg"].append(newEx)

                st["rhs"] = dirtyrhs
                sts.append(st)
                continue

            elif tag == "Ist_CAS":
                st["opc"] = "WrTmp"

                casrhs = self.resetSt()
                casrhs["opc"] = "CAS"
                casrhs["arg"].append(self.processIRExpr(stmt.expdLo, irsb))
                if stmt.expdHi is not None:
                    casrhs["arg"].append(self.processIRExpr(stmt.expdHi, irsb))
                else:
                    casrhs["arg"].append("NONE")
                casrhs["arg"].append(self.processIRExpr(stmt.dataLo, irsb))
                if stmt.dataHi is not None:
                    casrhs["arg"].append(self.processIRExpr(stmt.dataHi, irsb))
                else:
                    casrhs["arg"].append("NONE")
                casrhs["loc"] = self.processIRExpr(stmt.addr, irsb)
                st["rhs"] = casrhs
                sts.append(st)
                continue

            elif tag == "Ist_LLSC":
                st["opc"] = "WrTmp"

                rhs = self.resetSt()
                if stmt.storedata is None:
                    rhs["opc"] = "Load"
                    rhs["arg"].append(self.processIRExpr(stmt.addr, irsb))
                    rhs["type"] = self.processIRType(irsb.tyenv.types[stmt.result])
                else:
                    rhs["opc"] = "Store"
                    rhs["loc"].append(self.processIRExpr(stmt.addr, irsb))
                    rhs["arg"].append(self.processIRExpr(stmt.storedata, irsb))

                st["rhs"] = rhs
                sts.append(st)
                continue

            elif tag == "Ist_Exit":
                st["opc"] = "If"
                st["arg"].append(self.processIRExpr(stmt.guard, irsb))
                putrhs, exitrhs = self.processExit(irsb, stmt.dst, stmt.jk)
                st["rhs"] = [putrhs, exitrhs]
                sts.append(st)
                continue

                # END OF SUPER BLOCK
        st = self.resetSt()
        st["opc"] = "DUMMY_OP"
        putrhs, exitrhs = self.processExit(irsb, irsb.next, irsb.jumpkind)
        st["rhs"] = [putrhs, exitrhs]
        sts.append(st)
        # Separator for super block level triplets and embeddings
        sts.append("IRSB")
        return sts

    def processFunc(self, func):
        sts = []
        for block in func.blocks:
            if block.size > 0:
                sts.extend(self.processSB(block))
        return sts
