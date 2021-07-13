import ast, os, argparse
from exceptions import *

class Py2Cuda:
    def __init__(self, file_name, arch="sm_61"):
        self._predefiend_funcs = {
            "print" : "printf",
            "deviceSync" : "cudaDeviceSynchronize"
        }
        self._type_dict = {
            "str" : "std::string",
            "kernel" : "__global__ void",
            "device" : "__device__",
        }
        self._func_type_dict = {
            "stop_device_timer" : "float",
            "sqrtf" : "float",
            "sqrt" : "float",
            "ceil" : "float",
            "power" : "double",
            "pow" : "double",
        }
        self._func_arg_types = {
            "sqrtf" : ["float"],
            "sqrt" : ["double"],
            "power" : ["double", "double"],
            "pow" : ["double", "double"],
            "ceil" : ["double"],
        }
        self._var_type_dict = {
            "threadIdx.x" : "int",
            "threadIdx.y" : "int",
            "blockDim.x" : "int",
            "blockDim.y" : "int",
            "blockIdx.x" : "int",
            "blockIdx.y" : "int",
            "gridDim.x" : "int",
            "gridDim.y" : "int",
        }
        self._format_type = {
            "int" : "d",
            "float" : ".10f",
            "double" : ".10lf",
            "char" : "c",
            "str" : "s",
            "std::string" : "s"
        }
        self._no_semi_coma = {"{", "[", "(", "\n"}
        self._needs_free = []
        self._needs_cuda_free = []
        self._current_func = []
        self._in_kernel = False
        self._variable_scope = []

        self.file_name = file_name
        self.arch = arch

        f = open(self.file_name)
        self.original_text = f.read()
        f.close()

        self.original_tree = ast.parse(self.original_text)

        self.thrower = LineException(self.file_name)

        self.output_name = "./compiled/test.cu"
        self.generate_cuda()
        self.compile_cuda()

    def check_if_passed_to_kernel(self, ctx, var_name):
        kernel_calls = []
        for node in ast.walk(ctx):
            if type(node) == ast.Call and type(node.func) == ast.Subscript and type(node.func.slice.value) == ast.Tuple:
                if len(node.func.slice.value.elts) > 1:
                    kernel_calls.append(node)
        
        kernel_calls = [node for node in kernel_calls if any([self.visit(i)==var_name for i in node.args])]
        return kernel_calls

    def check_if_passed_to_function(self, ctx, var_name):
        calls = []
        for node in ast.walk(ctx):
            if type(node) == ast.Call:
                calls.append(node)
        
        calls = [node for node in calls if any([self.visit(i)==var_name for i in node.args])]
        return calls

    def get_function(self, name, ctx):
        for node in ast.walk(ctx):
            if type(node) == ast.FunctionDef and self.visit(node.name) == name:
                return node

    def contains_return(self, ctx):
        for node in ast.walk(ctx):
            if type(node) == ast.Return:
                return True
        return False
    

    def get_type(self, ctx, use_dict=True):
        string = str(type(ctx))
        string = string[string.find("'")+1:string.rfind("'")]
        if "." in string:
            string = string[string.find('.')+1:]
        if use_dict and string in self._type_dict:
            return self._type_dict[string]
        else:
            return string

    def is_const(self, ctx):
        if type(ctx) == ast.Constant:
            return True
        elif type(ctx) == ast.BinOp:
            return self.is_const(ctx.left) and self.is_const(ctx.right)
        elif type(ctx) == ast.UnaryOp:
            return self.is_const(ctx.operand)

    def get_actual_type(self, ctx): # TODO: better type detection.
        if self.is_const(ctx):
            return self.get_type(eval(self.visit(ctx)))

        if type(ctx) == ast.Call and type(ctx.func) == ast.Name:
            func = self.get_function(self.visit(ctx.func), self.original_tree)
            if func != None:
                if type(func.returns) != ast.Tuple:
                    return self.visit(func.returns)
            else:
                if self.visit(ctx.func) in self._func_type_dict:
                    return self._func_type_dict[self.visit(ctx.func)]
        
        if self.visit(ctx) in self._var_type_dict:
            return self._var_type_dict[self.visit(ctx)]
        if self.visit(ctx) in self._variable_scope[-1]:
            name = self.visit(ctx)
            if self._variable_scope[-1][name][0] != "auto":
                return self._variable_scope[-1][name][0]
        
        if type(ctx) == ast.Subscript and self.visit(ctx.value) in self._variable_scope[-1]:
            name = self.visit(ctx.value)
            var_type = self._variable_scope[-1][name][0]
            if var_type != "auto":
                if var_type[-1] == "*":
                    return var_type[:-1]
                else:
                    return var_type
        
        if type(ctx) == ast.BinOp:
            left = self.get_actual_type(ctx.left)
            right = self.get_actual_type(ctx.right)
            if left != None and right != None:
                if left == "int":
                    left = "1"
                else:
                    left = f"{left}()"
                
                if right == "int":
                    right = "1"
                else:
                    right = f"{right}()"

                return self.get_type(eval(f"{left} {self.visit(ctx.op)} {right}"))
            else:
                raise TypeError("TypeError: 'NoneType' object is not callable")
            
        if type(ctx) == ast.UnaryOp:
            return self.get_actual_type(ctx.operand)

    def get_var_type(self, name, val, ctx):
        var_type = ""
        calls = self.check_if_passed_to_function(ctx, name)
        for call in calls:
            try:
                func_name = call.func
                if type(func_name) == ast.Subscript:
                    func_name = func_name.value
                
                args = call.args
                keywords = call.keywords

                arg_index = [self.visit(i) for i in args].index(name)
                if arg_index == -1:
                    arg_index = [self.visit(i.value) for i in keywords].index(name)
                
                func = self.get_function(self.visit(func_name), self.original_tree)
                if func != None:
                    types = self.get_arg_types(func.args)
                    var_type = types[arg_index]
                    break
            except:
                pass
        
        if var_type == "":
            if type(val) in [ast.List, ast.Tuple, ast.Set]:
                for i in val.elts:
                    try:
                        var_type = self.get_actual_type(i)
                        break
                    except:
                        pass
            else:
                try:
                    var_type = self.get_actual_type(val)
                except:
                    pass

        if var_type == None:
            var_type = ""
        
        return var_type


    def visit(self, ctx):
        func_name = f"visit{self.get_type(ctx, False)}"

        if hasattr(self, func_name):
            return self.__getattribute__(func_name)(ctx)
        else:
            return ""
    

    def generate_cuda(self):
        # print(ast.dump(self.original_tree.body[2]))
        file_start = open("file_start.cu")
        cuda_output = file_start.read()
        file_start.close()
        cuda_output += self.visitlist(self.original_tree.body, is_main=True)
        
        print(cuda_output)

        cuda_file = open(self.output_name, "w")
        cuda_file.write(cuda_output)
        cuda_file.close()

    def compile_cuda(self):
        dir_name, file_name = os.path.split(os.path.abspath(self.output_name))
        os.system(f"cd {dir_name} && nvcc {file_name} -arch={self.arch} -o test && test")

    def visitlist(self, ctx:list, is_func=False, is_main=False):
        output = ""
        index = 0
        for child in ctx:
            if not is_main:
                line = "    " + self.visit(child)
                output += line.replace("\n", "\n    ")
            else:
                output += self.visit(child)
            
            if output != "" and output[-1] not in self._no_semi_coma:
                output += ";"
            output += "\n"

            if (index == len(ctx)-1 or self.contains_return(ctx[index+1])) and is_func:
                for to_free in self._needs_free[-1]:
                    output += f"    free({to_free});\n"
                for to_free in self._needs_cuda_free[-1]:
                    output += f"    cudaFree({to_free});\n"
                self._needs_free[-1] = []
                self._needs_cuda_free[-1] = []

            index += 1

        return output

    def visitstr(self, ctx: str):
        return ctx

    def visitName(self, ctx:ast.Name):
        return ctx.id

    def visitConstant(self, ctx:ast.Constant):
        if ctx.value == None:
            return "NULL"
        if type(ctx.value) == str:
            val = ctx.value.replace('\"', '\\"').replace("\'", "\\'").replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("\b", "\\b").replace("\f", "\\f")
            return '"' + val + '"'
        elif type(ctx.value) == bool:
            return str(ctx.value).lower()
        return str(ctx.value)

    def visitAdd(self, ctx: ast.Add):
        return "+"
    def visitSub(self, ctx: ast.Sub):
        return "-"
    def visitUSub(self, ctx: ast.USub):
        return "-"
    def visitMult(self, ctx: ast.Mult):
        return "*"
    def visitDiv(self, ctx: ast.Div):
        return "/"
    def visitMod(self, ctx: ast.Mod):
        return "%"
    def visitPow(self, ctx: ast.Pow):
        return "**"
     
    def visitEq(self, ctx: ast.Eq):
        return "=="
    def visitNotEq(self, ctx: ast.NotEq):
        return "!="
    def visitLt(self, ctx: ast.Lt):
        return "<"
    def visitLtE(self, ctx: ast.LtE):
        return "<="
    def visitGt(self, ctx: ast.Gt):
        return ">"
    def visitGtE(self, ctx: ast.GtE):
        return "<="

    def visitAnd(self, ctx: ast.And):
        return "&&"
    def visitOr(self, ctx: ast.Or):
        return "||"

    def visitBinOp(self, ctx:ast.BinOp):
        if type(ctx.op) not in (ast.Pow, ast.Div):
            return "(" + self.visit(ctx.left) + self.visit(ctx.op) + self.visit(ctx.right) + ")"
        elif type(ctx.op) == ast.Div:
            return "((double)(" + self.visit(ctx.left) + ")" + self.visit(ctx.op) + "(double)(" + self.visit(ctx.right) + "))"
        elif self._in_kernel and type(ctx.op) == ast.Pow:
            return f"power({self.visit(ctx.left)}, {self.visit(ctx.right)})"
        elif type(ctx.op) == ast.Pow:
            return f"pow({self.visit(ctx.left)}, {self.visit(ctx.right)})"

    def visitUnaryOp(self, ctx: ast.UnaryOp):
        return self.visit(ctx.op) + self.visit(ctx.operand)

    def visitImport(self, ctx: ast.Import):
        output = ""
        for name in ctx.names:
            output += f"#include <{name.name}>\n"
        return output

    def visitReturn(self, ctx:ast.Return):
        return "return " + self.visit(ctx.value)
        

    def visitCompare(self, ctx:ast.Compare):
        output = ""
        comparators = [ctx.left] + ctx.comparators
        index = 0
        for op in ctx.ops:
            output += "&&"*(output!="") + "(" + self.visit(comparators[index]) + self.visit(op) + self.visit(comparators[index+1]) + ")"
            index += 1
        return output

    def visitIndex(self, ctx: ast.Index):
        if type(ctx.value) == ast.Tuple:
            return "[" + ", ".join([self.visit(i) for i in ctx.value.elts]) + "]"
        return "[" + self.visit(ctx.value) + "]"

    def visitSlice(self, ctx: ast.slice):
        return self.visit(ctx.index)

    def visitSubscript(self, ctx: ast.Subscript): # TODO: fix kernel call identification
        val = self.visit(ctx.value)
        slice = self.visit(ctx.slice)
        
        if type(ctx.slice.value) == ast.Tuple:
            slice = "<<<" + slice[1:-1] + ">>>"
        return val + slice

    def visitkeyword(self, ctx: ast.keyword):
        return f"{self.visit(ctx.arg)} = {self.visit(ctx.value)}"

    def visitCall(self, ctx: ast.Call):
        func_name = self.visit(ctx.func)
        if func_name in self._predefiend_funcs:
            func_name = self._predefiend_funcs[func_name]
        
        if func_name == "printf":
            end = "\\n"
            for keyword in ctx.keywords:
                if self.visit(keyword.arg) == "end":
                    end = self.visit(keyword.value)
            ctx.keywords = [i for i in ctx.keywords if self.visit(i.arg) != "end"]
        
        base_func_name = func_name if "<" not in func_name else func_name[:func_name.find("<")]
        func = self.get_function(base_func_name, self.original_tree)
        if func != None:
            arg_types = self.get_arg_types(func.args)
            args = ""
            index = 0
            for i in ctx.args:
                args += f"({arg_types[index]})({self.visit(i)}), "
                index += 1
            for i in ctx.keywords:
                args += f"{self.visit(i.arg)} = ({arg_types[index]})({self.visit(i.value)}), "
                index += 1
            if args[-2:] == ", ":
                args = args[:-2]
        
        elif base_func_name in self._func_arg_types:
            arg_types = self._func_arg_types[base_func_name]
            args = ""
            index = 0
            for i in ctx.args:
                args += f"({arg_types[index]})({self.visit(i)}), "
                index += 1
            for i in ctx.keywords:
                args += f"{self.visit(i.arg)} = ({arg_types[index]})({self.visit(i.value)}), "
                index += 1
            if args[-2:] == ", ":
                args = args[:-2]
        else:
            args = ", ".join([self.visit(i) for i in ctx.args]) + ", "*(len(ctx.keywords)!=0) + ", ".join([self.visit(i) for i in ctx.keywords])
        

        if func_name == "printf":
            s = ""
            for arg in ctx.args:
                arg_type = self.get_actual_type(arg)
                if arg_type != None:
                    s += "%" + self._format_type[arg_type]
            s += "%s"
            output = func_name + '("' + s + '", ' + args + ', "' + end + '")'
        else:
            output = func_name + "(" + args + ")"


        return output

    def visitAttribute(self, ctx: ast.Attribute):
        return f"{self.visit(ctx.value)}.{self.visit(ctx.attr)}"

    def visitExpr(self, ctx: ast.Expr):
        return self.visit(ctx.value)


    def visitList(self, ctx: ast.List, is_cuda_assign=False):
        if not is_cuda_assign:
            return "[" + ", ".join(self.visit(i) for i in ctx.elts) + "]"
        


    def visitArrayAssign(self, name, arr_type, value, length):
        var_name = self.visit(name)
        output = ""
        if type(arr_type) != str:
            arr_type = self.visit(arr_type.value)

        re_define = False
        if var_name in self._variable_scope[-1]:
            re_define = True
        
        if var_name in self._needs_cuda_free[-1]:
            output = f"cudaFree({var_name});\n"
            self._needs_cuda_free[-1].remove(var_name)

        if var_name in self._needs_free[-1]:
            output = f"cudaFree({var_name});\n"
            self._needs_free[-1].remove(var_name)


        calls = self.check_if_passed_to_kernel(self._current_func[-1], var_name)
        if len(calls) > 0:
            self._needs_cuda_free[-1].append(var_name)
            
            if not re_define:
                output += f"{arr_type} {var_name};\n"
            output += f"cudaMallocManaged(&{var_name}, {length}*sizeof({arr_type}))"
            if type(value) == ast.List:
                output += ";\n" + ";\n".join([f"{var_name}[{i}] = {self.visit(value.elts[i])}" for i in range(len(value.elts))])
        else:
            if not re_define:
                output = f"{arr_type} {var_name}[{length}]"
                if type(value) == ast.List:
                    output += " = {" + self.visit(value)[1:-1] + "}"
            else:
                output += ";\n".join([f"{var_name}[{i}] = {self.visit(value.elts[i])}" for i in range(len(value.elts))])

        return output

    def visitAnnAssign(self, ctx: ast.AnnAssign):
        if type(ctx.annotation) == ast.Subscript:
            return self.visitArrayAssign(ctx.target, ctx.annotation, ctx.value, self.visit(ctx.annotation.slice.value))
        
        else:
            var_type = self.visit(ctx.annotation)
            name = self.visit(ctx.target)
            val = ctx.value

            self._variable_scope[-1][name] = (var_type, val)
            
            return f"{var_type} {name} = {self.visit(val)}"
    
    
    def visitAssign(self, ctx: ast.Assign):
        output = ""
        if type(ctx.targets[0]) != ast.Tuple:
            ctx.targets[0] = ast.Tuple([ctx.targets[0]])


        for i, var_name in enumerate(ctx.targets[0].elts):
            if type(ctx.value) == ast.Tuple:
                val = ctx.value[i]
            else:
                val = ctx.value

            name = self.visit(var_name)
            if name not in self._variable_scope[-1]:
                if type(val) == ast.List:
                    arr_type = ""

                    arr_type = self.get_var_type(name, val, self._current_func[-1])
                    
                    if arr_type == "":
                        self.thrower._raise((ctx.lineno, self.original_text.split("\n")[ctx.lineno-1]), "SyntaxError", "Must specify array type.")

                    output += self.visitArrayAssign(var_name, arr_type, val, len(val.elts))
                    self._variable_scope[-1][name] = (arr_type, val)
                elif type(val) == ast.Str:
                    print(ast.dump(val))
                else:
                    var_type = self.get_var_type(name, val, self._current_func[-1])
                    
                    if var_type == "":
                        var_type = "auto"
                    
                    output += f"{var_type} {name} = {self.visit(val)}"
                    self._variable_scope[-1][name] = (var_type, val)
            else:
                if type(val) == ast.List:
                    arr_type = ""

                    arr_type = self.get_var_type(name, val, self._current_func[-1])
                    
                    if arr_type == "":
                        self.thrower._raise((ctx.lineno, self.original_text.split("\n")[ctx.lineno-1]), "SyntaxError", "Must specify array type.")
                    
                    output += self.visitArrayAssign(var_name, arr_type, val, len(val.elts))
                else:
                    output += f"{name} = {self.visit(val)}"
        return output

    def visitAugAssign(self, ctx: ast.AugAssign):
        return f"{self.visit(ctx.target)} {self.visit(ctx.op)}= {self.visit(ctx.value)}"
    
    def visitIf(self, ctx:ast.If):
        output = "if ("
        output += self.visit(ctx.test) + ")"
        output += "{\n" + self.visit(ctx.body) + "}"

        for orelse in ctx.orelse:
            if type(orelse) == ast.If:
                output += "\nelse " + self.visit(orelse)
            else:
                output += "\nelse {\n" + self.visit(orelse) + "}"
        return output


    def visitFor(self, ctx:ast.For):
        output = ""
        var_name = self.visit(ctx.target)

        if ctx.iter.func.id == "range":
            args = ctx.iter.args
            low = 0
            high = 0
            jump = 1
            if len(args) == 1:
                high = self.visit(args[0])
            elif len(args) == 2:
                low = self.visit(args[0])
                high = self.visit(args[1])
            elif len(args) == 3:
                low = self.visit(args[0])
                high = self.visit(args[1])
                jump = self.visit(args[2])

            if float(jump) == 1:
                output = f"for (int {var_name} = {low}; {var_name}<{high}; {var_name}++)"
            elif float(jump) == -1:
                output = f"for (int {var_name} = {low}; {var_name}<{high}; {var_name}--)"
            else:
                output = f"for (int {var_name} = {low}; {var_name}<{high}; {var_name} += {jump})"

        output += "{\n" + self.visit(ctx.body) + "}"
        return output

    def visitWhile(self, ctx: ast.While):
        return "while (" + self.visit(ctx.test) + "){\n" + self.visit(ctx.body) + "}"

    def get_arg_types(self, ctx: ast.arguments):
        index = 0
        types = []

        for arg in ctx.args:
            if arg.annotation != None:
                if type(arg.annotation) == ast.Subscript:
                    types.append(self.visit(arg.annotation.value).replace('"', '').replace("'", "") + "*")
                else:
                    types.append(self.visit(arg.annotation).replace('"', '').replace("'", ""))
            

            elif len(ctx.args) - index <= len(ctx.defaults):
                try:
                    arg_type = self.get_actual_type(ctx.defaults[index - len(ctx.args)])
                    types.append(arg_type)
                except:
                    types.append("auto")
                    
            else:
                types.append("auto")

            index += 1
        return types

    def visitArguments(self, ctx: ast.arguments, func=None):
        output = ""
        index = 0
        no_type_error = ((func.lineno, self.original_text.split("\n")[func.lineno-1]), "SyntaxError", "Must specify argument type.")
        types = self.get_arg_types(ctx)
        for arg in ctx.args:
            arg_type = types[index]
            if func != None and arg_type == "auto":
                self.thrower._raise(*no_type_error)
            output += ", "*(output!="") + arg_type + " " + arg.arg
            
            if len(ctx.args) - index <= len(ctx.defaults):
                output += "=" + self.visit(ctx.defaults[index - len(ctx.args)])
            index += 1
        
        return output

    def visitFunctionDef(self, ctx:ast.FunctionDef):
        self._variable_scope.append(dict())

        name = ctx.name
        args = self.visitArguments(ctx.args, ctx)
        if ctx.returns != None:
            if type(ctx.returns) == ast.Tuple:
                ret_type = [self.visit(i) for i in ctx.returns.elts]

                for i, t in enumerate(ret_type):
                    if t in self._type_dict:
                        ret_type[i] = self._type_dict[t]
                
                if "__global__" in ret_type[0]:
                    ret_type[0] = "__global__"
                
                ret_type = " ".join(ret_type)
            else:
                ret_type = self.visit(ctx.returns)
                if ret_type in self._type_dict:
                    ret_type = self._type_dict[ret_type]
        else:
            if name == "main":
                ret_type = "int"
            else:
                ret_type = "auto"

        output = f"{ret_type} {name}({args})" + "{\n"
        self._current_func.append(ctx)
        self._needs_free.append([])
        self._needs_cuda_free.append([])

        prev_in_kernel = self._in_kernel
        if self.visit(ctx.returns) in ["kernel", "device"] or type(ctx.returns) == ast.Tuple and self.visit(ctx.returns.elts[0]) == "device":
            self._in_kernel = True

        output += self.visitlist(ctx.body, is_func=True)

        self._in_kernel = prev_in_kernel

        
        self._needs_free.pop(-1)
        self._needs_cuda_free.pop(-1)
        self._current_func.pop(-1)
        output += "}"

        self._variable_scope.pop(-1)
        return  output

if __name__ == "__main__":
    # py2cuda = Py2Cuda("test.py", "sm_86")
    parser = argparse.ArgumentParser(description="Convert Python code to CUDA.")
    parser.add_argument("-f", "--file", default="test.py", help="Python file to convert to CUDA.")
    parser.add_argument("-a", "--arch", default="sm_61", help="Compute Capatability.")

    args = parser.parse_args()

    py2cuda = Py2Cuda(args.file, args.arch)