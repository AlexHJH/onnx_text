from typing import OrderedDict
import onnx
import re
import os
import numpy as np
import onnx.onnx_ml_pb2 as onnx_ml_pb2
import onnx.parser
import pickle
from io import BytesIO

MODEL_PROFILE_FORMAT = \
    '''
<
    ir_version: {ir_version},
    opset_import: {opset_import}
>
'''

GRAPH_PROFILE_FORMAT = \
    '''
{graph_name} ({input_info}) => ({output_info})
<

>
'''

def _format_str(ori_str):
    number_str = [str(i) for i in range(0, 10)]
    if ori_str[0] in number_str:
        ori_str = "n{}".format(ori_str)
    
    return re.sub(r'[\.\{\}\[\]\<\>\-]', "_", ori_str)

def _tensor_type_to_str(tensor_type, name):
    declare_dtype = onnx.mapping.STORAGE_TENSOR_TYPE_TO_FIELD[tensor_type.elem_type].split("_")[0]
    declare_shape = [di.dim_param if di.dim_value < 0 else str(di.dim_value) for di in tensor_type.shape.dim]

    declare_shape = "" if len(declare_shape) < 1 else "[{}]".format(", ".join(declare_shape))

    return "{}{} {}".format(declare_dtype, declare_shape, name)

def _tensor_to_str(tensor):
    tensor_bytes = BytesIO()
    declare_dtype = onnx.mapping.STORAGE_TENSOR_TYPE_TO_FIELD[tensor.data_type].split("_")[0]

    declare_shape = [str(di) for di in tensor.dims]
    declare_shape = "" if len(declare_shape) < 1 else "[{}]".format(", ".join(declare_shape))
    
    tensor_data = np.reshape(np.frombuffer(
            tensor.raw_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]), tensor.dims)
        
    tensor_data_str = str(tensor_data.flatten())

    if tensor_data_str[0] == "[":
        tensor_data_str = tensor_data_str[1:-1]

    tensor_bytes.write("{}{} = ".format(declare_dtype, declare_shape).encode('utf-8'))
    tensor_bytes.write("{".encode('utf-8'))
    tensor_bytes.write(tensor_data_str.encode('utf-8'))
    tensor_bytes.write("}".encode('utf-8'))
    return tensor_bytes.getvalue().decode('utf-8')

ATTRIBUTEPROTO_TO_STR_FUNC = {
    onnx_ml_pb2.AttributeProto.INT: ["i", str],
    onnx_ml_pb2.AttributeProto.INTS: ["ints", lambda x : "[{}]".format(", ".join([str(xi) for xi in x]))],

    onnx_ml_pb2.AttributeProto.FLOAT: ["f", str],
    onnx_ml_pb2.AttributeProto.FLOATS: ["floats", lambda x : "[{}]".format(", ".join([str(xi) for xi in x]))],

    onnx_ml_pb2.AttributeProto.STRING: ["s", str],
    onnx_ml_pb2.AttributeProto.STRINGS: ["strings", lambda x : "[{}]".format(", ".join(['"{}"'.format(xi) for xi in x]))],

    onnx_ml_pb2.AttributeProto.TENSOR: ["t", _tensor_to_str],
    onnx_ml_pb2.AttributeProto.TENSORS: ["tensors", lambda x : "[{}]".format(", ".join([_tensor_to_str(xi) for xi in x]))],
}

def _node_to_str(node, bytes_io): 
    input_info_str  = ", ".join([_format_str(i) for i in node.input])
    output_info_str  = ", ".join([_format_str(i) for i in node.output])

    bytes_io.write("    {} = {}".format(output_info_str, node.op_type).encode('utf-8'))

    if len(node.attribute) < 1:
        pass
        #bytes_io.write("({})\n\n".format(output_info_str).encode('utf-8'))
    else:
        bytes_io.write("<\n".encode('utf-8'))
        for i, attri in enumerate(node.attribute):
            attr_func = ATTRIBUTEPROTO_TO_STR_FUNC.get(attri.type, None)

            if attr_func is None:
                raise TypeError

            bytes_io.write("        {} = {}".format(
                attri.name,
                attr_func[1](attri.__getattribute__(attr_func[0]))
            ).encode('utf-8'))

            if i == len(node.attribute) - 1:
                bytes_io.write("\n        >".encode('utf-8'))
            else:
                bytes_io.write(",\n".encode('utf-8'))

    bytes_io.write("({})\n\n".format(input_info_str).encode('utf-8'))



def onnx_model_textual(onnx_model):
    text_model_fp = BytesIO()

    opset_import_str = []
    for item in onnx_model.opset_import:
        opset_import_str.append(
            '"{}": {}'.format(item.domain, str(item.version))
        )
    opset_import_str = "[{}]".format(", ".join(opset_import_str))

    text_model_fp.write(
        MODEL_PROFILE_FORMAT.format(
            ir_version=int(onnx_model.ir_version),
            opset_import=opset_import_str,
        ).encode('utf-8')
    )

    graph_input_dict = OrderedDict()
    graph_init_dict = OrderedDict()
    graph_output_dict = OrderedDict()

    for item in onnx_model.graph.input:
        graph_input_dict[_format_str(item.name)] = item.type.tensor_type

    for item in onnx_model.graph.output:
        graph_output_dict[_format_str(item.name)] = item.type.tensor_type

    for item in onnx_model.graph.initializer:
        graph_init_dict[_format_str(item.name)] = np.reshape(np.frombuffer(
            item.raw_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[item.data_type]), item.dims)

    value_input_names = set(graph_input_dict.keys()) - \
        set(graph_init_dict.keys())

    input_info_str = []
    output_info_str = []
    for k, v in graph_input_dict.items():
        if k in value_input_names:
            input_info_str.append(
                _tensor_type_to_str(v, k)
            )
    for k, v in graph_output_dict.items():
        output_info_str.append(
            _tensor_type_to_str(v, k)
        )
    input_info_str = ", ".join(input_info_str)
    output_info_str = ", ".join(output_info_str)

    text_model_fp.write(
        GRAPH_PROFILE_FORMAT.format(
            graph_name=_format_str(onnx_model.graph.name),
            input_info=input_info_str,
            output_info=output_info_str,
        ).encode('utf-8')
    )

    text_model_fp.write("{\n".encode('utf-8'))

    for node_i in onnx_model.graph.node:
        _node_to_str(node_i, text_model_fp)

    text_model_fp.write("}\n".encode('utf-8'))

    return text_model_fp.getvalue().decode('utf-8'), graph_init_dict


if __name__ == '__main__':
    onnx_model_path = "mobile_centernet_v3_large.onnx"

    onnx_model = onnx.load_model(onnx_model_path)

    onnx_model_txt, graph_init_dict = onnx_model_textual(onnx_model)

    with open("deploy.onnxtxt", 'w') as fp:
        fp.write(onnx_model_txt)
        fp.close()
    
    with open("deploy.pkl", 'wb') as fp:
        pickle.dump(graph_init_dict, fp)
        fp.close()

    onnx_model_new = onnx.parser.parse_model(onnx_model_txt)

    print(onnx_model_new)
