
#include <iostream>
#include <fstream>
#include <onnx/onnx_pb.h>
#include <string>
#include <map>
#include <sstream>
#include "fbs/gnt_generated.h"

void usage(const std::string &filename)
{
    std::cout << "Usage: " << filename
              << " onnx_model output_filename [table_file]" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4)
    {
        usage(argv[0]);
        return -1;
    }
    const std::string table_file = argc == 4 ? argv[3] : "";

    onnx::ModelProto model_proto;
    std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
    std::stringstream ss;
    ss << ifs.rdbuf();
    // FIXME: Handle the return value
    model_proto.ParseFromString(ss.str());
    std::cout << "model_proto.ir_version = " << model_proto.ir_version() << std::endl;

    if (model_proto.has_graph())
    {
        auto graph = model_proto.graph();
        for (const auto &input : graph.input())
        {
            if (input.has_name())
                std::cout << "input.name() " << input.name() << '\n';
            else
                std::cout << "noname input graph\n";
        }

        for (const auto &tensor : model_proto.graph().initializer())
        {
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT:
                break;
            case onnx::TensorProto_DataType_FLOAT16:
                break;
            case onnx::TensorProto_DataType_INT32:
                break;
            case onnx::TensorProto_DataType_INT16:
                break;
            case onnx::TensorProto_DataType_INT8:
                break;
            case onnx::TensorProto_DataType_INT64:
                break;
            default:
                std::cout << "onnx::TensorProto_DataType " << tensor.data_type() << " not support\n";
                return -1;
            };
        }
        for (const auto &node : model_proto.graph().node())
        {
            if (node.has_name())
                std::cout << "node.name() = " << node.name() << '\n';
            else
                std::cout << &node << " no name\n";
            node.has_op_type();
            node.op_type();
        }
    }

    return 0;
}