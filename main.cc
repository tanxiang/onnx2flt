
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

        for (const auto &tensor : model_proto.graph().initializer())
        {
            
        }
    }

    return 0;
}