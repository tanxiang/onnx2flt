
#include "context.hh"
#include "nodemap.hh"
#include "noderemap.hh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void usage(const std::string &filename) {
  std::cout << "Usage: " << filename
            << " onnx_model output_filename [table_file]" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 4) {
    usage(argv[0]);
    return -1;
  }

  onnx::ModelProto model_proto;
  std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
  std::stringstream ss;
  ss << ifs.rdbuf();
  // FIXME: Handle the return value
  model_proto.ParseFromString(ss.str());
  std::cout << "model_proto.ir_version = " << model_proto.ir_version()
            << std::endl;


  static OpToFuncMap mapOpFunc;

  mapContext context;

  if (model_proto.has_graph()) {
    flatbuffers::FlatBufferBuilder flatbuffers;
    auto graph = model_proto.graph();

    for (const auto &tensor : model_proto.graph().initializer()) {
      if (tensor.has_name()) {
        context.tensorMap.emplace(tensor.name(), tensor);
      }

      switch (tensor.data_type()) {
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
        std::cerr << "onnx::TensorProto_DataType " << tensor.data_type()
                  << " not support\n";
        return -1;
      };
    }

    for (const auto &tensor : model_proto.graph().sparse_initializer()) {
      std::cout << "graph().sparse_initializer() " << &tensor;

      //     if (tensor.has_name()) {
      // context.sparseTensorMap.emplace(tensor.name(), tensor);
      //}
    }
    std::cout << "model_proto.graph().node_size() = "
              << model_proto.graph().node_size() << std::endl;
    for (const auto &node : model_proto.graph().node()) {
      if (node.has_name() && !node.name().empty()) {
        if (!context.nodeMap.emplace(node.name(), node).second) {
          std::cerr << "context.nodeMap.emplace error same name : "
                    << node.name() << " not support\n";
        }
      }
      for (const auto &input : node.input()) {
        context.inputNodeMap.emplace(input, node);
      }
      for (const auto &output : node.output()) {
        if (!context.outputNodeMap.emplace(output, node).second) {
          std::cerr << "context.outputNodeMap.emplace error same output : "
                    << output << " in " << &node << " not support\n";
        }
      }
    }
    std::vector<std::vector<const onnx::NodeProto *>> vvRemap;
    for (const auto &input : graph.input()) {
      if (input.has_name()) {
        std::cout << "input.name() " << input.name() << '\n';
        vvRemap = createNodeVVFromInput(input, context);
        std::cout << "vvRemap size = " << vvRemap.size() << std::endl;
        int allSize{0};
        for (auto &vRemap : vvRemap) {
          allSize += vRemap.size();
        }
        std::cout << "vvRemap allSize = " << allSize << std::endl;

      } else
        std::cout << "noname input graph\n";
    }
    int conventNodeNum{0};
    for (const auto &node : model_proto.graph().node()) {
      bool nodeRemapd = false;
      for (auto &vRemap : vvRemap) {
        for (auto &Remap : vRemap) {
          if (&node == Remap) {
            nodeRemapd = true;
            break;
          }
        }
        if (nodeRemapd)
          break;
      }
      if (!nodeRemapd) {
        ++conventNodeNum;
        std::cout << nodeID(node) << " need to tensor:" << conventNodeNum
                  << '\t';
        for (auto &input : node.input()) {
          if (context.tensorMap.contains(input))
            std::cout << '<' << input << "> ";
          else if (context.outputNodeMap.contains(input))
            std::cout << '[' << nodeID(context.outputNodeMap.at(input)) << "] ";
          else
            std::cout << '{' << input << "} ";
        }
        std::cout << std::endl;
      }
    }

    std::vector<uint8_t> nodeTypes;
    std::vector<flatbuffers::Offset<void>> nodeVals;

    for (auto &vRemap :vvRemap) {

      auto opItr = mapOpFunc.find(vRemap[0]->op_type());
      if (opItr != mapOpFunc.end()) {
        auto ftnode = opItr->second(flatbuffers, vRemap, context);
        nodeTypes.emplace_back(ftnode.first);
        nodeVals.emplace_back(ftnode.second);
      } else {
        std::cerr << "error: " << vRemap[0]->op_type() << " is not support!\n";
      }
    }

    nn::versionInfo version{FLATBUFFERS_VERSION_MAJOR * 10000 +
                                FLATBUFFERS_VERSION_MINOR * 100 +
                                FLATBUFFERS_VERSION_REVISION,
                            model_proto.ir_version()};
    auto flatbuffersGraph = nn::CreateGraph(flatbuffers, &version,
                                            flatbuffers.CreateVector(nodeTypes),
                                            flatbuffers.CreateVector(nodeVals));
    flatbuffers.Finish(flatbuffersGraph);
    std::ofstream outputfile{argv[2]};
    outputfile.write(reinterpret_cast<char *>(flatbuffers.GetBufferPointer()),
                     flatbuffers.GetSize());
  }

  return 0;
}