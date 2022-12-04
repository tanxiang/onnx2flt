
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

    for (const auto &tensor : graph.initializer()) {
      if (tensor.has_name()) {
        context.tensorMap.emplace(tensor.name(), tensor);
      }
    }

    for (const auto &tensor : graph.sparse_initializer()) {
      std::cout << "graph().sparse_initializer() " << &tensor;

      //     if (tensor.has_name()) {
      // context.sparseTensorMap.emplace(tensor.name(), tensor);
      //}
    }
    std::cout << "model_proto.graph().node_size() = " << graph.node_size()
              << std::endl;
    for (const auto &node : graph.node()) {
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
      if (node.output_size() > 1) {
        std::cerr << nodeID(node) << node.DebugString();
      }
    }

    for (auto &input : graph.input()) {
      context.graphsInputs.emplace(input.name(), input);
    }

    std::vector<std::string> outputs;
    for (const auto &output : graph.output()) {
      if (output.has_name() && !output.name().empty())
        outputs.emplace_back(output.name());

      std::cout << "graphs output:" << output.DebugString() << std::endl;
    }

    std::map<std::string, int> symbols;
    auto nodesMap =
        writeFlNodeFromOutputs(flatbuffers, outputs, context, symbols);

    nn::versionInfo version{FLATBUFFERS_VERSION_MAJOR * 10000 +
                                FLATBUFFERS_VERSION_MINOR * 100 +
                                FLATBUFFERS_VERSION_REVISION,
                            model_proto.ir_version()};
    auto flatbuffersGraph = nn::CreateGraph(
        flatbuffers, &version,
        flatbuffers.CreateVector(
            nodesMap.size(), std::function<nn::Layer(size_t)>{[&](size_t i) {
              std::cout << "get type:" << nodesMap.size() << " at " << i
                        << std::endl;
              return nodesMap.find(i)->type;
            }}),
        flatbuffers.CreateVector(
            nodesMap.size(),
            std::function<flatbuffers::Offset<void>(size_t i)>{[&](size_t i) {
              std::cout << "get data:" << i << std::endl;
              return nodesMap.find(i)->data;
            }}),
        flatbuffers.CreateVector(
            graph.output_size(),
            std::function<flatbuffers::Offset<nn::Output>(size_t i)>{[&](size_t i) {
              return writeFlOutputs(flatbuffers, graph.output(i), symbols);
            }}));

    flatbuffers.Finish(flatbuffersGraph);
    {

      std::ofstream outputfile{argv[2]};
      outputfile.write(reinterpret_cast<char *>(flatbuffers.GetBufferPointer()),
                       flatbuffers.GetSize());
    }
  }

  return 0;
}