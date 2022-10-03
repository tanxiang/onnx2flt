
#include "fbs/gnt_generated.h"
#include <fstream>
#include <iostream>
#include <map>
#include <onnx/onnx_pb.h>
#include <sstream>
#include <string>
#include <vector>

void usage(const std::string &filename) {
  std::cout << "Usage: " << filename
            << " onnx_model output_filename [table_file]" << std::endl;
}

auto getNodeLink(flatbuffers::FlatBufferBuilder &flatbuffers,
                 const onnx::NodeProto &node) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatInputs{};
  for (const auto &nodeInput : node.input()) {
    flatInputs.emplace_back(flatbuffers.CreateString(nodeInput));
  }
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatOutputs{};
  for (const auto &nodeOutput : node.output()) {
    flatOutputs.emplace_back(flatbuffers.CreateString(nodeOutput));
  }
  if (node.has_name())
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs),
                          flatbuffers.CreateString(node.name()));
  else
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs));
}

auto getConvNode(flatbuffers::FlatBufferBuilder &flatbuffers,
                 const onnx::NodeProto &node) {
  auto flatLink = getNodeLink(flatbuffers, node);
  // std::cout<<node.op_type()<<" attribute_size() :
  // "<<node.attribute_size()<<std::endl;
  bool getStrides = false, getPads = false, getKernelShapr = false;
  for (const auto &attribute : node.attribute()) {
    if (attribute.has_name()) {
      // std::cout<<attribute.DebugString()<<std::endl;
      if (attribute.name() == "pads" && attribute.type() == onnx::AttributeProto_AttributeType_INTS) {
        nn::Pads pads{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
            static_cast<int32_t>(attribute.ints()[2]),
            static_cast<int32_t>(attribute.ints()[3]),
            };
        getPads = true;
      } else if (attribute.name() == "kernel_shape"  && attribute.type() == onnx::AttributeProto_AttributeType_INTS) {
        nn::KernelShape kernelShaper{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        getKernelShapr = true;
      } else if (attribute.name() == "strides" && attribute.type() == onnx::AttributeProto_AttributeType_INTS) {
        nn::Stride strides{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        getStrides = true;
      }
    }
  }
  if (!(getStrides && getPads && getKernelShapr)) {
    std::cout << "no comp node \n " << node.DebugString() << std::endl;
  }
}

struct OpToFuncMap
    : public std::map<std::string,
                      std::function<void(flatbuffers::FlatBufferBuilder &,
                                         const onnx::NodeProto &)>> {
  OpToFuncMap() {

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node) {
      getConvNode(flatbuffers, node);
    });

    emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &) { ; });

    emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &) { ; });

    emplace("MaxPool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &) { ; });

    emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &) { ; });

    emplace("Dropout", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &) { ; });

    emplace("GlobalAveragePool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                                    const onnx::NodeProto &) { ; });

    emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &) { ; });

    emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      const onnx::NodeProto &) { ; });

    emplace("Shape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                        const onnx::NodeProto &) { ; });

    emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &) { ; });

    emplace("Unsqueeze", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                            const onnx::NodeProto &) { ; });

    emplace("Constant", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                           const onnx::NodeProto &) { ; });

    emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &) { ; });

    emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &) { ; });
  }
};

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

  if (model_proto.has_graph()) {
    flatbuffers::FlatBufferBuilder flatbuffers;
    auto graph = model_proto.graph();
    for (const auto &input : graph.input()) {
      if (input.has_name())
        std::cout << "input.name() " << input.name() << '\n';
      else
        std::cout << "noname input graph\n";
    }

    for (const auto &tensor : model_proto.graph().initializer()) {
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
    }
    for (const auto &node : model_proto.graph().node()) {
      // auto flatLink = getNodeLink(node, flatbuffers);
      auto opItr = mapOpFunc.find(node.op_type());
      if (opItr != mapOpFunc.end()) {
        opItr->second.operator()(flatbuffers, node);
      } else {
        std::cerr << "error: " << node.op_type() << " is not support!\n";
      }
    }

    nn::versionInfo version{FLATBUFFERS_VERSION_MAJOR * 10000 +
                                FLATBUFFERS_VERSION_MINOR * 100 +
                                FLATBUFFERS_VERSION_REVISION,
                            model_proto.ir_version()};
    auto flatbuffersGraph = nn::CreateGraph(flatbuffers, &version);
    flatbuffers.Finish(flatbuffersGraph);
    std::ofstream outputfile{argv[2]};
    outputfile.write(reinterpret_cast<char *>(flatbuffers.GetBufferPointer()),
                     flatbuffers.GetSize());
  }

  return 0;
}