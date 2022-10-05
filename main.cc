
#include "fbs/gnt_generated.h"
#include <fstream>
#include <iostream>
#include <map>
#include <onnx/onnx_pb.h>
#include <sstream>
#include <string>
#include <vector>

struct mapContext {
  std::map<std::string, const onnx::TensorProto &> tensorMap;
  // std::map<std::string, const onnx::SparseTensorProto &> sparseTensorMap;
  std::map<std::string, const onnx::NodeProto &> nodeMap;
  std::map<std::string, const onnx::NodeProto &> outputNodeMap;
};

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

auto kernelShapeMarchTensor(std::string name, mapContext &context,
                            nn::KernelShape &kernelShaper) {}

auto getConvNode(flatbuffers::FlatBufferBuilder &flatbuffers,
                 const onnx::NodeProto &node, mapContext &context) {
  auto flatLink = getNodeLink(flatbuffers, node);
  // std::cout<<node.op_type()<<" attribute_size() :
  // "<<node.attribute_size()<<std::endl;
  bool getStrides = false, getPads = false, getKernelShapr = false,
       getDilation = false;
  nn::Pads pads;
  nn::Stride strides;
  nn::KernelShape kernelShaper;
  nn::Dilation dilation;

  for (const auto &attribute : node.attribute()) {
    if (attribute.has_name()) {
      // std::cout<<attribute.DebugString()<<std::endl;
      if (attribute.name() == "pads" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
          attribute.ints().size() == 4) {
        pads = {
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
            static_cast<int32_t>(attribute.ints()[2]),
            static_cast<int32_t>(attribute.ints()[3]),
        };
        getPads = true;
      } else if (attribute.name() == "kernel_shape" &&
                 attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
                 attribute.ints().size() == 2) {
        kernelShaper = {
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        getKernelShapr = true;
      } else if (attribute.name() == "strides" &&
                 attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
                 attribute.ints().size() == 2) {
        strides = {
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        getStrides = true;
      } else if (attribute.name() == "dilations" &&
                 attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
                 attribute.ints().size() == 2) {
        dilation = {
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        getDilation = true;
      } else {
        std::cout<<"node "<<node.op_type()<<" attribute "<<attribute.name()<<" unsupport!\n";
      }
    }
  }
  if (getKernelShapr) {
    auto &tensor = context.tensorMap.at(node.input()[1]);
    if (kernelShaper.width() != tensor.dims()[2] ||
        kernelShaper.height() != tensor.dims()[3]) {
      std::cout << "check tensor dim != KernelShapr\n"
                << tensor.dims_size() << '\n'
                << tensor.dims()[2] << " x : y " << tensor.dims()[3] << '\n'
                << kernelShaper.width() << " x : y " << kernelShaper.height()
                << '\n';
    }
  }
  if (getStrides && getPads) {
    if (getDilation)
      return nn::CreateCONV_2D(flatbuffers, flatLink, &pads, &strides,
                               &dilation);
    else
      return nn::CreateCONV_2D(flatbuffers, flatLink, &pads, &strides);
  }

  std::cerr << "no comp node \n " << node.DebugString() << std::endl;
  return nn::CreateCONV_2D(flatbuffers, flatLink);
}

template<typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer>& layer){
    return std::pair<uint8_t,flatbuffers::Offset<void>>{nn::LayerTraits<Layer>::enum_value,layer.Union()};
}

struct OpToFuncMap
    : public std::map<
          std::string,
          std::function<std::pair<uint8_t,flatbuffers::Offset<void>>(flatbuffers::FlatBufferBuilder &,
                             const onnx::NodeProto &, mapContext &context)>> {
  OpToFuncMap() {

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getConvNode(flatbuffers, node, context);
      return UnionPair(flNode);
    });
/*
    emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &, mapContext &context) { ; });

    emplace("MaxPool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Dropout", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &, mapContext &context) { ; });

    emplace("GlobalAveragePool",
            [](flatbuffers::FlatBufferBuilder &flatbuffers,
               const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Shape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                        const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Unsqueeze",
            [](flatbuffers::FlatBufferBuilder &flatbuffers,
               const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Constant", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                           const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &, mapContext &context) { ; });

    emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &, mapContext &context) { ; });
                       */
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

  mapContext context;

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
    for (const auto &node : model_proto.graph().node()) {
      if (node.has_name()) {
        context.nodeMap.emplace(node.name(), node);
      }
      for (const auto &output : node.output()) {
        context.outputNodeMap.emplace(output, node);
      }
    }
      std::vector<uint8_t> nodeTypes;
        std::vector<flatbuffers::Offset<void>> nodeVals;


    for (const auto &node : model_proto.graph().node()) {
      // auto flatLink = getNodeLink(node, flatbuffers);
      auto opItr = mapOpFunc.find(node.op_type());
      if (opItr != mapOpFunc.end()) {
        auto ftnode = opItr->second(flatbuffers, node, context);
        nodeTypes.emplace_back(ftnode.first);
        nodeVals.emplace_back(ftnode.second);
      } else {
        std::cerr << "error: " << node.op_type() << " is not support!\n";
      }
    }

    nn::versionInfo version{FLATBUFFERS_VERSION_MAJOR * 10000 +
                                FLATBUFFERS_VERSION_MINOR * 100 +
                                FLATBUFFERS_VERSION_REVISION,
                            model_proto.ir_version()};
    auto flatbuffersGraph = nn::CreateGraph(flatbuffers, &version,flatbuffers.CreateVector(nodeTypes),flatbuffers.CreateVector(nodeVals));
    flatbuffers.Finish(flatbuffersGraph);
    std::ofstream outputfile{argv[2]};
    outputfile.write(reinterpret_cast<char *>(flatbuffers.GetBufferPointer()),
                     flatbuffers.GetSize());
  }

  return 0;
}