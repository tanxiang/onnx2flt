#include "nodemap.hh"


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
        std::cout << "node " << node.op_type() << " attribute "
                  << attribute.name() << " unsupport!\n";
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

template <typename NodeTypeBuilder>
auto getFlNode(flatbuffers::FlatBufferBuilder &flatbuffers,
               const onnx::NodeProto &node, mapContext &context) {
  auto flatLink = getNodeLink(flatbuffers, node);
  NodeTypeBuilder builder{flatbuffers};
  builder.add_link(flatLink);
  return builder.Finish();
}

template <typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer> &layer) {
  return std::pair<uint8_t, flatbuffers::Offset<void>>{
      nn::LayerTraits<Layer>::enum_value, layer.Union()};
}

OpToFuncMap::OpToFuncMap() {

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getConvNode(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::RELUBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &node, mapContext &context) {
      auto flNode =
          getFlNode<nn::CONCATENATIONBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("MaxPool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &node, mapContext &context) {
      auto flNode =
          getFlNode<nn::MAX_POOL_2DBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::SOFTMAXBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::ADDBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("GlobalAveragePool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                                    const onnx::NodeProto &node,
                                    mapContext &context) {
      auto flNode =
          getFlNode<nn::AVERAGE_POOL_2DBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::RELUBuilder>(flatbuffers, node,
                                               context); // clip [-1,1] [0,6]
      return UnionPair(flNode);
    });

    emplace("Unsqueeze", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                            const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::RESHAPEBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::RESHAPEBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         const onnx::NodeProto &node, mapContext &context) {
      auto flNode = getFlNode<nn::GATHERBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });

    emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       const onnx::NodeProto &node, mapContext &context) {
      auto flNode =
          getFlNode<nn::FULLY_CONNECTEDBuilder>(flatbuffers, node, context);
      return UnionPair(flNode);
    });
  }