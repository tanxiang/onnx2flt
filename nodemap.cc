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
  if (node.has_name() && !node.name().empty())
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs),
                          flatbuffers.CreateString(node.name()));
  else
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs));
}

auto getNodeLink(flatbuffers::FlatBufferBuilder &flatbuffers,
                 std::vector<const onnx::NodeProto *> &nodes) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatInputs{};
  auto startItr =nodes.begin();
  for (auto &nodeInput : (*startItr)->input()) {
    flatInputs.emplace_back(flatbuffers.CreateString(nodeInput));
  }
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatOutputs{};
    auto endItr =nodes.rbegin();

  for (auto &nodeOutput : (*endItr)->output()) {
    flatOutputs.emplace_back(flatbuffers.CreateString(nodeOutput));
  }
  if ((*startItr)->has_name() && !(*startItr)->name().empty())
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs),
                          flatbuffers.CreateString((*startItr)->name()));
  else
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateVector(flatOutputs));
}

template <typename NodeTypeBuilder>
auto getFlNode(flatbuffers::FlatBufferBuilder &flatbuffers,
               std::vector<const onnx::NodeProto *> &nodes,
               mapContext &context) {
  auto flatLink = getNodeLink(flatbuffers, nodes);
  NodeTypeBuilder builder{flatbuffers};
  builder.add_link(flatLink);
  return builder.Finish();
}

template <>
auto getFlNode<nn::CONV_2DBuilder>(flatbuffers::FlatBufferBuilder &flatbuffers,
                                   std::vector<const onnx::NodeProto *> &nodes,
                                   mapContext &context) {

  auto flatLink = getNodeLink(flatbuffers, nodes);
  nn::CONV_2DBuilder builder{flatbuffers};
  builder.add_link(flatLink);
  bool getKernelShapr = false;
  nn::KernelShape kernelShaper;
  for (const auto &attribute : nodes[0]->attribute()) {
    if (attribute.has_name()) {
      // std::cout<<attribute.DebugString()<<std::endl;
      if (attribute.name() == "pads" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
          attribute.ints().size() == 4) {
        nn::Pads pads{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
            static_cast<int32_t>(attribute.ints()[2]),
            static_cast<int32_t>(attribute.ints()[3]),
        };
        builder.add_padding(&pads);
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
        nn::Stride strides{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        builder.add_stride(&strides);
      } else if (attribute.name() == "dilations" &&
                 attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
                 attribute.ints().size() == 2) {
        nn::Dilation dilation{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        builder.add_dilation(&dilation);
      } else if (attribute.name() == "group" &&
                 attribute.type() == onnx::AttributeProto_AttributeType_INT) {
        nn::Group group{
            static_cast<int32_t>(attribute.i()),
        };
        builder.add_group(&group);
      } else {
        std::cout << "node " << nodes[0]->op_type() << " attribute "
                  << attribute.DebugString() << "!\n";
      }
    }
  }
  if (getKernelShapr) {
    auto &tensor = context.tensorMap.at(nodes[0]->input()[1]);
    if (kernelShaper.width() != tensor.dims()[2] ||
        kernelShaper.height() != tensor.dims()[3]) {
      std::cout << "check tensor dim != KernelShapr\n"
                << tensor.dims_size() << '\n'
                << tensor.dims()[2] << " x : y " << tensor.dims()[3] << '\n'
                << kernelShaper.width() << " x : y " << kernelShaper.height()
                << '\n';
    }
  }
  return builder.Finish();
}

template <typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer> &layer) {
  return std::pair<uint8_t, flatbuffers::Offset<void>>{
      nn::LayerTraits<Layer>::enum_value, layer.Union()};
}

OpToFuncMap::OpToFuncMap() {
  emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                     std::vector<const onnx::NodeProto *> &nodes,
                     mapContext &context) {
    auto flNode = getFlNode<nn::CONV_2DBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                     std::vector<const onnx::NodeProto *> &nodes,
                     mapContext &context) {
    auto flNode = getFlNode<nn::RELUBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       std::vector<const onnx::NodeProto *> &nodes,
                       mapContext &context) {
    auto flNode =
        getFlNode<nn::CONCATENATIONBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("MaxPool",
          [](flatbuffers::FlatBufferBuilder &flatbuffers,
             std::vector<const onnx::NodeProto *> &nodes, mapContext &context) {
            auto flNode =
                getFlNode<nn::MAX_POOL_2DBuilder>(flatbuffers, nodes, context);
            return UnionPair(flNode);
          });

  emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                        std::vector<const onnx::NodeProto *> &nodes,
                        mapContext &context) {
    auto flNode = getFlNode<nn::SOFTMAXBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                    std::vector<const onnx::NodeProto *> &nodes,
                    mapContext &context) {
    auto flNode = getFlNode<nn::ADDBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("GlobalAveragePool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                                  std::vector<const onnx::NodeProto *> &nodes,
                                  mapContext &context) {
    auto flNode =
        getFlNode<nn::AVERAGE_POOL_2DBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                     std::vector<const onnx::NodeProto *> &nodes,
                     mapContext &context) {
    auto flNode = getFlNode<nn::RELUBuilder>(flatbuffers, nodes,
                                             context); // clip [-1,1] [0,6]
    return UnionPair(flNode);
  });

  emplace("Unsqueeze", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          std::vector<const onnx::NodeProto *> &nodes,
                          mapContext &context) {
    auto flNode = getFlNode<nn::RESHAPEBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                        std::vector<const onnx::NodeProto *> &nodes,
                        mapContext &context) {
    auto flNode = getFlNode<nn::RESHAPEBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       std::vector<const onnx::NodeProto *> &nodes,
                       mapContext &context) {
    auto flNode = getFlNode<nn::GATHERBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });

  emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                     std::vector<const onnx::NodeProto *> &nodes,
                     mapContext &context) {
    auto flNode =
        getFlNode<nn::FULLY_CONNECTEDBuilder>(flatbuffers, nodes, context);
    return UnionPair(flNode);
  });
}