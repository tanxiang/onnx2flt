#include "nodemap.hh"
#include "noderemap.hh"

auto getNodeLink(flatbuffers::FlatBufferBuilder &flatbuffers,
                 const onnx::NodeProto &node) {
  std::vector<uint32_t> flatInputs{};
  for (const auto &nodeInput : node.input()) {
    // flatInputs.emplace_back(flatbuffers.CreateString(nodeInput));
  }
  std::vector<uint32_t> flatOutputs{};
  for (const auto &nodeOutput : node.output()) {
    // flatOutputs.emplace_back(flatbuffers.CreateString(nodeOutput));
  }
  if (node.has_name() && !node.name().empty())
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateString(node.name()));
  else
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs));
}

auto getNodeLink(flatbuffers::FlatBufferBuilder &flatbuffers,
                 std::vector<const onnx::NodeProto *> &nodes) {
  std::vector<uint32_t> flatInputs{};
  auto startItr = nodes.begin();
  for (auto &nodeInput : (*startItr)->input()) {
    // flatInputs.emplace_back(flatbuffers.CreateString(nodeInput));
  }
  std::vector<uint32_t> flatOutputs{};
  auto endItr = nodes.rbegin();

  for (auto &nodeOutput : (*endItr)->output()) {
    // flatOutputs.emplace_back(flatbuffers.CreateString(nodeOutput));
  }
  if ((*startItr)->has_name() && !(*startItr)->name().empty())
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs),
                          flatbuffers.CreateString((*startItr)->name()));
  else
    return nn::CreateLink(flatbuffers, flatbuffers.CreateVector(flatInputs));
}

template <typename NodeTypeBuilder>
auto getFlNode(flatbuffers::FlatBufferBuilder &flatbuffers,
               std::vector<const onnx::NodeProto *> &nodes,
               mapContext &context) {
  auto flatLink = getNodeLink(flatbuffers, nodes);
  NodeTypeBuilder builder{flatbuffers};
  builder.add_link(flatLink);
  for (const auto &attribute : nodes[0]->attribute()) {

    if constexpr (requires(NodeTypeBuilder & builder, nn::Pads & pads) {
                    builder.add_padding(&pads);
                  }) {
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
        continue;
      }
    }

    if constexpr (requires(NodeTypeBuilder & builder, nn::Stride & strides) {
                    builder.add_stride(&strides);
                  }) {
      if (attribute.name() == "strides" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
          attribute.ints().size() == 2) {
        nn::Stride strides{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        builder.add_stride(&strides);
        continue;
      }
    }

    if constexpr (requires(NodeTypeBuilder & builder, nn::Group & group) {
                    builder.add_group(&group);
                  }) {
      if (attribute.name() == "group" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INT) {
        nn::Group group{
            static_cast<int32_t>(attribute.i()),
        };
        builder.add_group(&group);
        continue;
      }
    }

    if constexpr (requires(NodeTypeBuilder & builder, nn::Dilation & dilation) {
                    builder.add_dilation(&dilation);
                  }) {
      if (attribute.name() == "dilations" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
          attribute.ints().size() == 2) {
        nn::Dilation dilation{
            static_cast<int32_t>(attribute.ints()[0]),
            static_cast<int32_t>(attribute.ints()[1]),
        };
        builder.add_dilation(&dilation);
        continue;
      }
    }
    if constexpr (requires(
                      NodeTypeBuilder & builder,
                      flatbuffers::Offset<flatbuffers::Vector<int32_t>> axes) {
                    builder.add_axes(axes);
                  }) {
      if (attribute.name() == "axes" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS) {
        std::vector<int32_t> axesv(attribute.ints().begin(),
                                   attribute.ints().end());
        builder.add_axes(flatbuffers.CreateVector(axesv));
        continue;
      }
    }

    if constexpr (requires(NodeTypeBuilder & builder, int32_t & axis) {
                    builder.add_axis(axis);
                  }) {
      if (attribute.name() == "axis" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INT) {

        builder.add_axis(static_cast<int32_t>(attribute.i()));
        continue;
      }
    }

    if constexpr (std::same_as<NodeTypeBuilder, nn::FULLY_CONNECTEDBuilder>) {
      if (attribute.name() == "alpha" &&
          attribute.type() == onnx::AttributeProto_AttributeType_FLOAT &&
          attribute.f() == 1.f) {

        // builder.add_bias(attribute.f());
        continue;
      }
      if (attribute.name() == "beta" &&
          attribute.type() == onnx::AttributeProto_AttributeType_FLOAT &&
          attribute.f() == 1.f) {

        // builder.add_bias(attribute.f());
        continue;
      }
      if (attribute.name() == "transB" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INT) {

        // builder.add_bias(attribute.f());
        continue;
      }
    }

    if (attribute.name() == "kernel_shape") {
      continue;
    }
    std::cout << nodeID(*nodes[0]) << " attr " << attribute.DebugString();
  }
  if constexpr (requires(NodeTypeBuilder & builder, nn::FuseCode & fuseCode) {
                  builder.add_fuse_code(&fuseCode);
                }) {
    if (nodes.size() > 1) {
      // std::cout << nodes[1]->DebugString();
      if (nodes[1]->op_type() == "Relu") {
        builder.add_fuse_code(nn::FuseCode::Relu);
      }
      if (nodes[1]->op_type() == "Clip") {
        auto &min = context.tensorMap.at(nodes[1]->input()[1]);
        auto &max = context.tensorMap.at(nodes[1]->input()[2]);
        if (min.data_type() !=
                onnx::TensorProto_DataType::TensorProto_DataType_FLOAT ||
            max.data_type() !=
                onnx::TensorProto_DataType::TensorProto_DataType_FLOAT ||
            min.dims_size() != 1 || max.dims_size() != 1) {
          std::cout << min.DebugString() << max.DebugString();
        } else if (min.float_data()[0] == 0.0f && max.float_data()[0] == 6.0f) {
          builder.add_fuse_code(nn::FuseCode::Relu6);
        } else if (min.float_data()[0] == 0.0f && max.float_data()[0] == 1.0f) {
          builder.add_fuse_code(nn::FuseCode::Relu1);
        } else {
          std::cout << "Clip:" << min.float_data()[0] << '-'
                    << max.float_data()[0] << std::endl;
        }
      }
    }
    if (nodes.size() > 2) {
      std::cout << nodes[2]->DebugString() << "error" << std::endl;
    }
  }
  return builder.Finish();
}
/*
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
  if (nodes.size() > 1) {
    // std::cout << nodes[1]->DebugString();
    if (nodes[1]->op_type() == "Relu") {
      builder.add_fuse_code(nn::FuseCode::FuseCode_Relu);
    }
    if (nodes[1]->op_type() == "Clip") {
      auto &min = context.tensorMap.at(nodes[1]->input()[1]);
      auto &max = context.tensorMap.at(nodes[1]->input()[2]);
      if (min.data_type() !=
              onnx::TensorProto_DataType::TensorProto_DataType_FLOAT ||
          max.data_type() !=
              onnx::TensorProto_DataType::TensorProto_DataType_FLOAT ||
          min.dims_size() != 1 || max.dims_size() != 1) {
        std::cout << min.DebugString() << max.DebugString();
      } else if (min.float_data()[0] == 0.0f && max.float_data()[0] == 6.0f) {
        builder.add_fuse_code(nn::FuseCode::FuseCode_Relu6);
      } else if (min.float_data()[0] == 0.0f && max.float_data()[0] == 1.0f) {
        builder.add_fuse_code(nn::FuseCode::FuseCode_Relu1);
      } else {
        std::cout << "Clip:" << min.float_data()[0] << '-'
                  << max.float_data()[0] << std::endl;
      }
    }
  }
  if (nodes.size() > 2) {
    std::cout << nodes[2]->DebugString() << "error" << std::endl;
  }
  return builder.Finish();
}
*/
template <typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer> &layer) {
  return std::pair<nn::Layer, flatbuffers::Offset<void>>{
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