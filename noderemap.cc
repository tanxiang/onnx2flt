#include "noderemap.hh"
#include <queue>
#include <sstream>

std::string nodeID(const onnx::NodeProto &node) {
  std::ostringstream id{};

  if (node.has_name() && !node.name().empty())
    id << node.name() << ' ' << &node;
  else
    id << node.op_type() << ' ' << &node;
  return id.str();
}

std::string tensorID(const onnx::TensorProto &tensor) {

  std::ostringstream id{};

  if (tensor.has_name() && !tensor.name().empty())
    id << tensor.name() << ' ' << &tensor;
  else
    id << "tensor " << &tensor;
  return id.str();
}

int32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::set<kLayerData, std::less<>> &nodesData,
                     mapContext &context, const std::string output,
                     std::map<std::string, int> &symbols);

template <typename NodeTypeBuilder>
auto writeFlNodeFinish(flatbuffers::FlatBufferBuilder &flBuilder,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node) {
  NodeTypeBuilder builder{flBuilder};
  if constexpr (requires(NodeTypeBuilder & builder,
                         flatbuffers::Offset<nn::Link> link) {
                  builder.add_link(link);
                }) {
    builder.add_link(link);
  }

  for (const auto &attribute : node.attribute()) {

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
        builder.add_axes(builder.fbb_.CreateVector(axesv));
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
    std::cout << nodeID(node) << " attr " << attribute.DebugString();
  }

  return builder.Finish();
}

template <typename Layer>
inline auto UnionType(flatbuffers::Offset<Layer> layer) {
  return nn::LayerTraits<Layer>::enum_value;
}

template <typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer> layer) {
  return std::pair<nn::Layer, flatbuffers::Offset<void>>{UnionType(layer),
                                                         layer.Union()};
}

int32_t writeFlFuseNode(flatbuffers::FlatBufferBuilder &builder,
                         std::set<kLayerData, std::less<>> &nodesData,
                         const nn::FuseCode code,
                         std::map<std::string, int> &symbols) {
  std::string output = nn::EnumNameFuseCode(code);
  output += "__helper_node__";
  if (auto itr = symbols.find(output); itr != symbols.end()) {
    return itr->second;
  }

  auto tensorIndex = symbols.size();
  symbols.emplace(output, tensorIndex);
  nodesData.emplace(tensorIndex, nn::Layer::FuseNode,
                    builder.CreateStruct<nn::FuseNode>(code).Union());

  return tensorIndex;
}

template <typename ScalaType>
int32_t writeFlScalaNode(flatbuffers::FlatBufferBuilder &builder,
                         std::set<kLayerData, std::less<>> &nodesData,
                         const ScalaType val,
                         std::map<std::string, int> &symbols) {
  std::string output = std::to_string(val.data());
  output += "__helper_node__";
  output += nn::EnumNameLayer(nn::LayerTraits<ScalaType>::enum_value);
  if (auto itr = symbols.find(output); itr != symbols.end()) {
    return itr->second;
  }

  auto tensorIndex = symbols.size();
  symbols.emplace(output, tensorIndex);
  nodesData.emplace(tensorIndex, nn::LayerTraits<ScalaType>::enum_value,
                    builder.CreateStruct<ScalaType>(val).Union());

  return tensorIndex;
}

template <typename NodeTypeBuilder>
auto writeFlNodeFinish2(flatbuffers::FlatBufferBuilder &flatbuffers,
                        mapContext &context, const nn::FuseCode fuseCode,
                        std::string output, const onnx::NodeProto &node,
                        std::set<kLayerData, std::less<>> &nodesData,
                        std::map<std::string, int> &symbols) {
  NodeTypeBuilder builder{flatbuffers};
  if constexpr (requires(NodeTypeBuilder & builder,
                         flatbuffers::Offset<nn::Link> link) {
                  builder.add_link(link);
                }) {
    builder.add_link(nn::CreateLink(
        flatbuffers,
        flatbuffers.CreateVector(
            node.input_size(), std::function<int32_t(size_t)>{[&](size_t i) {
              return writeFlNode(flatbuffers, nodesData, context,
                                 node.input()[i], symbols);
            }})));
  }

  if constexpr (requires(NodeTypeBuilder & builder, int32_t fuseNode) {
                  builder.add_fuse_node(fuseNode);
                }) {
                    builder.add_fuse_node(writeFlFuseNode(flatbuffers, nodesData,fuseCode,symbols));
  }

  for (const auto &attribute : node.attribute()) {

    if constexpr (requires(NodeTypeBuilder & builder, nn::Pads & pads) {
                    builder.add_padding(&pads);
                  }) {
      if (attribute.name() == "pads" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INTS &&
          attribute.ints().size() == 4) {
        nn::Pads pads{
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[0])},
                symbols),
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[1])},
                symbols),
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[2])},
                symbols),
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[3])},
                symbols),
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
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[0])},
                symbols),
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[1])},
                symbols),
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
            writeFlScalaNode(flatbuffers, nodesData,
                             nn::I32Scalar{static_cast<int32_t>(attribute.i())},
                             symbols),
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
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[0])},
                symbols),
            writeFlScalaNode(
                flatbuffers, nodesData,
                nn::I32Scalar{static_cast<int32_t>(attribute.ints()[1])},
                symbols),
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
        builder.add_axes(builder.fbb_.CreateVector(
            attribute.ints().size(),
            std::function<int32_t(size_t i)>{[&](size_t i) {
              return writeFlScalaNode(
                  flatbuffers, nodesData,
                  nn::I32Scalar{static_cast<int32_t>(attribute.ints()[i])},
                  symbols);
            }}));
        continue;
      }
    }

    if constexpr (requires(NodeTypeBuilder & builder, int32_t & axis) {
                    builder.add_axis(axis);
                  }) {
      if (attribute.name() == "axis" &&
          attribute.type() == onnx::AttributeProto_AttributeType_INT) {

        builder.add_axis(writeFlScalaNode(
            flatbuffers, nodesData,
            nn::I32Scalar{static_cast<int32_t>(attribute.i())}, symbols));
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
    std::cout << nodeID(node) << " attr " << attribute.DebugString();
  }

  auto flNode = builder.Finish();

  auto tensorIndex = symbols.size();
  symbols.emplace(output, tensorIndex);
  nodesData.emplace(tensorIndex,UnionType(flNode),flNode.Union());
  return tensorIndex; // builder.Finish();
}

struct OpToFusedLayerBuilderMap
    : public std::map<std::string,
                      std::function<int(flatbuffers::FlatBufferBuilder &,
                                        mapContext &, const nn::FuseCode,
                                        std::string, const onnx::NodeProto &,
                                        std::set<kLayerData, std::less<>> &,
                                        std::map<std::string, int> &symbols)>> {

  OpToFusedLayerBuilderMap() {

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       mapContext &context, const nn::FuseCode fuseCode,
                       std::string output, const onnx::NodeProto &node,
                       std::set<kLayerData, std::less<>> &nodesData,
                       std::map<std::string, int> &symbols) {
      return writeFlNodeFinish2<nn::CONV_2DBuilder>(
          flatbuffers, context, fuseCode, output, node, nodesData, symbols);
    });
  }
};

struct OpToLayerBuilderMap
    : public std::map<
          std::string,
          std::function<std::pair<nn::Layer, flatbuffers::Offset<void>>(
              flatbuffers::FlatBufferBuilder &, flatbuffers::Offset<nn::Link>,
              const onnx::NodeProto &)>> {
  OpToLayerBuilderMap() {

    emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::RELUBuilder>(flatbuffers, link, node));
    });

    emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::RELUBuilder>(flatbuffers, link, node));
    });

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::CONV_2DBuilder>(flatbuffers, link, node));
    });

    emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         flatbuffers::Offset<nn::Link> link,
                         const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::CONCATENATIONBuilder>(flatbuffers, link, node));
    });

    emplace("MaxPool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::MAX_POOL_2DBuilder>(flatbuffers, link, node));
    });

    emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::SOFTMAXBuilder>(flatbuffers, link, node));
    });

    emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      flatbuffers::Offset<nn::Link> link,
                      const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::ADDBuilder>(flatbuffers, link, node));
    });

    emplace("GlobalAveragePool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                                    flatbuffers::Offset<nn::Link> link,
                                    const onnx::NodeProto &node) {
      return UnionPair(writeFlNodeFinish<nn::AVERAGE_POOL_2DBuilder>(
          flatbuffers, link, node));
    });

    emplace("Unsqueeze", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                            flatbuffers::Offset<nn::Link> link,
                            const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::RESHAPEBuilder>(flatbuffers, link, node));
    });

    emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::RESHAPEBuilder>(flatbuffers, link, node));
    });
    emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         flatbuffers::Offset<nn::Link> link,
                         const onnx::NodeProto &node) {
      return UnionPair(
          writeFlNodeFinish<nn::GATHERBuilder>(flatbuffers, link, node));
    });

    emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node) {
      return UnionPair(writeFlNodeFinish<nn::FULLY_CONNECTEDBuilder>(
          flatbuffers, link, node));
    });
  }
};

auto fuseSGUC(flatbuffers::FlatBufferBuilder &flatbuffers, mapContext &context,
              flatbuffers::Offset<nn::Link> link, const onnx::NodeProto &Shape,
              const onnx::NodeProto &Gather, const onnx::NodeProto &Unsqueeze,
              const onnx::NodeProto &Concat) {
  nn::ConfigureBuilder builder{flatbuffers};
  builder.add_link(link);
  for (auto &attr : Gather.attribute()) {
    if (attr.name() == "axis" &&
        attr.type() == onnx::AttributeProto_AttributeType_INT) {
      builder.add_gather_axis(static_cast<int32_t>(attr.i()));
    } else {
      std::cerr << nodeID(Gather) << " attr : \n"
                << attr.DebugString() << __LINE__ << "not support\n";
    }
  }
  auto constansNode = context.outputNodeMap.at(Gather.input(1));
  for (auto &attr : constansNode.attribute()) {

    if (attr.name() == "value" &&
        attr.t().data_type() == onnx::TensorProto_DataType_INT64) {
      builder.add_gather_indices(flatbuffers.CreateVector(
          attr.t().int64_data_size(),
          std::function<int32_t(size_t i)>{
              [&](size_t i) { return attr.t().int64_data(i); }}));
    } else {
      std::cerr << nodeID(constansNode) << " attr : " << attr.type() << "\n"
                << attr.DebugString() << __LINE__ << "not support\n";
    }
  }

  for (auto &attr : Unsqueeze.attribute()) {
    if (attr.name() == "axes" &&
        attr.type() == onnx::AttributeProto_AttributeType_INTS) {
      builder.add_unsqueeze_axes(flatbuffers.CreateVector(
          attr.ints_size(), std::function<int32_t(size_t i)>{
                                [&](size_t i) { return attr.ints(i); }}));
    } else {
      std::cerr << nodeID(Gather) << " attr : \n"
                << attr.DebugString() << "not support\n";
    }
  }
  for (auto &attr : Concat.attribute()) {
    if (attr.name() == "axis" &&
        attr.type() == onnx::AttributeProto_AttributeType_INT) {
      builder.add_concat_axis(static_cast<int32_t>(attr.i()));
    } else {
      std::cerr << nodeID(Concat) << " attr : \n"
                << attr.DebugString() << "not support\n";
    }
  }
  auto ConcatTensor = context.tensorMap.at(Concat.input(1));
  if (ConcatTensor.data_type() == onnx::TensorProto_DataType_INT64) {

    builder.add_concat(flatbuffers.CreateVector(
        ConcatTensor.int32_data_size(),
        std::function<int32_t(size_t i)>{
            [&](size_t i) { return ConcatTensor.int64_data(i); }}));
  } else {
    std::cerr << nodeID(Concat) << " Tensor input : \n"
              << &ConcatTensor << "not support\n";
  }

  return builder.Finish();
}

int32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::set<kLayerData, std::less<>> &nodesData,
                     mapContext &context, const onnx::NodeProto &node,
                     std::map<std::string, int> &symbols) {

  // builder.
  static OpToLayerBuilderMap opMap{};

  static OpToFusedLayerBuilderMap opMapBuilder;
  const onnx::NodeProto *nodeP = &node;
  nn::FuseCode fuseCode = nn::FuseCode::None;
  if (nodeP->op_type() == "Clip") { // relu clip to tensor fuse
    if (nodeP->input_size() != 3) {
      std::cerr << "error: " << nodeP->op_type() << " mult input\n"
                << nodeP->DebugString();
    }
    if (context.tensorMap.at(nodeP->input()[1]).float_data(0) == 0.0f) {
      auto max = context.tensorMap.at(nodeP->input()[2]).float_data(0);
      if (max == 6.0f) {
        fuseCode = nn::FuseCode::Relu6;
      } else if (max == 1.0f) {
        fuseCode = nn::FuseCode::Relu1;
      } else {
        std::cerr << "::::::::::::::::::" << nodeID(*nodeP) << " idx "
                  << " max: " << max << std::endl;
      }
    }
    auto nodeItr = context.outputNodeMap.find(nodeP->input()[0]);
    static std::set<std::string> fusedOpCode{"Conv"};

    if (nodeItr != context.outputNodeMap.end() &&
        fusedOpCode.contains(nodeItr->second.op_type())) {
      nodeP = &(nodeItr->second);
      auto link = nn::CreateLink(
          builder, builder.CreateVector(
                       nodeP->input_size() + 1,
                       std::function<int32_t(size_t)>{[&](size_t i) {
                         if (nodeP->input_size() > i)
                           return writeFlNode(builder, nodesData, context,
                                              nodeP->input()[i], symbols);
                         else
                           return writeFlFuseNode(builder, nodesData, fuseCode,
                                                  symbols);
                       }}));
      auto tensorIndex = symbols.size();
      symbols.emplace(node.output()[0], tensorIndex);

      auto opItr = opMap.find(nodeP->op_type());
      if (opItr != opMap.end()) {
        auto ftnode = opItr->second(builder, link, *nodeP);
        nodesData.emplace(tensorIndex, ftnode.first, ftnode.second);
      } else {
        std::cerr << "error: " << nodeP->op_type() << " is not support!\n"
                  << nodeP->DebugString();
      }
      std::cout << "node idx:" << tensorIndex << " id " << nodeID(*nodeP)
                << std::endl;
      return tensorIndex;
    } else {
      std::cerr << "::::::::::::::::::" << nodeID(*nodeP) << " can not fused\n";
      exit(-1);
    }
  }

  if (nodeP->op_type() == "Concat") {
    auto &nodeConcatInput1 =
        context.outputNodeMap.find(nodeP->input()[1])->second;

    auto &nodeUnsqueeze = context.outputNodeMap.find(nodeP->input()[0])->second;
    if (nodeUnsqueeze.op_type() == "Unsqueeze") {
      auto &nodeGather =
          context.outputNodeMap.find(nodeUnsqueeze.input()[0])->second;
      if (nodeGather.op_type() == "Gather") {
        auto &nodeShape =
            context.outputNodeMap.find(nodeGather.input()[0])->second;
        if (nodeShape.op_type() == "Shape") {
          std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>" << nodeID(*nodeP)
                    << " idx " << std::endl;
          auto link = nn::CreateLink(
              builder, builder.CreateVector(
                           nodeShape.input_size(),
                           std::function<int32_t(size_t)>{[&](size_t i) {
                             return writeFlNode(builder, nodesData, context,
                                                nodeShape.input()[i], symbols);
                           }}));

          auto ftnode = fuseSGUC(builder, context, link, nodeShape, nodeGather,
                                 nodeUnsqueeze, *nodeP);

          auto tensorIndex = symbols.size();
          symbols.emplace(node.output()[0], tensorIndex);
          nodesData.emplace(tensorIndex, UnionType(ftnode), ftnode.Union());
          std::cout << "node idx:" << tensorIndex << " id " << nodeID(*nodeP)
                    << std::endl;
          return tensorIndex;
        }
      }
    }
  }

  auto link = nn::CreateLink(
      builder,
      builder.CreateVector(nodeP->input_size(),
                           std::function<int32_t(size_t)>{[&](size_t i) {
                             return writeFlNode(builder, nodesData, context,
                                                nodeP->input()[i], symbols);
                           }}));

  auto tensorIndex = symbols.size();
  symbols.emplace(node.output()[0], tensorIndex);
  auto opItr = opMap.find(nodeP->op_type());
  if (opItr != opMap.end()) {
    auto ftnode = opItr->second(builder, link, *nodeP);
    nodesData.emplace(tensorIndex, ftnode.first, ftnode.second);
  } else {
    std::cerr << "error: " << nodeP->op_type() << " is not support!\n"
              << nodeP->DebugString();
  }
  std::cout << "node idx:" << tensorIndex << " id " << nodeID(*nodeP)
            << std::endl;
  return tensorIndex;
}

auto onnxDataTypeTonn(int32_t onnxType) {
  switch (onnxType) {
  case onnx::TensorProto_DataType_FLOAT16:
    return nn::DataType::Float16;
  case onnx::TensorProto_DataType_FLOAT:
    return nn::DataType::Float32;
  case onnx::TensorProto_DataType_DOUBLE:
    return nn::DataType::Float64;
  case onnx::TensorProto_DataType_INT8:
    return nn::DataType::Int8;
  case onnx::TensorProto_DataType_INT16:
    return nn::DataType::Int16;
  case onnx::TensorProto_DataType_INT32:
    return nn::DataType::Int32;
  case onnx::TensorProto_DataType_INT64:
    return nn::DataType::Int64;
  default:
    std::cerr << "error: " << onnxType << " not convert\n";
    return nn::DataType::Unknown;
  }
}

int32_t writeFlNode(flatbuffers::FlatBufferBuilder &flbuilder,
                     std::set<kLayerData, std::less<>> &nodesData,
                     const onnx::TensorProto &tensor,
                     std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(tensor.name(), tensorIndex);
  nn::TensorInfoBuilder info{flbuilder};
  info.add_dim(flbuilder.CreateVector(
      tensor.dims_size(), std::function<uint16_t(size_t)>{
                              [&](size_t i) { return tensor.dims(i); }}));

  if (tensor.has_raw_data()) {
    nn::RawTensorBuilder builder{flbuilder};
    builder.add_info(info.Finish());
    builder.add_data(flbuilder.CreateString(tensor.raw_data()));
    builder.add_type(onnxDataTypeTonn(tensor.data_type()));
    auto flnode = builder.Finish();
    nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());
  } else {
    std::cout << tensor.DebugString();
    switch (tensor.data_type()) {
    case onnx::TensorProto_DataType_FLOAT16: {
      nn::F16TensorBuilder builder{flbuilder};
      builder.add_info(info.Finish());
      builder.add_data(
          flbuilder.CreateVector(tensor.int32_data_size(),
                                 std::function<uint16_t(size_t)>{[&](size_t i) {
                                   return tensor.int32_data(i);
                                 }}));
      auto flnode = builder.Finish();
      nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());
    } break;
    case onnx::TensorProto_DataType_FLOAT: {

      nn::F32TensorBuilder builder{flbuilder};
      builder.add_info(info.Finish());
      builder.add_data(flbuilder.CreateVector(
          tensor.float_data_size(), std::function<float(size_t)>{[&](size_t i) {
            return tensor.float_data(i);
          }}));

      auto flnode = builder.Finish();
      nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());

    } break;
    case onnx::TensorProto_DataType_DOUBLE: {
      nn::F64TensorBuilder builder{flbuilder};
      builder.add_info(info.Finish());
      builder.add_data(
          flbuilder.CreateVector(tensor.double_data_size(),
                                 std::function<double(size_t)>{[&](size_t i) {
                                   return tensor.double_data(i);
                                 }}));
      auto flnode = builder.Finish();
      nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());
    } break;

    case onnx::TensorProto_DataType_INT32: {
      nn::I32TensorBuilder builder{flbuilder};
      builder.add_info(info.Finish());
      builder.add_data(
          flbuilder.CreateVector(tensor.int32_data_size(),
                                 std::function<int32_t(size_t)>{[&](size_t i) {
                                   return tensor.int32_data(i);
                                 }}));
      auto flnode = builder.Finish();
      nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());
    } break;
    /*case onnx::TensorProto_DataType_UINT32: {
      nn::U32TensorBuilder builder{flbuilder};
      builder.add_info(info.Finish());
      builder.add_data(
          flbuilder.CreateVector(tensor.uint32_data_size(),
                                 std::function<uint32_t(size_t)>{[&](size_t i) {
                                   return tensor.uint32_data(i);
                                 }}));
      auto flnode = builder.Finish();
      nodesData.emplace(tensorIndex, UnionType(flnode), flnode.Union());
    } break;*/
    default:
      std::cerr << "onnx::TensorProto_DataType " << tensor.data_type()
                << " not support\n";
      nodesData.emplace(tensorIndex);
    }
  }

  std::cout << "tensor idx:" << tensorIndex << " id " << tensor.name()
            << std::endl;

  return tensorIndex;
}

auto TensorShapeHelper(flatbuffers::FlatBufferBuilder &flbuilder,
                       const onnx::ValueInfoProto &valueInfo) {
  nn::TensorShapeBuilder builder{flbuilder};

  builder.add_type(
      onnxDataTypeTonn(valueInfo.type().tensor_type().elem_type()));

  builder.add_dims_type(flbuilder.CreateVector(
      valueInfo.type().tensor_type().shape().dim_size(),
      std::function<nn::Dim(size_t)>{[&](size_t i) {
        switch (valueInfo.type().tensor_type().shape().dim(i).value_case()) {
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
          return nn::Dim::DimParam;
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
          return nn::Dim::DimValue;
        default:
          return nn::Dim::NONE;
        }
      }}));
  builder.add_dims(flbuilder.CreateVector(
      valueInfo.type().tensor_type().shape().dim_size(),
      std::function<flatbuffers::Offset<void>(size_t)>{[&](size_t i) {
        switch (valueInfo.type().tensor_type().shape().dim(i).value_case()) {
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
          return nn::CreateDimParam(flbuilder,
                                    flbuilder.CreateString(valueInfo.type()
                                                               .tensor_type()
                                                               .shape()
                                                               .dim(i)
                                                               .dim_param()))
              .Union();
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
          return flbuilder
              .CreateStruct<nn::DimValue>(
                  valueInfo.type().tensor_type().shape().dim(i).dim_value())
              .Union();
        default:
          return flbuilder.CreateStruct<nn::DimValue>(-1).Union();
        }
      }}));

  return builder.Finish();
}

int32_t writeFlNode(flatbuffers::FlatBufferBuilder &flbuilder,
                     std::set<kLayerData, std::less<>> &nodesData,
                     const onnx::ValueInfoProto &valueInfo,
                     std::map<std::string, int> &symbols) {
  auto tensorIndex = symbols.size();
  symbols.emplace(valueInfo.name(), tensorIndex);
  auto flNode =
      nn::CreateInputTensor(flbuilder, TensorShapeHelper(flbuilder, valueInfo));

  nodesData.emplace(tensorIndex, UnionType(flNode), flNode.Union());
  std::cout << "input idx:" << tensorIndex << " id " << valueInfo.name()
            << std::endl;
  return tensorIndex;
}

int32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::set<kLayerData, std::less<>> &nodesData,
                     mapContext &context, const std::string output,
                     std::map<std::string, int> &symbols) {
  if (auto itr = symbols.find(output); itr != symbols.end()) {
    return itr->second;
  }
  if (auto nodeItrByOP = context.outputNodeMap.find(output);
      nodeItrByOP != context.outputNodeMap.end()) {
    // std::cerr << __func__ << " node output " << output << std::endl;
    return writeFlNode(builder, nodesData, context, nodeItrByOP->second,
                       symbols);
  } else if (auto dataItrByOP = context.tensorMap.find(output);
             dataItrByOP != context.tensorMap.end()) {
    // std::cerr << __func__ << " tensor " << output << std::endl;
    return writeFlNode(builder, nodesData, dataItrByOP->second, symbols);

  } else if (auto inpouItrByName = context.graphsInputs.find(output);
             inpouItrByName != context.graphsInputs.end()) {
    // std::cerr << __func__ << " input " << output << " : "
    //           << inpouItrByName->second.name() << std::endl;
    return writeFlNode(builder, nodesData, inpouItrByName->second, symbols);
  }
  throw std::logic_error{"need output not found!"};
}

std::set<kLayerData, std::less<>>
writeFlNodeFromOutputs(flatbuffers::FlatBufferBuilder &builder,
                       const std::vector<std::string> outputs,
                       mapContext &context,
                       std::map<std::string, int> &symbols) {

  std::deque<std::string> outputsNeed{outputs.begin(), outputs.end()};
  std::set<kLayerData, std::less<>> nodesData;

  while (!outputsNeed.empty()) {
    auto outputName = outputsNeed.front();
    auto nodeIndex =
        writeFlNode(builder, nodesData, context, outputName, symbols);
    outputsNeed.pop_front();

    // outputsNeed.insert(outputsNeed.end(), needNodeNames.begin(),
    //                    needNodeNames.end());
  }

  return nodesData;
}

flatbuffers::Offset<nn::Output>
writeFlOutputs(flatbuffers::FlatBufferBuilder &flBuilder,
               const onnx::ValueInfoProto &output,
               std::map<std::string, int> &symbols) {
  return nn::CreateOutput(flBuilder, symbols.at(output.name()),
                          TensorShapeHelper(flBuilder, output));
}