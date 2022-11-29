#include "noderemap.hh"
#include <queue>
#include <sstream>

struct nodeGroup : public std::vector<onnx::NodeProto *> {
  // nodeGroupLinks getLinks() { return {}; }

  std::vector<std::string> getNeedInput() {
    auto itr = begin();
    std::vector<std::string> names;
    for (auto &name : (*itr)->input()) {
      names.emplace_back(name);
    }
    return names;
  }
  std::vector<std::string> getNeedOutput() {
    auto itr = rbegin();
    std::vector<std::string> names;
    for (auto &name : (*itr)->output()) {
      names.emplace_back(name);
    }
    return names;
  }

  auto getItrFromPtr(onnx::NodeProto *pnode) {
    auto nodeItr = begin();
    while (nodeItr != end()) {
      if ((*nodeItr) == pnode) {
        return nodeItr;
      }
      nodeItr++;
    }
    return end();
  }

  auto getItrFormInput(std::string input) {
    auto nodeItr = begin();
    while (nodeItr != end()) {
      for (auto &inputName : (*nodeItr)->input()) {
        if (input == inputName)
          return nodeItr;
      }
      nodeItr++;
    }
    return end();
  }

  auto getItrFormOutput(std::string input) {
    auto nodeItr = rbegin();
    while (nodeItr != rend()) {
      for (auto &inputName : (*nodeItr)->output()) {
        if (input == inputName)
          return nodeItr;
      }
      nodeItr++;
    }
    return rend();
  }
  nodeGroup(std::vector<onnx::NodeProto *> nodes)
      : std::vector<onnx::NodeProto *>{nodes} {}
};

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

template <typename Range>
auto packNode(Range range, std::set<const onnx::NodeProto *> &addNodes) {
  std::vector<const onnx::NodeProto *> ret{};
  for (auto itr = range.first; itr != range.second; ++itr) {

    if (addNodes.emplace(&itr->second).second) {
      ret.emplace_back(&itr->second);
      std::cout << '<' << nodeID(itr->second) << '>';

    } else {
      std::cout << '[' << nodeID(itr->second) << ']';
    }
  }
  std::cout << std::endl;
  return ret;
}

auto packNode(const onnx::NodeProto *node,
              std::set<const onnx::NodeProto *> &addNodes) {
  std::vector<const onnx::NodeProto *> ret{};
  if (addNodes.emplace(node).second) {
    ret.emplace_back(node);
    std::cout << '<' << nodeID(*node) << "> ";
  } else {
    std::cout << '[' << nodeID(*node) << "] ";
  }
  std::cout << std::endl;
  return ret;
}

struct OpReMap
    : public std::map<
          std::string,
          std::function<std::vector<const onnx::NodeProto *>(
              const onnx::NodeProto *, std::vector<const onnx::NodeProto *> &,
              std::set<const onnx::NodeProto *> &, mapContext &)>> {

  std::vector<const onnx::NodeProto *> opReMapDefault(
      const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
      std::set<const onnx::NodeProto *> &addNodes, mapContext &context) {
    if (!vRemap.empty()) {
      std::cout << "\tnew group";
      return packNode(node, addNodes);
    }
    auto output = node->output()[0];
    auto outCount = context.inputNodeMap.count(output);
    vRemap.emplace_back(node);
    std::cout << "\t\t" << nodeID(*node) << " to ";
    return packNode(context.inputNodeMap.equal_range(output), addNodes);
  }

  template <typename... Targs>
  auto checkNode(const onnx::NodeProto *node, Targs &&...args) {
    auto opItr = find(node->op_type());
    if (opItr != end()) {
      return opItr->second(node, std::forward<Targs>(args)...);
    }
    return opReMapDefault(node, std::forward<Targs>(args)...);
  }

  OpReMap() {
    emplace(
        "Conv",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &addNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          if (!vRemap.empty()) {
            std::cout << "\tnew group";

            return packNode(node, addNodes);
          }
          vRemap.emplace_back(node);
          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);

          switch (outCount) {
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << '\t' << nodeID(*node) << " to "
                      << nodeID(walkItr->second) << " output "
                      << walkItr->second.output()[0] << std::endl;

            return checkNode(&(walkItr->second), vRemap, addNodes, context);
          }
          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output), addNodes);
          }

          return {};
        });

    emplace(
        "Clip",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &addNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          /*if (parsedNodes.contains(node)) {
            return {};
          }*/
          vRemap.emplace_back(node);
          addNodes.emplace(node);

          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);
          switch (outCount) {
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << '\t' << nodeID(*node) << " to "
                      << nodeID(walkItr->second) << " output "
                      << walkItr->second.output()[0] << std::endl;

            return checkNode(&(walkItr->second), vRemap, addNodes, context);
          }

          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output), addNodes);
          }
          return {};
        });

    emplace(
        "Relu",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &addNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          vRemap.emplace_back(node);
          addNodes.emplace(node);

          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);
          switch (outCount) {
          case 0:
            std::cerr << "node output : " << node->output()[0]
                      << "need in graphs output" << std::endl;
            return {};
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << '\t' << nodeID(*node) << " to "
                      << nodeID(walkItr->second) << " output "
                      << walkItr->second.output()[0] << std::endl;

            return checkNode(&(walkItr->second), vRemap, addNodes, context);
          }

          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output), addNodes);
          }

          return {};
        });
  }
};

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &input, mapContext &context) {
  static OpReMap opReMap{};
  std::vector<std::vector<const onnx::NodeProto *>> vvRemap{};
  auto inputItr = context.inputNodeMap.find(input.name()); // FIXME need eq
                                                           // range
  if (inputItr != context.inputNodeMap.end()) {
    std::set<const onnx::NodeProto *> addNodes;
    std::deque<const onnx::NodeProto *> needNodes;
    needNodes.emplace_back(&(inputItr->second));
    while (!needNodes.empty()) {
      auto nodePtr = needNodes.front();
      std::vector<const onnx::NodeProto *> vRemap{};

      if (nodePtr->has_name() && !nodePtr->name().empty()) {
        std::cout << "node group start at " << nodePtr->name() << " op "
                  << nodePtr->op_type() << " : " << needNodes.size()
                  << std::endl;
      } else {
        std::cout << "node group start at op " << nodePtr->op_type() << " : "
                  << needNodes.size() << std::endl;
      }

      auto needs = opReMap.checkNode(nodePtr, vRemap, addNodes, context);
      needNodes.insert(needNodes.end(), needs.begin(), needs.end());

      // nodePtr->op_type();
      // nodePtr->output();
      if (!vRemap.empty())
        vvRemap.emplace_back(vRemap);
      needNodes.pop_front();
    }
  }
  return vvRemap;
}

struct fusedNode : public std::set<std::string> {
  fusedNode() : std::set<std::string>{"Conv", "Add", "Sub", "Pool"} {}
};

struct OpRReMap
    : public std::map<
          std::string,
          std::function<std::vector<const onnx::NodeProto *>(
              const onnx::NodeProto *, std::vector<const onnx::NodeProto *> &,
              std::set<const onnx::NodeProto *> &, mapContext &)>> {

  auto packNodeByInput(const onnx::NodeProto &node, mapContext &context,
                       std::set<const onnx::NodeProto *> &addNodes) {
    std::vector<const onnx::NodeProto *> ret{};

    for (auto &input : node.input()) {

      if (auto const nodePair = context.outputNodeMap.find(input);
          context.outputNodeMap.end() != nodePair) {
        if (addNodes.emplace(&nodePair->second).second) {
          ret.emplace_back(&nodePair->second);
          std::cout << '<' << nodeID(nodePair->second) << "> ";
        } else {
          std::cout << '[' << nodeID(nodePair->second) << "] ";
        }
      } else if (auto const tensorPair = context.tensorMap.find(input);
                 context.tensorMap.end() != tensorPair) {
        std::cout << '{' << tensorID(tensorPair->second) << "} ";
      } else {
        std::cout << '(' << input << ") ";
      }
    }
    std::cout << std::endl;
    return ret;
  }

  std::vector<const onnx::NodeProto *> opReMapDefault(
      const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
      std::set<const onnx::NodeProto *> &addNodes, mapContext &context) {

    if (!vRemap.empty()) {
      std::cout << "\tnew group";
      return packNode(node, addNodes);
    }

    vRemap.emplace_back(node);
    addNodes.emplace(node);

    std::cout << "\t\t" << nodeID(*node) << " to ";

    return packNodeByInput(*node, context, addNodes);
  }

  fusedNode fusedNodeSet;
  std::vector<const onnx::NodeProto *> checkfusedNode(
      const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
      std::set<const onnx::NodeProto *> &addNodes, mapContext &context) {

    if (fusedNodeSet.contains(node->op_type())) {

      vRemap.emplace_back(node);
      addNodes.emplace(node);

      std::cout << "\t\t" << nodeID(*node) << " to ";
      return packNodeByInput(*node, context, addNodes);
    }

    return packNode(node, addNodes);
  }

  std::vector<const onnx::NodeProto *> checkfuseNode(
      const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
      std::set<const onnx::NodeProto *> &addNodes, mapContext &context) {
    addNodes.emplace(node);

    auto input = node->input()[0];
    auto inputNode = context.outputNodeMap.find(input);

    std::vector<const onnx::NodeProto *> needs;
    switch (context.inputNodeMap.count(input)) {
    case 1:
      std::cout << '\t' << nodeID(*node) << " to " << nodeID(inputNode->second)
                << " output " << inputNode->second.output()[0] << std::endl;

      needs = checkfusedNode(&(inputNode->second), vRemap, addNodes, context);
      break;

    default:
      std::cout << "\t\t" << nodeID(*node) << " to ";
      needs = packNodeByInput(*node, context, addNodes);
    }

    vRemap.emplace_back(node);

    return needs;
  }

  template <typename... Targs>
  auto checkNode(const onnx::NodeProto *node, Targs &&...args) {
    auto opItr = find(node->op_type());
    if (opItr != end()) {
      return opItr->second(node, std::forward<Targs>(args)...);
    }
    return opReMapDefault(node, std::forward<Targs>(args)...);
  }

  OpRReMap() {

    emplace(
        "Clip",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &addNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          /*if (parsedNodes.contains(node)) {
            return {};
          }*/
          return checkfuseNode(node, vRemap, addNodes, context);
        });

    emplace(
        "Relu",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &addNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          return checkfuseNode(node, vRemap, addNodes, context);
        });
  }
};

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromOutputs(const std::vector<std::string> outputs,
                        mapContext &context) {
  static OpRReMap opReMap{};
  std::vector<std::vector<const onnx::NodeProto *>> vvRemap{};
  std::deque<const onnx::NodeProto *> needNodes;

  for (auto &output : outputs) {
    auto outputItr = context.outputNodeMap.find(output);
    if (outputItr != context.outputNodeMap.end()) {
      needNodes.emplace_back(&(outputItr->second));
    }
  }

  std::set<const onnx::NodeProto *> addNodes;
  while (!needNodes.empty()) {
    auto nodePtr = needNodes.front();
    std::vector<const onnx::NodeProto *> vRemap{};

    auto needs = opReMap.checkNode(nodePtr, vRemap, addNodes, context);
    if (!vRemap.empty())
      vvRemap.emplace_back(std::move(vRemap));
    needNodes.insert(needNodes.end(), needs.begin(), needs.end());
    needNodes.pop_front();
  }
  return vvRemap;
}

uint32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::map<int, uLayerData> &nodesData, mapContext &context,
                     const std::string output,
                     std::map<std::string, int> &symbols);

template <typename NodeTensorBuilder>
auto writeFlTensorFinish(flatbuffers::FlatBufferBuilder &flBuilder,
                         const onnx::TensorProto &Tensor,
                         nn::FuseCode *fuseCode) {
  NodeTensorBuilder builder{flBuilder};
  return builder.Finish();
}

template <typename NodeTypeBuilder>
auto writeFlNodeFinish(flatbuffers::FlatBufferBuilder &flBuilder,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
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
  if constexpr (requires(NodeTypeBuilder & builder, nn::FuseCode & fuseCode) {
                  builder.add_fuse_code(fuseCode);
                }) {
    // std::cout << nodes[1]->DebugString();
    if (fuseCode)
      builder.add_fuse_code(*fuseCode);
  }
  return builder.Finish();
}

template <typename Layer>
inline auto UnionPair(flatbuffers::Offset<Layer> layer) {
  return std::pair<nn::Layer, flatbuffers::Offset<void>>{
      nn::LayerTraits<Layer>::enum_value, layer.Union()};
}

struct OpToLayerBuilderMap
    : public std::map<
          std::string,
          std::function<std::pair<nn::Layer, flatbuffers::Offset<void>>(
              flatbuffers::FlatBufferBuilder &, flatbuffers::Offset<nn::Link>,
              const onnx::NodeProto &, nn::FuseCode *)>> {
  OpToLayerBuilderMap() {

    emplace("Clip", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::RELUBuilder>(flatbuffers, link,
                                                          node, fuseCode));
    });

    emplace("Relu", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::RELUBuilder>(flatbuffers, link,
                                                          node, fuseCode));
    });

    emplace("Conv", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::CONV_2DBuilder>(flatbuffers, link,
                                                             node, fuseCode));
    });

    emplace("Concat", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         flatbuffers::Offset<nn::Link> link,
                         const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::CONCATENATIONBuilder>(
          flatbuffers, link, node, fuseCode));
    });

    emplace("MaxPool", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::MAX_POOL_2DBuilder>(
          flatbuffers, link, node, fuseCode));
    });

    emplace("Softmax", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::SOFTMAXBuilder>(flatbuffers, link,
                                                             node, fuseCode));
    });

    emplace("Add", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      flatbuffers::Offset<nn::Link> link,
                      const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(
          writeFlNodeFinish<nn::ADDBuilder>(flatbuffers, link, node, fuseCode));
    });

    emplace("GlobalAveragePool",
            [](flatbuffers::FlatBufferBuilder &flatbuffers,
               flatbuffers::Offset<nn::Link> link, const onnx::NodeProto &node,
               nn::FuseCode *fuseCode) {
              return UnionPair(writeFlNodeFinish<nn::AVERAGE_POOL_2DBuilder>(
                  flatbuffers, link, node, fuseCode));
            });

    emplace("Unsqueeze", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                            flatbuffers::Offset<nn::Link> link,
                            const onnx::NodeProto &node,
                            nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::RESHAPEBuilder>(flatbuffers, link,
                                                             node, fuseCode));
    });

    emplace("Reshape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                          flatbuffers::Offset<nn::Link> link,
                          const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::RESHAPEBuilder>(flatbuffers, link,
                                                             node, fuseCode));
    });
    emplace("Gather", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                         flatbuffers::Offset<nn::Link> link,
                         const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::GATHERBuilder>(flatbuffers, link,
                                                            node, fuseCode));
    });

    emplace("Gemm", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                       flatbuffers::Offset<nn::Link> link,
                       const onnx::NodeProto &node, nn::FuseCode *fuseCode) {
      return UnionPair(writeFlNodeFinish<nn::FULLY_CONNECTEDBuilder>(
          flatbuffers, link, node, fuseCode));
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
    }else {
      std::cerr << nodeID(Concat) << " attr : \n"
                << attr.DebugString() << "not support\n";
    }
  }
    auto ConcatTensor = context.tensorMap.at(Concat.input(1));
    if(ConcatTensor.data_type()==onnx::TensorProto_DataType_INT64){

      builder.add_concat(flatbuffers.CreateVector(
          ConcatTensor.int32_data_size(), std::function<int32_t(size_t i)>{
                                [&](size_t i) { return ConcatTensor.int64_data(i); }}));
    }else {
      std::cerr << nodeID(Concat) << " Tensor input : \n"
                << &ConcatTensor << "not support\n";
    }

  return UnionPair(builder.Finish());
}

uint32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::map<int, uLayerData> &nodesData, mapContext &context,
                     const onnx::NodeProto &node,
                     std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(node.output()[0], tensorIndex);

  // builder.
  const onnx::NodeProto *nodeP = &node;
  nn::FuseCode *fuseCodeP{nullptr};
  nn::FuseCode fuseCode;
  if (nodeP->input_size() == 1) {
    if (nodeP->op_type() == "Relu" ||
        nodeP->op_type() == "Clip") { // relu clip to tensor fuse
      if (nodeP->input_size() != 1) {
        std::cerr << "error: " << nodeP->op_type() << " mult input\n"
                  << nodeP->DebugString();
      }
      fuseCode = nn::FuseCode::Relu;
      auto nodeItr = context.outputNodeMap.find(nodeP->input()[0]);
      if (nodeItr != context.outputNodeMap.end()) {
        nodeP = &(nodeItr->second);
      }
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
                    << " idx " << tensorIndex << std::endl;
          auto link = nn::CreateLink(
              builder, builder.CreateVector(
                           nodeShape.input_size(),
                           std::function<uint32_t(size_t)>{[&](size_t i) {
                             return writeFlNode(builder, nodesData, context,
                                                nodeShape.input()[i], symbols);
                           }}));

          auto ftnode = fuseSGUC(builder, context, link, nodeShape, nodeGather,
                                 nodeUnsqueeze, *nodeP);
          nodesData.emplace(tensorIndex,
                            uLayerData{ftnode.first, ftnode.second});

          return tensorIndex;
        }
      }
    }
  }

  auto link = nn::CreateLink(
      builder,
      builder.CreateVector(nodeP->input_size(),
                           std::function<uint32_t(size_t)>{[&](size_t i) {
                             return writeFlNode(builder, nodesData, context,
                                                nodeP->input()[i], symbols);
                           }}));

  static OpToLayerBuilderMap opMap{};
  auto opItr = opMap.find(nodeP->op_type());
  if (opItr != opMap.end()) {
    auto ftnode = opItr->second(builder, link, *nodeP, fuseCodeP);
    nodesData.emplace(tensorIndex, uLayerData{ftnode.first, ftnode.second});
  } else {
    std::cerr << "error: " << nodeP->op_type() << " is not support!\n"
              << nodeP->DebugString();
  }
  std::cout << "node idx:" << tensorIndex << " id " << nodeID(*nodeP)
            << std::endl;
  return tensorIndex;
}

uint32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::map<int, uLayerData> &nodesData, mapContext &context,
                     const onnx::TensorProto &tensor,
                     std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(tensor.name(), tensorIndex);

  std::cout << "tensor idx:" << tensorIndex << " id " << tensor.name()
            << std::endl;

  nodesData.emplace(tensorIndex, uLayerData{});
  return tensorIndex;
}

uint32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::map<int, uLayerData> &nodesData, mapContext &context,
                     const onnx::ValueInfoProto &valueInfo,
                     std::map<std::string, int> &symbols) {
  auto tensorIndex = symbols.size();
  symbols.emplace(valueInfo.name(), tensorIndex);

  std::cout << "input idx:" << tensorIndex << " id " << valueInfo.name()
            << std::endl;

  nodesData.emplace(tensorIndex, uLayerData{});
  return tensorIndex;
}

uint32_t writeFlNode(flatbuffers::FlatBufferBuilder &builder,
                     std::map<int, uLayerData> &nodesData, mapContext &context,
                     const std::string output,
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
    return writeFlNode(builder, nodesData, context, dataItrByOP->second,
                       symbols);

  } else if (auto inpouItrByName = context.graphsInputs.find(output);
             inpouItrByName != context.graphsInputs.end()) {
    // std::cerr << __func__ << " input " << output << " : "
    //           << inpouItrByName->second.name() << std::endl;
    return writeFlNode(builder, nodesData, context, inpouItrByName->second,
                       symbols);
  }
  throw std::logic_error{"need output not found!"};
}

std::map<int, uLayerData>
writeFlNodeFromOutputs(flatbuffers::FlatBufferBuilder &builder,
                       const std::vector<std::string> outputs,
                       mapContext &context) {

  std::deque<std::string> outputsNeed{outputs.begin(), outputs.end()};
  std::map<std::string, int> symbols;
  std::map<int, uLayerData> nodesData;

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