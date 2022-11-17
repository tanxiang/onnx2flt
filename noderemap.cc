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

struct uLayerData //: public flatbuffers::Offset<void> , nn::Layer
{
  nn::Layer type;
  flatbuffers::Offset<void> data;
  /* data */
};

auto writeFlNode(
    flatbuffers::FlatBufferBuilder &builder,
    std::map<int, uLayerData> &nodesData,
    mapContext &context, const std::string output,
    std::map<std::string, int> &symbols);

auto writeFlNode(
    flatbuffers::FlatBufferBuilder &builder,
    std::map<int, uLayerData> &nodesData,
    mapContext &context, const onnx::NodeProto &node,
    std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(node.output()[0], tensorIndex);
  // builder.

  nodesData.emplace(tensorIndex, uLayerData{});
  return tensorIndex;
}

auto writeFlNode(
    flatbuffers::FlatBufferBuilder &builder,
    std::map<int, uLayerData> &nodesData,
    mapContext &context, const onnx::TensorProto &tensor,
    std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(tensor.name(), tensorIndex);

  nodesData.emplace(tensorIndex, uLayerData{});
  return tensorIndex;
}

auto writeFlNode(
    flatbuffers::FlatBufferBuilder &builder,
    std::map<int, uLayerData> &nodesData,
    mapContext &context, const onnx::ValueInfoProto &valueInfo,
    std::map<std::string, int> &symbols) {

  auto tensorIndex = symbols.size();
  symbols.emplace(valueInfo.name(), tensorIndex);
  
  nodesData.emplace(tensorIndex, uLayerData{});
  return tensorIndex;
}

auto writeFlNode(
    flatbuffers::FlatBufferBuilder &builder,
    std::map<int, uLayerData> &nodesData,
    mapContext &context, const std::string output,
    std::map<std::string, int> &symbols) {

  if (auto nodeItrByOP = context.outputNodeMap.find(output);
      nodeItrByOP != context.outputNodeMap.end()) {
    return writeFlNode(builder, nodesData, context, nodeItrByOP->second,
                       symbols);
  } else if (auto dataItrByOP = context.tensorMap.find(output);
             dataItrByOP != context.tensorMap.end()) {
    return writeFlNode(builder, nodesData, context, dataItrByOP->second,
                       symbols);

  } else if (auto inpouItrByName = context.graphsInputs.find(output);
             inpouItrByName != context.graphsInputs.end()) {
    return writeFlNode(builder, nodesData, context, inpouItrByName->second,
                       symbols);
  }
  // nodeTypes.emplace_back();
  // nodeVals.emplace_back();
  throw std::logic_error{"need output not found!"};
}

std::map<int,uLayerData>
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