#include "noderemap.hh"
#include <queue>
#include <sstream>

OpToReMap::OpToReMap() {
  emplace("Dropout",
          [](flatbuffers::FlatBufferBuilder &flatbuffers,
             const onnx::NodeProto &node,
             mapContext &
                 context) { // auto flNode = getFlNode<nn::Dropout>(flatbuffers,
                            // node, context); return UnionPair(flNode);
          });

  emplace("Shape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                      const onnx::NodeProto &node, mapContext &context) {
    // auto flNode = getFlNode<nn::>(flatbuffers, node, context);
    // return UnionPair(flNode);
  });

  emplace("Constant",
          [](flatbuffers::FlatBufferBuilder &flatbuffers,
             const onnx::NodeProto &node,
             mapContext &context) { // auto flNode =
                                    // getFlNode<nn::Cons>(flatbuffers,
                                    // node, context); return
                                    // UnionPair(flNode);
          });
}

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
  if (node.has_name() && !node.name().empty()) {
    return node.name();
  }
  std::ostringstream id{};
  id << node.op_type() << ' ' << &node;
  return id.str();
}

struct OpReMap
    : public std::map<
          std::string,
          std::function<std::vector<const onnx::NodeProto *>(
              const onnx::NodeProto *,
              std::vector<const onnx::NodeProto *> &vRemap,
              std::set<const onnx::NodeProto *> &parsedNodes, mapContext &)>> {

  template <typename Range>
  auto packNode(Range range, std::set<const onnx::NodeProto *> &parsedNodes) {
    std::vector<const onnx::NodeProto *> ret{};
    for (auto itr = range.first; itr != range.second; ++itr) {
      if (parsedNodes.contains(&itr->second)) {
        std::cout << '[' << nodeID(itr->second) << ']';

      } else {
        std::cout << '<' << nodeID(itr->second) << '>';
        ret.emplace_back(&itr->second);
      }
    }
    std::cout << std::endl;
    return ret;
  }

  std::vector<const onnx::NodeProto *> opReMapDefault(
      const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
      std::set<const onnx::NodeProto *> &parsedNodes, mapContext &context) {
    if (!vRemap.empty()) {
      // std::cerr << "!vRemap.empty() in " << node->op_type() << std::endl;
      return {node};
    }
    if (parsedNodes.contains(node)) {
      return {};
    }
    vRemap.emplace_back(node);
    parsedNodes.emplace(node);

    auto output = node->output()[0];
    auto outCount = context.inputNodeMap.count(output);
    std::cout << "\t\t" << nodeID(*node) << " to ";
    return packNode(context.inputNodeMap.equal_range(output), parsedNodes);
  }

  template <typename... Targs>
  auto checkNode(const onnx::NodeProto *node, Targs... args) {
    auto opItr = find(node->op_type());
    if (opItr != end()) {
      return opItr->second(node, args...);
    }
    return opReMapDefault(node, args...);
  }

  OpReMap() {
    emplace(
        "Conv",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          if (!vRemap.empty()) {
            // std::cerr << "!vRemap.empty() in " << node->op_type() <<
            // std::endl;
            return {node};
          }
          // std::cerr <<"std::cerr cpp v = "<< __cplusplus<<std::endl;
          if (parsedNodes.contains(node)) {
            return {};
          }
          parsedNodes.emplace(node);
          vRemap.emplace_back(node);
          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);

          switch (outCount) {
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << '\t' << nodeID(*node) << " to "
                      << nodeID(walkItr->second) << " output "
                      << walkItr->second.output()[0] << std::endl;

            return checkNode(&(walkItr->second), vRemap, parsedNodes, context);
          }
          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output),
                            parsedNodes);
          }

          return {};
        });

    emplace(
        "Clip",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          if (parsedNodes.contains(node)) {
            return {};
          }
          vRemap.emplace_back(node);
          parsedNodes.emplace(node);

          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);
          switch (outCount) {
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << '\t' << nodeID(*node) << " to "
                      << nodeID(walkItr->second) << " output "
                      << walkItr->second.output()[0] << std::endl;

            return checkNode(&(walkItr->second), vRemap, parsedNodes, context);
          }

          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output),
                            parsedNodes);
          }
          return {};
        });

    emplace(
        "Relu",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          if (parsedNodes.contains(node)) {
            return {};
          }
          vRemap.emplace_back(node);
          parsedNodes.emplace(node);

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

            return checkNode(&(walkItr->second), vRemap, parsedNodes, context);
          }

          default:
            std::cout << "\t\t" << nodeID(*node) << " to ";
            return packNode(context.inputNodeMap.equal_range(output),
                            parsedNodes);
          }

          return {};
        });
  }
};

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &input, mapContext &context) {
  static OpReMap opReMap{};
  std::vector<std::vector<const onnx::NodeProto *>> vvRemap{};
  auto inputItr = context.inputNodeMap.find(input.name());
  if (inputItr != context.inputNodeMap.end()) {
    std::set<const onnx::NodeProto *> parsedNodes;
    std::deque<const onnx::NodeProto *> needNodes;
    needNodes.emplace_back(&(inputItr->second));
    while (!needNodes.empty()) {
      auto nodePtr = needNodes.front();
      if (parsedNodes.find(nodePtr) == parsedNodes.end()) {
        std::vector<const onnx::NodeProto *> vRemap{};

        if (nodePtr->has_name() && !nodePtr->name().empty()) {
          std::cout << "node group start at " << nodePtr->name() << " op "
                    << nodePtr->op_type() << " : " << needNodes.size()
                    << std::endl;
        } else {
          std::cout << "node group start at op " << nodePtr->op_type() << " : "
                    << needNodes.size() << std::endl;
        }

        auto needs = opReMap.checkNode(nodePtr, vRemap, parsedNodes, context);
        needNodes.insert(needNodes.end(), needs.begin(), needs.end());

        nodePtr->op_type();
        nodePtr->output();
        vvRemap.emplace_back(vRemap);
      }
      needNodes.pop_front();
    }
  }
  return vvRemap;
}
