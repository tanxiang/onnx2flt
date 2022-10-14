#include "noderemap.hh"
#include <queue>

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

auto opReMapDefault(std::vector<const onnx::NodeProto *> &vRemap,
                    std::set<const onnx::NodeProto *> &parsedNodes,
                    mapContext &context) {
  const onnx::NodeProto *node = vRemap[0];

  parsedNodes.emplace(node);
}

struct OpReMap
    : public std::map<
          std::string,
          std::function<std::vector<const onnx::NodeProto *>(
              const onnx::NodeProto *, std::vector<const onnx::NodeProto *> &vRemap,
              std::set<const onnx::NodeProto *> &parsedNodes, mapContext &)>> {
  OpReMap() {
    emplace("Conv", [this](const onnx::NodeProto *node,
                           std::vector<const onnx::NodeProto *> &vRemap,
                           std::set<const onnx::NodeProto *> &parsedNodes,
                           mapContext &context) {
      parsedNodes.emplace(node);
      auto output = node->output()[0];
      auto outCount = context.inputNodeMap.count(output);
      switch (outCount) {
      case 0: {
        // test graph output
        break;
      }
      case 1: {
        auto walkItr = context.inputNodeMap.find(output);
        auto needNodes =
            (*this)[walkItr->second.op_type()](&(walkItr->second),vRemap, parsedNodes, context);
        break;
      }

      default: {
        std::cout << "Conv to " << outCount << " nodes\n";
        auto range = context.inputNodeMap.equal_range(output);
        for (auto itr = range.first; itr != range.second; ++itr) {
          auto walkItr = itr;
          std::cout << "Conv to " << walkItr->second.op_type() << " output "
                    << walkItr->second.output()[0] << std::endl;
          if (walkItr->second.op_type() == "Clip") {
            vRemap.emplace_back(&walkItr->second);
          }
          if (walkItr->second.op_type() == "Relu") {
          }
        }
      }
      }

      return std::vector<const onnx::NodeProto *>{};
    });

    emplace("Clip", [](const onnx::NodeProto *node,
                       std::vector<const onnx::NodeProto *> &vRemap,
                       std::set<const onnx::NodeProto *> &parsedNodes,
                       mapContext &context) {
      return std::vector<const onnx::NodeProto *>{};
      parsedNodes.emplace(node);
      auto output = node->output()[0];
      auto range = context.inputNodeMap.equal_range(output);
      for (auto itr = range.first; itr != range.second; ++itr) {
        auto walkItr = itr;
        std::cout << "Conv to " << walkItr->second.op_type() << " output "
                  << walkItr->second.output()[0] << std::endl;
        if (walkItr->second.op_type() == "Clip") {
          vRemap.emplace_back(&walkItr->second);
        }
        if (walkItr->second.op_type() == "Relu") {
        }
      }
      return std::vector<const onnx::NodeProto *>{};
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
    std::queue<const onnx::NodeProto *> needNodes;
    needNodes.emplace(&(inputItr->second));
    while (!needNodes.empty()) {
      auto nodePtr = needNodes.front();

      if (parsedNodes.find(nodePtr) == parsedNodes.end()) {
        std::vector<const onnx::NodeProto *> vRemap{};

        auto opItr = opReMap.find(nodePtr->op_type());
        if (opItr != opReMap.end()) {
          opItr->second(nodePtr, vRemap, parsedNodes, context);
        } else {
        }

        if (nodePtr->has_name()) {
          std::cout << "root node " << nodePtr->name() << std::endl;
        } else {
          std::cout << "root node is noname" << std::endl;
        }

        nodePtr->op_type();
        nodePtr->output();
        needNodes.pop();
        vvRemap.emplace_back(vRemap);
      }
    }
  }
  return vvRemap;
}
