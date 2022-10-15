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

std::vector<const onnx::NodeProto *> opReMapDefault(
    const onnx::NodeProto *node, std::vector<const onnx::NodeProto *> &vRemap,
    std::set<const onnx::NodeProto *> &parsedNodes, mapContext &context) {
  if (!vRemap.empty()) {
    std::cerr << "!vRemap.empty() in " << node->op_type() << std::endl;
    return {node};
  }
  vRemap.emplace_back(node);
  parsedNodes.emplace(node);
  return {};
}

struct OpReMap
    : public std::map<
          std::string,
          std::function<std::vector<const onnx::NodeProto *>(
              const onnx::NodeProto *,
              std::vector<const onnx::NodeProto *> &vRemap,
              std::set<const onnx::NodeProto *> &parsedNodes, mapContext &)>> {
  OpReMap() {
    emplace(
        "Conv",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
          if (!vRemap.empty()) {
            std::cerr << "!vRemap.empty() in " << node->op_type() << std::endl;
            return {node};
          }
          //std::cerr <<"std::cerr cpp v = "<< __cplusplus<<std::endl;
          if (parsedNodes.contains(node)) {
          }
          parsedNodes.emplace(node);
          vRemap.emplace_back(node);
          auto output = node->output()[0];

          auto outCount = context.inputNodeMap.count(output);

          switch (outCount) {
          case 0: {
            std::cerr << "node output : " << node->output()[0]
                      << "need in graphs output" << std::endl;
            return {};
          }
          case 1: {
            auto walkItr = context.inputNodeMap.find(output);
            std::cout << "Conv to " << walkItr->second.op_type() << " output "
                      << walkItr->second.output()[0] << std::endl;

            return (*this)[walkItr->second.op_type()](
                &(walkItr->second), vRemap, parsedNodes, context);
          }

          default: {
            std::cout << "Conv to " << outCount << " nums nodes\n";
            auto range = context.inputNodeMap.equal_range(output);
            for (auto itr = range.first; itr != range.second; ++itr) {
              auto walkItr = itr;
              std::cout << "Conv to " << walkItr->second.op_type() << " output "
                        << walkItr->second.output()[0] << std::endl;
            }
          }
          }

          return {};
        });

    emplace(
        "Clip",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
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
            std::cout << "Clip to " << walkItr->second.op_type() << " output "
                      << walkItr->second.output()[0] << std::endl;

            return (*this)[walkItr->second.op_type()](
                &(walkItr->second), vRemap, parsedNodes, context);
          }

          default: {
            std::cout << "Clip to " << outCount << " nums nodes\n";
            auto range = context.inputNodeMap.equal_range(output);
            for (auto itr = range.first; itr != range.second; ++itr) {
              auto walkItr = itr;
              std::cout << "Clip to " << walkItr->second.op_type() << " output "
                        << walkItr->second.output()[0] << std::endl;
            }
          }
          }
          return std::vector<const onnx::NodeProto *>{};
        });

    emplace(
        "Relu",
        [this](const onnx::NodeProto *node,
               std::vector<const onnx::NodeProto *> &vRemap,
               std::set<const onnx::NodeProto *> &parsedNodes,
               mapContext &context) -> std::vector<const onnx::NodeProto *> {
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
            std::cout << "Relu to " << walkItr->second.op_type() << " output "
                      << walkItr->second.output()[0] << std::endl;

            auto needNodes = (*this)[walkItr->second.op_type()](
                &(walkItr->second), vRemap, parsedNodes, context);
            break;
          }

          default: {
            std::cout << "Relu to " << outCount << " nums nodes\n";
            auto range = context.inputNodeMap.equal_range(output);
            for (auto itr = range.first; itr != range.second; ++itr) {
              auto walkItr = itr;
              std::cout << "Relu to " << walkItr->second.op_type() << " output "
                        << walkItr->second.output()[0] << std::endl;
            }
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
