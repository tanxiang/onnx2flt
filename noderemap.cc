#include "noderemap.hh"

OpToReMap::OpToReMap() {
     emplace("Dropout", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                              const onnx::NodeProto &node, mapContext &context)
       { //auto flNode = getFlNode<nn::Dropout>(flatbuffers, node, context);
          //return UnionPair(flNode);
        });

        emplace("Shape", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                            const onnx::NodeProto &node, mapContext &context) {
          //auto flNode = getFlNode<nn::>(flatbuffers, node, context);
          //return UnionPair(flNode);
        });


        emplace("Constant", [](flatbuffers::FlatBufferBuilder &flatbuffers,
                               const onnx::NodeProto &node, mapContext &context)
       { //auto flNode = getFlNode<nn::Cons>(flatbuffers, node, context); return
       //UnionPair(flNode);
        });
  }