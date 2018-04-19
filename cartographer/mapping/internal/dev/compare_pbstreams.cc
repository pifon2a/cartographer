
#include "cartographer/io/proto_stream.h"
#include "cartographer/mapping/id.h"
#include "cartographer/mapping/proto/pose_graph.pb.h"
#include "cartographer/mapping/proto/trajectory_builder_options.pb.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <iostream>

DEFINE_string(trimmed_pb, "", "trimmed pbstream.");
DEFINE_string(golden_pb, "", "golden pbstream.");
DEFINE_string(out, "", "CSV with output.");

namespace cartographer {

using mapping::SubmapId;
using transform::Rigid3d;

std::map<SubmapId, Rigid3d> GetSubmapPoses(io::ProtoStreamReader* reader) {
  std::map<SubmapId, Rigid3d> poses;

  mapping::proto::PoseGraph pose_graph_proto;
  CHECK(reader->ReadProto(&pose_graph_proto));

  for (const auto& trajectory_proto : pose_graph_proto.trajectory()) {
    for (const auto& submap_proto : trajectory_proto.submap()) {
      SubmapId submap_id{trajectory_proto.trajectory_id(),
                         submap_proto.submap_index()};
      poses[submap_id] = transform::ToRigid3(submap_proto.pose());
    }
  }
  return poses;
}

void Run() {
  LOG(INFO) << "Loading trimmed file '" << FLAGS_trimmed_pb << "'...";
  cartographer::io::ProtoStreamReader stream_trimmed(FLAGS_trimmed_pb);
  std::map<SubmapId, Rigid3d> poses_trimmed = GetSubmapPoses(&stream_trimmed);
  LOG(INFO) << "Loading golden file '" << FLAGS_golden_pb << "'...";
  cartographer::io::ProtoStreamReader stream_golden(FLAGS_golden_pb);
  std::map<SubmapId, Rigid3d> poses_golden = GetSubmapPoses(&stream_golden);
  LOG(INFO) << "Computing norms...";
  std::map<SubmapId,
           std::pair<double /* translation norm */, double /* angle diff */>>
      result;
  for (const auto& trimmed_item : poses_trimmed) {
    auto it = poses_golden.find(trimmed_item.first);
    if (it == poses_golden.end()) continue;

    result[trimmed_item.first] = std::make_pair(
        (it->second.translation() - trimmed_item.second.translation()).norm(),
        it->second.rotation().angularDistance(trimmed_item.second.rotation()));
  }

  std::unique_ptr<std::ostream> output(
      common::make_unique<std::ofstream>(FLAGS_out, std::ios_base::out));
  (*output) << "submap_id,translation_diff,angle_diff\n";
  int i = 0;
  for (const auto& item : result) {
    (*output) << i++ << "," << item.first.trajectory_id << ","
              << item.first.submap_index << "," << item.second.first << ","
              << item.second.second << std::endl;
  }
}

}  // namespace cartographer

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  cartographer::Run();
}
